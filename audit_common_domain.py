from __future__ import annotations

import argparse
import json
import math
import zipfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import rasterio
from rasterio.features import geometry_mask
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import from_bounds


DEM: Path
LAND: Path
RATE: Path
SUB_2100: Path
WORLDCOVER: list[Path]
GADM: Path
POP_ZIP: Path
OUT: Path

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def ellipsoid_strip_primitive(latitude_rad: np.ndarray) -> np.ndarray:
    u = np.sin(latitude_rad)
    e = math.sqrt(WGS84_E2)
    return (1.0 - WGS84_E2) * (
        u / (2.0 * (1.0 - WGS84_E2 * u * u))
        + np.arctanh(e * u) / (2.0 * e)
    )


def row_pixel_areas_km2(transform, height: int) -> np.ndarray:
    latitude_edges = transform.f + transform.e * np.arange(height + 1, dtype=float)
    strip = np.abs(np.diff(ellipsoid_strip_primitive(np.deg2rad(latitude_edges))))
    return WGS84_A**2 * abs(math.radians(transform.a)) * strip / 1e6


def area(mask: np.ndarray, row_area: np.ndarray) -> float:
    return float(np.dot(np.count_nonzero(mask, axis=1), row_area))


def valid(ds: rasterio.io.DatasetReader, array: np.ndarray) -> np.ndarray:
    result = np.isfinite(array)
    if ds.nodata is not None:
        result &= array != ds.nodata
    return result


def reproject_class_fraction(path: Path, class_value: int, profile: dict) -> np.ndarray:
    with rasterio.open(path) as src:
        src_data = (src.read(1) == class_value).astype("float32")
        dst = np.zeros((profile["height"], profile["width"]), dtype="float32")
        reproject(
            src_data,
            dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=profile["transform"],
            dst_crs=profile["crs"],
            resampling=Resampling.average,
            src_nodata=None,
            dst_nodata=0.0,
        )
    return np.clip(dst, 0.0, 1.0)


def extract_population_2020() -> Path:
    target = OUT / "SSP2_2020.tif"
    if target.exists():
        return target
    with zipfile.ZipFile(POP_ZIP) as zf:
        matches = [n for n in zf.namelist() if n.replace("\\", "/").lower().endswith("ssp2_2020.tif")]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one SSP2_2020.tif in {POP_ZIP}; found {matches}")
        target.write_bytes(zf.read(matches[0]))
    return target


def population_sum(mask: np.ndarray, reference, population_path: Path) -> float:
    with rasterio.open(population_path) as pop:
        pop_values = pop.read(1, masked=True).astype("float64").filled(np.nan)
        pop_values[~np.isfinite(pop_values) | (pop_values < 0)] = np.nan
        fraction = np.zeros(pop.shape, dtype="float32")
        reproject(
            mask.astype("float32"),
            fraction,
            src_transform=reference.transform,
            src_crs=reference.crs,
            dst_transform=pop.transform,
            dst_crs=pop.crs,
            resampling=Resampling.average,
            src_nodata=None,
            dst_nodata=0.0,
        )
        return float(np.nansum(pop_values * np.clip(fraction, 0.0, 1.0)))


def gadm_layer() -> str:
    layers = list(gpd.io.file.fiona.listlayers(GADM)) if getattr(gpd.io.file, "fiona", None) else []
    if not layers:
        import pyogrio

        layers = [row[0] for row in pyogrio.list_layers(GADM)]
    preferred = [x for x in layers if x.endswith("_2") or x.upper().endswith("ADM_2")]
    return preferred[0] if preferred else layers[-1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit the common analysis domain and its representativeness."
    )
    parser.add_argument("--dem", type=Path, required=True, help="Aligned Copernicus DEM raster.")
    parser.add_argument("--land-mask", type=Path, required=True, help="Aligned GSHHG land-mask raster.")
    parser.add_argument("--rate", type=Path, required=True, help="OLS subsidence-rate raster.")
    parser.add_argument("--subsidence-2100", type=Path, required=True, help="OLS 2100 cumulative-subsidence raster.")
    parser.add_argument("--worldcover", type=Path, nargs="+", required=True, help="One or more ESA WorldCover tiles.")
    parser.add_argument("--gadm", type=Path, required=True, help="GADM GeoPackage.")
    parser.add_argument("--population-zip", type=Path, required=True, help="SSP2 population ZIP archive.")
    parser.add_argument("--out", type=Path, default=Path("common_domain_audit"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    global DEM, LAND, RATE, SUB_2100, WORLDCOVER, GADM, POP_ZIP, OUT
    args = parse_args()
    DEM = args.dem
    LAND = args.land_mask
    RATE = args.rate
    SUB_2100 = args.subsidence_2100
    WORLDCOVER = args.worldcover
    GADM = args.gadm
    POP_ZIP = args.population_zip
    OUT = args.out

    OUT.mkdir(parents=True, exist_ok=True)
    required = [DEM, LAND, RATE, SUB_2100, GADM, POP_ZIP, *WORLDCOVER]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)

    with rasterio.open(DEM) as dem_ds:
        dem = dem_ds.read(1)
        dem_valid = valid(dem_ds, dem)
        profile = {
            "height": dem_ds.height,
            "width": dem_ds.width,
            "transform": dem_ds.transform,
            "crs": dem_ds.crs,
        }
        rows = row_pixel_areas_km2(dem_ds.transform, dem_ds.height)
        dem_meta = {
            "shape": list(dem_ds.shape),
            "crs": str(dem_ds.crs),
            "nodata": dem_ds.nodata,
            "bounds": list(dem_ds.bounds),
            "transform": list(dem_ds.transform),
        }

        with rasterio.open(LAND) as land_ds:
            land = land_ds.read(1) > 0
        with rasterio.open(RATE) as rate_ds:
            rate = rate_ds.read(1)
            rate_valid = valid(rate_ds, rate)
            rate_meta = {
                "nodata": rate_ds.nodata,
                "dtype": rate_ds.dtypes[0],
                "min_valid": float(np.min(rate[rate_valid])),
                "max_valid": float(np.max(rate[rate_valid])),
            }
        with rasterio.open(SUB_2100) as sub_ds:
            sub = sub_ds.read(1)
            sub_valid = valid(sub_ds, sub)
            sub_meta = {
                "nodata": sub_ds.nodata,
                "dtype": sub_ds.dtypes[0],
                "min_valid": float(np.min(sub[sub_valid])),
                "max_valid": float(np.max(sub[sub_valid])),
            }

        grid = dem_valid
        dem_land = dem_valid & land
        insar_support = rate_valid & sub_valid
        common_land = dem_land & insar_support
        excluded_land = dem_land & ~insar_support

        masks = {
            "full_grid_valid": grid,
            "dem_valid_land": dem_land,
            "insar_support_all_cells": insar_support,
            "common_analysis_land": common_land,
            "excluded_dem_land": excluded_land,
        }
        summary = []
        for name, mask in masks.items():
            summary.append({"domain": name, "pixels": int(mask.sum()), "area_km2": area(mask, rows)})

        built_fraction = np.maximum.reduce(
            [reproject_class_fraction(path, 50, profile) for path in WORLDCOVER]
        )
        built_total_km2 = float(np.dot(np.sum(built_fraction * dem_land, axis=1), rows))
        built_common_km2 = float(np.dot(np.sum(built_fraction * common_land, axis=1), rows))

        pop_2020 = extract_population_2020()
        pop_land = population_sum(dem_land, dem_ds, pop_2020)
        pop_common = population_sum(common_land, dem_ds, pop_2020)

        layer = gadm_layer()
        admin = gpd.read_file(GADM, layer=layer, bbox=tuple(dem_ds.bounds)).to_crs(dem_ds.crs)
        name_col = next((c for c in ["NAME_2", "NL_NAME_2", "NAME_1", "GID_2", "GID_1"] if c in admin.columns), None)
        admin_rows = []
        for idx, row in admin.iterrows():
            geom = row.geometry
            if geom is None or geom.is_empty:
                continue
            inside = geometry_mask([geom], dem_ds.shape, dem_ds.transform, invert=True, all_touched=False)
            admin_land = dem_land & inside
            admin_common = common_land & inside
            land_area = area(admin_land, rows)
            if land_area <= 0:
                continue
            admin_rows.append(
                {
                    "administrative_unit": str(row[name_col]) if name_col else str(idx),
                    "gadm_layer": layer,
                    "land_area_km2": land_area,
                    "common_area_km2": area(admin_common, rows),
                    "common_share_percent": 100.0 * area(admin_common, rows) / land_area,
                }
            )

    import pandas as pd

    summary_df = pd.DataFrame(summary)
    land_area = float(summary_df.loc[summary_df.domain == "dem_valid_land", "area_km2"].iloc[0])
    common_area = float(summary_df.loc[summary_df.domain == "common_analysis_land", "area_km2"].iloc[0])
    summary_df["share_of_dem_land_percent"] = 100.0 * summary_df["area_km2"] / land_area
    summary_df.to_csv(OUT / "common_domain_summary.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(admin_rows).sort_values("land_area_km2", ascending=False).to_csv(
        OUT / "common_domain_administrative_coverage.csv", index=False, encoding="utf-8-sig"
    )

    report = {
        "dem": dem_meta,
        "rate": rate_meta,
        "sub_2100": sub_meta,
        "common_domain": {
            "area_km2": common_area,
            "share_of_dem_valid_land_percent": 100.0 * common_area / land_area,
            "built_up_area_km2": built_common_km2,
            "share_of_built_up_area_percent": 100.0 * built_common_km2 / built_total_km2,
            "population_2020": pop_common,
            "share_of_population_2020_percent": 100.0 * pop_common / pop_land,
        },
        "reference_domain": {
            "dem_valid_land_area_km2": land_area,
            "built_up_area_km2": built_total_km2,
            "population_2020": pop_land,
        },
        "definitions": {
            "insar_support": "finite and non-nodata in both OLS rate and OLS 2100 cumulative-subsidence rasters",
            "common_analysis_land": "GSHHG land AND valid DEM AND InSAR support",
            "built_up": "ESA WorldCover 2021 class 50, fractional coverage after average resampling",
            "population": "SSP2 2020 grid, area-weighted overlap",
        },
    }
    (OUT / "common_domain_audit.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(pd.DataFrame(admin_rows).sort_values("land_area_km2", ascending=False).to_string(index=False))


if __name__ == "__main__":
    main()
