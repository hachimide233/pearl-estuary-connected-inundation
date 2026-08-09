from __future__ import annotations

import argparse
import hashlib
import json
import math
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from scipy import ndimage
import shapefile


REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = REPO_ROOT / "external_data" / "coastal_hazard"
PRIVATE_INPUT_DIR = REPO_ROOT / "private_inputs" / "inundation"
DEM = PRIVATE_INPUT_DIR / "dem_cop30_aligned_to_insar.tif"
OLD_LAND_MASK = PRIVATE_INPUT_DIR / "land_mask_naturalearth10m_aligned_to_insar.tif"
SCENARIO_CSV = (
    REPO_ROOT
    / "data"
    / "derived_tables"
    / "vertical_datum"
    / "sea_level_scenarios_shek_pik_absolute_egm2008.csv"
)
OLS_SUB = {
    2050: PRIVATE_INPUT_DIR / "ols_linear_subsidence_2050_m.tif",
    2100: PRIVATE_INPUT_DIR / "ols_linear_subsidence_2100_m.tif",
}
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "connected_inundation"

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ellipsoid_strip_primitive(latitude_rad: np.ndarray) -> np.ndarray:
    """Integral of WGS84 surface-area density with respect to latitude."""
    u = np.sin(latitude_rad)
    e = math.sqrt(WGS84_E2)
    return (1.0 - WGS84_E2) * (
        u / (2.0 * (1.0 - WGS84_E2 * u * u))
        + np.arctanh(e * u) / (2.0 * e)
    )


def row_pixel_areas_km2(transform, height: int) -> np.ndarray:
    if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
        raise ValueError("Rotated grids are not supported by the row-area implementation.")
    latitude_edges = transform.f + transform.e * np.arange(height + 1, dtype=float)
    phi = np.deg2rad(latitude_edges)
    strip = np.abs(np.diff(ellipsoid_strip_primitive(phi)))
    delta_lon = abs(math.radians(transform.a))
    return WGS84_A * WGS84_A * delta_lon * strip / 1e6


def geodesic_mask_area_km2(mask: np.ndarray, row_areas_km2: np.ndarray) -> float:
    counts = np.count_nonzero(mask, axis=1).astype(float)
    return float(np.dot(counts, row_areas_km2))


def extract_gshhg_levels(zip_path: Path, output: Path, resolution: str) -> dict[int, Path]:
    output.mkdir(parents=True, exist_ok=True)
    expected_stems = {level: f"GSHHS_{resolution}_L{level}" for level in range(1, 5)}
    components = {".shp", ".shx", ".dbf", ".prj", ".cpg"}
    found: dict[int, Path] = {}
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        for level, stem in expected_stems.items():
            matches = [name for name in names if Path(name).stem.lower() == stem.lower() and Path(name).suffix.lower() in components]
            if not any(Path(name).suffix.lower() == ".shp" for name in matches):
                raise FileNotFoundError(f"{stem}.shp was not found in {zip_path}")
            for name in matches:
                target = output / Path(name).name
                if not target.exists() or target.stat().st_size == 0:
                    target.write_bytes(zf.read(name))
            found[level] = output / f"{stem}.shp"
    return found


def bbox_intersects(shape_bbox, bounds) -> bool:
    xmin, ymin, xmax, ymax = shape_bbox
    return xmax >= bounds.left and xmin <= bounds.right and ymax >= bounds.bottom and ymin <= bounds.top


def geometries_in_bounds(shp_path: Path, bounds) -> list[dict]:
    reader = shapefile.Reader(str(shp_path))
    geometries = []
    for shape in reader.iterShapes():
        if shape.bbox and bbox_intersects(shape.bbox, bounds):
            geometries.append(shape.__geo_interface__)
    return geometries


def build_land_mask(level_paths: dict[int, Path], shape, transform, bounds, all_touched: bool) -> np.ndarray:
    land = np.zeros(shape, dtype=np.uint8)
    burn_values = {1: 1, 2: 0, 3: 1, 4: 0}
    for level in range(1, 5):
        geometries = geometries_in_bounds(level_paths[level], bounds)
        if not geometries:
            continue
        layer = rasterize(
            [(geometry, burn_values[level]) for geometry in geometries],
            out_shape=shape,
            transform=transform,
            fill=255,
            dtype="uint8",
            all_touched=all_touched,
        )
        update = layer != 255
        land[update] = layer[update]
    return land.astype(bool)


def write_mask(path: Path, mask: np.ndarray, profile: dict, tags: dict[str, str]) -> None:
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw", tiled=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **out_profile) as dst:
        dst.write(mask.astype("uint8"), 1)
        dst.update_tags(**tags)


def complete_surge_rows(
    frame: pd.DataFrame, target_surges: tuple[float, ...] = (0.0, 1.0, 1.5, 2.0, 3.0)
) -> pd.DataFrame:
    rows = []
    keys = ["station", "scenario", "year"]
    for _, group in frame.groupby(keys, dropna=False):
        base = group.sort_values("surge_m").iloc[0].copy()
        slr = float(base["slr_no_vlm_m"])
        source_h0 = (
            float(base["absolute_water_level_egm2008_m"])
            - slr
            - float(base["surge_m"])
        )
        datum_ci95 = float(base["station_datum_ci95_m"])
        for target_surge in target_surges:
            matches = group[np.isclose(group["surge_m"], target_surge)]
            row = (matches.iloc[0] if not matches.empty else base).copy()
            absolute = source_h0 + slr + target_surge
            row["surge_m"] = target_surge
            row["water_level_m"] = slr + target_surge
            row["absolute_water_level_egm2008_m"] = absolute
            row["absolute_water_level_egm2008_datum_ci95_low_m"] = absolute - datum_ci95
            row["absolute_water_level_egm2008_datum_ci95_high_m"] = absolute + datum_ci95
            row["surge_row_origin"] = "verified_source" if not matches.empty else "interpolated_surge_level"
            rows.append(row)
    return pd.DataFrame(rows).sort_values(keys + ["surge_m"]).reset_index(drop=True)


def assert_grid_compatible(
    reference_shape, reference_crs, reference_transform, candidate, path: Path
) -> None:
    if reference_shape != candidate.shape:
        raise ValueError(f"Shape mismatch for {path}: {candidate.shape} != {reference_shape}")
    if candidate.crs != reference_crs:
        raise ValueError(f"CRS mismatch for {path}: {candidate.crs} != {reference_crs}")
    if not np.allclose(tuple(candidate.transform), tuple(reference_transform), rtol=0.0, atol=1e-7):
        raise ValueError(
            f"Grid transform differs by more than 1e-7 degrees for {path}: "
            f"{candidate.transform} != {reference_transform}"
        )


def safe_name(row) -> str:
    station = str(row.get("station", "STA")).replace("-", "_").replace(" ", "_")
    scenario = str(row.get("scenario", row.get("scenario_code", "SSP"))).replace(".", "p").replace("-", "m")
    surge = float(row.get("surge_m", 0.0))
    return f"{station}_{scenario}_{int(row['year'])}_surge{surge:.1f}m".replace(".", "p")


def connected_ocean_flood(surface_dem, water_level, land, valid_dem, structure):
    low_land = land & valid_dem & (surface_dem <= water_level)
    ocean = (~land) & valid_dem
    domain = ocean | low_land
    seeds = np.zeros_like(domain, dtype=bool)
    seeds[0, :] = ocean[0, :]
    seeds[-1, :] = ocean[-1, :]
    seeds[:, 0] = ocean[:, 0]
    seeds[:, -1] = ocean[:, -1]
    connected = ndimage.binary_propagation(seeds, structure=structure, mask=domain)
    return connected & low_land


def main() -> int:
    global DEM, OLD_LAND_MASK, SCENARIO_CSV, OLS_SUB

    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--dem", type=Path, default=DEM)
    parser.add_argument("--old-land-mask", type=Path, default=OLD_LAND_MASK,
                        help="Optional Natural Earth mask used only for shoreline sensitivity.")
    parser.add_argument("--scenario-csv", type=Path, default=SCENARIO_CSV)
    parser.add_argument("--subsidence-2050", type=Path, default=OLS_SUB[2050])
    parser.add_argument("--subsidence-2100", type=Path, default=OLS_SUB[2100])
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", choices=["f", "h", "i", "l", "c"], default="f")
    args = parser.parse_args()

    DEM = args.dem
    OLD_LAND_MASK = args.old_land_mask
    SCENARIO_CSV = args.scenario_csv
    OLS_SUB = {2050: args.subsidence_2050, 2100: args.subsidence_2100}
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    zip_path = args.data_root / "raw" / "gshhg" / "gshhg-shp-2.3.7.zip"
    if not zip_path.exists():
        raise FileNotFoundError(f"Run download_coastal_storm_datum_inputs.py first: {zip_path}")

    for path in [DEM, SCENARIO_CSV, *OLS_SUB.values()]:
        if not path.exists():
            raise FileNotFoundError(path)

    with rasterio.open(DEM) as ds:
        dem = ds.read(1).astype("float32")
        profile = ds.profile.copy()
        dem_shape = ds.shape
        transform = ds.transform
        bounds = ds.bounds
        crs = ds.crs
        nodata = ds.nodata
    if str(crs).upper() != "EPSG:4326":
        raise ValueError(f"Expected EPSG:4326 DEM, found {crs}")
    valid_dem = np.isfinite(dem) & ((dem != nodata) if nodata is not None else True)
    row_areas = row_pixel_areas_km2(transform, dem.shape[0])

    extracted = args.data_root / "raw" / "gshhg" / f"extracted_{args.resolution}"
    levels = extract_gshhg_levels(zip_path, extracted, args.resolution)
    print("Building GSHHG land masks...")
    land_center = build_land_mask(levels, dem.shape, transform, bounds, all_touched=False) & valid_dem
    land_touched = build_land_mask(levels, dem.shape, transform, bounds, all_touched=True) & valid_dem

    tags = {
        "source": "GSHHG 2.3.7",
        "resolution": args.resolution,
        "alignment": str(DEM),
        "note": "Generated without modifying the original Natural Earth mask.",
    }
    center_mask_path = output / f"land_mask_gshhg_{args.resolution}_center_aligned_to_insar.tif"
    touched_mask_path = output / f"land_mask_gshhg_{args.resolution}_alltouched_aligned_to_insar.tif"
    write_mask(center_mask_path, land_center, profile, {**tags, "all_touched": "false"})
    write_mask(touched_mask_path, land_touched, profile, {**tags, "all_touched": "true"})

    mask_stats = {
        "gshhg_center_land_area_km2": geodesic_mask_area_km2(land_center, row_areas),
        "gshhg_alltouched_land_area_km2": geodesic_mask_area_km2(land_touched, row_areas),
        "gshhg_mask_difference_area_km2": geodesic_mask_area_km2(land_center ^ land_touched, row_areas),
    }
    if OLD_LAND_MASK.exists():
        with rasterio.open(OLD_LAND_MASK) as ds:
            assert_grid_compatible(dem_shape, crs, transform, ds, OLD_LAND_MASK)
            old_land = (ds.read(1) > 0) & valid_dem
        mask_stats.update(
            {
                "naturalearth_land_area_km2_geodesic": geodesic_mask_area_km2(old_land, row_areas),
                "naturalearth_vs_gshhg_changed_area_km2": geodesic_mask_area_km2(old_land ^ land_center, row_areas),
                "naturalearth_land_gshhg_water_km2": geodesic_mask_area_km2(old_land & ~land_center, row_areas),
                "naturalearth_water_gshhg_land_km2": geodesic_mask_area_km2(~old_land & land_center & valid_dem, row_areas),
            }
        )
    pd.DataFrame([mask_stats]).to_csv(output / "shoreline_mask_comparison.csv", index=False, encoding="utf-8-sig")

    subsidence = {}
    subsidence_stats = {}
    for year, path in OLS_SUB.items():
        with rasterio.open(path) as ds:
            assert_grid_compatible(dem_shape, crs, transform, ds, path)
            arr = ds.read(1).astype("float32")
            arr_nodata = ds.nodata
        if arr_nodata is not None:
            arr = np.where(arr == arr_nodata, 0.0, arr)
        arr = np.maximum(np.where(np.isfinite(arr), arr, 0.0), 0.0).astype("float32")
        subsidence[year] = arr
        subsidence_stats[str(year)] = {
            "path": str(path),
            "min_m": float(np.min(arr)),
            "max_m": float(np.max(arr)),
            "mean_m": float(np.mean(arr)),
            "sign": "positive_is_subsidence",
        }

    scenarios = pd.read_csv(SCENARIO_CSV)
    numeric_columns = [
        "year",
        "slr_no_vlm_m",
        "surge_m",
        "water_level_m",
        "h0_egm2008_ar6_1995_2014_m",
        "station_datum_ci95_m",
        "absolute_water_level_egm2008_m",
        "absolute_water_level_egm2008_datum_ci95_low_m",
        "absolute_water_level_egm2008_datum_ci95_high_m",
    ]
    for column in numeric_columns:
        scenarios[column] = pd.to_numeric(scenarios[column], errors="coerce")
    scenarios = scenarios.dropna(subset=numeric_columns)
    scenarios = scenarios[scenarios["station"].astype(str).str.upper() == "SHEK_PIK"].copy()
    if scenarios.empty:
        raise ValueError("No valid SHEK_PIK rows in the absolute EGM2008 scenario table")
    scenarios["year"] = scenarios["year"].astype(int)
    source_formula_error = np.abs(
        scenarios["absolute_water_level_egm2008_m"]
        - scenarios["h0_egm2008_ar6_1995_2014_m"]
        - scenarios["slr_no_vlm_m"]
        - scenarios["surge_m"]
    )
    if float(source_formula_error.max()) > 1e-8:
        raise ValueError(
            "Absolute EGM2008 source formula failed: "
            f"max error={float(source_formula_error.max()):.12g} m"
        )
    if not scenarios["chart_datum_offset_applied_to_ar6_increment"].astype(str).str.lower().eq("false").all():
        raise ValueError("Chart Datum offset must not be applied directly to AR6 increments")
    if scenarios["h0_egm2008_ar6_1995_2014_m"].nunique() != 1:
        raise ValueError("Expected one common Shek Pik station H0 baseline")
    if scenarios["station_datum_ci95_m"].nunique() != 1:
        raise ValueError("Expected one common Shek Pik station datum uncertainty")
    scenarios = complete_surge_rows(scenarios)
    completed_formula_error = np.abs(
        scenarios["absolute_water_level_egm2008_m"]
        - scenarios["h0_egm2008_ar6_1995_2014_m"]
        - scenarios["slr_no_vlm_m"]
        - scenarios["surge_m"]
    )
    if float(completed_formula_error.max()) > 1e-8:
        raise ValueError("Completed surge rows failed the absolute EGM2008 formula")
    scenarios.to_csv(
        output / "scenario_inputs_shek_pik_egm2008_completed.csv",
        index=False,
        encoding="utf-8-sig",
    )

    structure = ndimage.generate_binary_structure(2, 2)
    out_profile = profile.copy()
    out_profile.update(driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw", tiled=True)
    records = []
    shoreline_variants = {
        "gshhg_full_center": land_center,
        "gshhg_full_alltouched_sensitivity": land_touched,
    }
    datum_variants = {
        "station_central": "absolute_water_level_egm2008_m",
        "station_datum_ci95_low": "absolute_water_level_egm2008_datum_ci95_low_m",
        "station_datum_ci95_high": "absolute_water_level_egm2008_datum_ci95_high_m",
    }
    analysis_variants = [
        ("gshhg_full_center", "station_central"),
        ("gshhg_full_center", "station_datum_ci95_low"),
        ("gshhg_full_center", "station_datum_ci95_high"),
        ("gshhg_full_alltouched_sensitivity", "station_central"),
    ]
    future_dem = {year: dem - array for year, array in subsidence.items()}
    for shoreline_name, datum_name in analysis_variants:
        land = shoreline_variants[shoreline_name]
        water_column = datum_variants[datum_name]
        for _, row in scenarios.iterrows():
            year = int(row["year"])
            if year not in subsidence:
                continue
            water_level = float(row[water_column])
            name = safe_name(row)
            base = connected_ocean_flood(dem, water_level, land, valid_dem, structure)
            ols = connected_ocean_flood(
                future_dem[year], water_level, land, valid_dem, structure
            )
            added = ols & ~base

            is_main = shoreline_name == "gshhg_full_center" and datum_name == "station_central"
            if is_main:
                for label, array in [
                    ("base_flood_connected", base),
                    ("ols_flood_connected", ols),
                    ("added_flood_connected_ols", added),
                ]:
                    target = output / f"{label}_{name}.tif"
                    with rasterio.open(target, "w", **out_profile) as dst:
                        dst.write(array.astype("uint8"), 1)
                        dst.update_tags(
                            station="SHEK_PIK",
                            water_level_reference="EGM2008",
                            water_level_m=f"{water_level:.9f}",
                            water_level_formula=(
                                "station_H0_EGM2008_1995_2014 + AR6_noVLM + surge"
                            ),
                            dem_vertical_reference=(
                                "EGM2008 inferred from Copernicus DEM product lineage"
                            ),
                            shoreline=(
                                "GSHHG 2.3.7 full resolution, pixel-center rasterization"
                            ),
                            area_method="WGS84 ellipsoidal row pixel areas",
                            connectivity="8-neighbour connected to boundary ocean",
                        )

            rec = row.to_dict()
            rec.update(
                {
                    "model": "OLS_linear_subsidence_connected_bathtub",
                    "mask_variant": shoreline_name,
                    "datum_variant": datum_name,
                    "water_level_column_used": water_column,
                    "water_level_used_egm2008_m": water_level,
                    "main_result": is_main,
                    "area_method": "WGS84_ellipsoid_exact_row_pixel_area",
                    "base_flood_area_km2": geodesic_mask_area_km2(base, row_areas),
                    "ols_flood_area_km2": geodesic_mask_area_km2(ols, row_areas),
                    "added_by_ols_subsidence_area_km2": geodesic_mask_area_km2(
                        added, row_areas
                    ),
                }
            )
            records.append(rec)
            print(shoreline_name, datum_name, name)

    summary = pd.DataFrame(records)
    summary.to_csv(
        output / "flood_area_summary_shek_pik_egm2008_all_variants.csv",
        index=False,
        encoding="utf-8-sig",
    )
    main_summary = summary[summary["main_result"]].copy()
    main_summary.to_csv(
        output / "flood_area_summary_shek_pik_egm2008_main.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary[summary["mask_variant"] == "gshhg_full_center"].to_csv(
        output / "flood_area_datum_sensitivity_shek_pik_egm2008.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary[summary["datum_variant"] == "station_central"].to_csv(
        output / "flood_area_shoreline_sensitivity_shek_pik_egm2008.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "dem": str(DEM),
        "dem_vertical_reference": "EGM2008 inferred from Copernicus DEM product lineage",
        "dem_vertical_reference_local_tag_status": "not_explicitly_tagged",
        "scenario_csv": str(SCENARIO_CSV),
        "scenario_station": "SHEK_PIK only",
        "scenario_water_level_field": "absolute_water_level_egm2008_m",
        "scenario_formula": (
            "station_H0_EGM2008_1995_2014 + AR6_noVLM_increment + surge"
        ),
        "surge_levels_m": sorted(float(value) for value in scenarios["surge_m"].unique()),
        "datum_variants": datum_variants,
        "sensitivity_design": (
            "one-factor-at-a-time: station datum interval on the main shoreline mask, "
            "and shoreline rasterization on the central station datum"
        ),
        "station_datum_ci95_m": float(scenarios["station_datum_ci95_m"].iloc[0]),
        "gshhg_zip": str(zip_path),
        "gshhg_resolution": args.resolution,
        "main_land_mask": str(center_mask_path),
        "sensitivity_land_mask": str(touched_mask_path),
        "shoreline_mask_statistics": mask_stats,
        "area_method": "WGS84 ellipsoid exact surface area for each latitude row and pixel width",
        "connectivity": "8-neighbour low-land connection to ocean seeded at raster boundaries",
        "ols_subsidence": subsidence_stats,
        "grid_alignment_tolerance_degrees": 1e-7,
        "wgs84": {"semi_major_axis_m": WGS84_A, "flattening": WGS84_F},
        "row_counts": {
            "completed_scenarios": int(len(scenarios)),
            "main": int(len(main_summary)),
            "all_sensitivity_combinations": int(len(summary)),
        },
        "source_files": {
            str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in [DEM, SCENARIO_CSV, zip_path, *OLS_SUB.values()]
        },
        "output": str(output),
        "warnings": [
            "Static connected bathtub exposure screening, not a hydrodynamic simulation.",
            "The local DEM lacks an explicit vertical-datum tag; EGM2008 is inferred from Copernicus DEM product lineage.",
            "Datum sensitivity varies the station conversion by its empirical +/-95% interval only.",
            "ERA5 products are not used in this inundation calculation.",
        ],
    }
    (output / "metadata_shek_pik_egm2008.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("Main summary:", output / "flood_area_summary_shek_pik_egm2008_main.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
