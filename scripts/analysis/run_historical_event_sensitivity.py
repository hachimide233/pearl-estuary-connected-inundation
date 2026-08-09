#!/usr/bin/env python3
"""Rebuild 11 historical-event flood scenarios on a Shek Pik EGM2008 datum."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy import ndimage


REPO_ROOT = Path(__file__).resolve().parents[2]
PRIVATE_INPUT_DIR = REPO_ROOT / "private_inputs" / "inundation"
DEM = PRIVATE_INPUT_DIR / "dem_cop30_aligned_to_insar.tif"
LAND_MASK = PRIVATE_INPUT_DIR / "land_mask_gshhg_f_center_aligned_to_insar.tif"
EVENTS = (
    REPO_ROOT
    / "external_data"
    / "coastal_hazard"
    / "processed"
    / "nested_archive_recovery"
    / "storm_event_water_level_summary_recovered.csv"
)
SEA_LEVEL = (
    REPO_ROOT
    / "data"
    / "derived_tables"
    / "vertical_datum"
    / "sea_level_scenarios_shek_pik_absolute_egm2008.csv"
)
POPULATION_2025 = REPO_ROOT / "external_data" / "population" / "chn_pop_2025_CN_100m_R2025A_v1.tif"
OLS_SUBSIDENCE = {
    2050: PRIVATE_INPUT_DIR / "ols_linear_subsidence_2050_m.tif",
    2100: PRIVATE_INPUT_DIR / "ols_linear_subsidence_2100_m.tif",
}
DEFAULT_OUTPUT = REPO_ROOT / "outputs" / "historical_event_sensitivity"

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EVENT_NAME_ZH = {"MANGKHUT": "山竹", "HATO": "天鸽", "SAOLA": "苏拉"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ellipsoid_strip_primitive(latitude_rad: np.ndarray) -> np.ndarray:
    u = np.sin(latitude_rad)
    e = math.sqrt(WGS84_E2)
    return (1.0 - WGS84_E2) * (
        u / (2.0 * (1.0 - WGS84_E2 * u * u))
        + np.arctanh(e * u) / (2.0 * e)
    )


def row_pixel_areas_km2(transform, height: int) -> np.ndarray:
    if abs(transform.b) > 1e-12 or abs(transform.d) > 1e-12:
        raise ValueError("Rotated grids are not supported")
    latitude_edges = transform.f + transform.e * np.arange(height + 1, dtype=float)
    strip = np.abs(np.diff(ellipsoid_strip_primitive(np.deg2rad(latitude_edges))))
    return WGS84_A * WGS84_A * abs(math.radians(transform.a)) * strip / 1e6


def mask_area_km2(mask: np.ndarray, row_areas: np.ndarray) -> float:
    return float(np.dot(np.count_nonzero(mask, axis=1), row_areas))


def assert_grid(reference, candidate, path: Path) -> None:
    if reference.shape != candidate.shape:
        raise ValueError(f"Shape mismatch for {path}")
    if reference.crs != candidate.crs:
        raise ValueError(f"CRS mismatch for {path}")
    if not np.allclose(
        tuple(reference.transform), tuple(candidate.transform), rtol=0.0, atol=1e-7
    ):
        raise ValueError(f"Transform mismatch for {path}")


def connected_ocean_flood(
    surface_dem: np.ndarray,
    water_level: float,
    land: np.ndarray,
    valid_dem: np.ndarray,
    structure: np.ndarray,
) -> np.ndarray:
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


def safe_name(row: pd.Series) -> str:
    event = str(row["NAME"]).upper().replace("-", "_").replace(" ", "_")
    scenario = str(row["scenario"]).replace("-", "m").replace(".", "p")
    return f"event_{event}_{scenario}_{int(row['year'])}"


def clamp_window(window: Window, width: int, height: int) -> Window | None:
    col0 = max(0, int(math.floor(window.col_off)))
    row0 = max(0, int(math.floor(window.row_off)))
    col1 = min(width, int(math.ceil(window.col_off + window.width)))
    row1 = min(height, int(math.ceil(window.row_off + window.height)))
    if col1 <= col0 or row1 <= row0:
        return None
    return Window(col0, row0, col1 - col0, row1 - row0)


def population_window(population, mask_reference) -> Window | None:
    bounds = mask_reference.bounds
    if mask_reference.crs != population.crs:
        bounds = transform_bounds(
            mask_reference.crs, population.crs, *bounds, densify_pts=21
        )
    return clamp_window(
        from_bounds(*bounds, transform=population.transform),
        population.width,
        population.height,
    )


def population_exposure(
    mask: np.ndarray,
    source_transform,
    source_crs,
    population,
    window: Window,
    population_values: np.ndarray,
) -> tuple[float, float]:
    shape = (int(window.height), int(window.width))
    average = np.zeros(shape, dtype="float32")
    nearest = np.zeros(shape, dtype="float32")
    common = {
        "source": mask.astype("float32", copy=False),
        "src_transform": source_transform,
        "src_crs": source_crs,
        "src_nodata": None,
        "dst_transform": population.window_transform(window),
        "dst_crs": population.crs,
        "dst_nodata": 0.0,
    }
    reproject(destination=average, resampling=Resampling.average, **common)
    reproject(destination=nearest, resampling=Resampling.nearest, **common)
    average = np.clip(average, 0.0, 1.0)
    valid = np.isfinite(population_values)
    weighted_total = float(np.sum(np.where(valid, population_values * average, 0.0)))
    nearest_total = float(
        np.sum(np.where(valid & (nearest > 0.5), population_values, 0.0))
    )
    return weighted_total, nearest_total


def build_scenarios() -> tuple[pd.DataFrame, float, float]:
    events = pd.read_csv(EVENTS)
    event_numeric = [
        "max_observed_total_level_m_cd",
        "predicted_tide_at_observed_peak_m_cd",
        "surge_residual_at_observed_peak_m",
    ]
    for column in event_numeric:
        events[column] = pd.to_numeric(events[column], errors="coerce")
    events = events[
        events["status"].eq("observed_and_predicted_available")
        & events[event_numeric].notna().all(axis=1)
    ].copy()
    events = events.drop_duplicates("NAME")
    if len(events) != 11:
        raise ValueError(f"Expected 11 valid unique events, found {len(events)}")
    component_error = np.abs(
        events["max_observed_total_level_m_cd"]
        - events["predicted_tide_at_observed_peak_m_cd"]
        - events["surge_residual_at_observed_peak_m"]
    )
    if float(component_error.max()) > 1e-8:
        raise ValueError("Observed total level does not equal tide plus surge residual")

    sea = pd.read_csv(SEA_LEVEL)
    for column in [
        "year",
        "surge_m",
        "slr_no_vlm_m",
        "chart_datum_to_egm2008_m",
        "station_datum_ci95_m",
    ]:
        sea[column] = pd.to_numeric(sea[column], errors="coerce")
    sea = sea[
        sea["station"].astype(str).str.upper().eq("SHEK_PIK")
        & sea["scenario"].isin(["SSP2-4.5", "SSP5-8.5"])
        & sea["year"].isin([2050, 2100])
        & np.isclose(sea["surge_m"], 0.0)
    ].copy()
    sea = sea.drop_duplicates(["scenario", "year"])
    if len(sea) != 4:
        raise ValueError(f"Expected four no-surge sea-level rows, found {len(sea)}")
    if sea["chart_datum_to_egm2008_m"].nunique() != 1:
        raise ValueError("Expected one Shek Pik Chart Datum to EGM2008 offset")
    if sea["station_datum_ci95_m"].nunique() != 1:
        raise ValueError("Expected one Shek Pik datum uncertainty")
    offset = float(sea["chart_datum_to_egm2008_m"].iloc[0])
    datum_ci95 = float(sea["station_datum_ci95_m"].iloc[0])

    rows = []
    for _, event in events.iterrows():
        for _, climate in sea.iterrows():
            row = event.to_dict()
            row.update(climate.to_dict())
            row["source_station"] = "SHEK_PIK"
            row["historical_total_level_egm2008_m"] = (
                float(event["max_observed_total_level_m_cd"]) + offset
            )
            row["historical_tide_egm2008_m"] = (
                float(event["predicted_tide_at_observed_peak_m_cd"]) + offset
            )
            row["future_total_level_egm2008_m"] = (
                row["historical_total_level_egm2008_m"]
                + float(climate["slr_no_vlm_m"])
            )
            row["future_total_level_egm2008_ci95_low_m"] = (
                row["future_total_level_egm2008_m"] - datum_ci95
            )
            row["future_total_level_egm2008_ci95_high_m"] = (
                row["future_total_level_egm2008_m"] + datum_ci95
            )
            row["event_boundary_formula"] = (
                "max_observed_total_level_m_cd + chart_datum_to_egm2008_m "
                "+ slr_no_vlm_m"
            )
            row["storm_surge_added_again"] = False
            rows.append(row)
    scenarios = pd.DataFrame(rows).sort_values(
        ["max_observed_total_level_m_cd", "NAME", "scenario", "year"],
        ascending=[False, True, True, True],
    )
    if len(scenarios) != 44:
        raise AssertionError(f"Expected 44 event scenarios, found {len(scenarios)}")
    formula_error = np.abs(
        scenarios["future_total_level_egm2008_m"]
        - scenarios["max_observed_total_level_m_cd"]
        - scenarios["chart_datum_to_egm2008_m"]
        - scenarios["slr_no_vlm_m"]
    )
    if float(formula_error.max()) > 1e-8:
        raise AssertionError("Future event-water formula did not close")
    return scenarios, offset, datum_ci95


def main() -> int:
    global DEM, LAND_MASK, EVENTS, SEA_LEVEL, POPULATION_2025, OLS_SUBSIDENCE

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dem", type=Path, default=DEM)
    parser.add_argument("--land-mask", type=Path, default=LAND_MASK)
    parser.add_argument("--events", type=Path, default=EVENTS)
    parser.add_argument("--sea-level", type=Path, default=SEA_LEVEL)
    parser.add_argument("--population-2025", type=Path, default=POPULATION_2025)
    parser.add_argument("--subsidence-2050", type=Path, default=OLS_SUBSIDENCE[2050])
    parser.add_argument("--subsidence-2100", type=Path, default=OLS_SUBSIDENCE[2100])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    DEM = args.dem
    LAND_MASK = args.land_mask
    EVENTS = args.events
    SEA_LEVEL = args.sea_level
    POPULATION_2025 = args.population_2025
    OLS_SUBSIDENCE = {2050: args.subsidence_2050, 2100: args.subsidence_2100}
    args.out.mkdir(parents=True, exist_ok=True)

    inputs = [DEM, LAND_MASK, EVENTS, SEA_LEVEL, POPULATION_2025, *OLS_SUBSIDENCE.values()]
    for path in inputs:
        if not path.is_file():
            raise FileNotFoundError(path)

    scenarios, datum_offset, datum_ci95 = build_scenarios()
    scenario_path = args.out / "all_11_events_shek_pik_egm2008_scenarios.csv"
    scenarios.to_csv(scenario_path, index=False, encoding="utf-8-sig")

    with rasterio.open(DEM) as reference:
        dem = reference.read(1).astype("float32")
        profile = reference.profile.copy()
        transform = reference.transform
        crs = reference.crs
        nodata = reference.nodata
        row_areas = row_pixel_areas_km2(transform, reference.height)
        if str(crs).upper() != "EPSG:4326":
            raise ValueError(f"Expected EPSG:4326 DEM, found {crs}")
        valid_dem = np.isfinite(dem) & ((dem != nodata) if nodata is not None else True)

        with rasterio.open(LAND_MASK) as source:
            assert_grid(reference, source, LAND_MASK)
            land = (source.read(1) > 0) & valid_dem

        subsidence = {}
        for year, path in OLS_SUBSIDENCE.items():
            with rasterio.open(path) as source:
                assert_grid(reference, source, path)
                array = source.read(1).astype("float32")
                source_nodata = source.nodata
            if source_nodata is not None:
                array = np.where(array == source_nodata, 0.0, array)
            subsidence[year] = np.maximum(
                np.where(np.isfinite(array), array, 0.0), 0.0
            ).astype("float32")

        with rasterio.open(POPULATION_2025) as population:
            window = population_window(population, reference)
            if window is None:
                raise ValueError("Population raster does not overlap the flood grid")
            population_values = population.read(1, window=window, masked=True).astype(
                "float64"
            ).filled(np.nan)
            population_values[~np.isfinite(population_values) | (population_values < 0)] = np.nan

            structure = ndimage.generate_binary_structure(2, 2)
            out_profile = profile.copy()
            out_profile.update(
                driver="GTiff", dtype="uint8", count=1, nodata=0, compress="lzw", tiled=True
            )
            records = []
            for number, (_, row) in enumerate(scenarios.iterrows(), 1):
                year = int(row["year"])
                water_level = float(row["future_total_level_egm2008_m"])
                name = safe_name(row)
                base = connected_ocean_flood(
                    dem, water_level, land, valid_dem, structure
                )
                ols = connected_ocean_flood(
                    dem - subsidence[year], water_level, land, valid_dem, structure
                )
                added = ols & ~base
                masks = {"base": base, "ols": ols, "added": added}
                paths = {}
                exposure = {}
                for mask_type, mask in masks.items():
                    path = args.out / f"{mask_type}_flood_connected_{name}.tif"
                    with rasterio.open(path, "w", **out_profile) as destination:
                        destination.write(mask.astype("uint8"), 1)
                        destination.update_tags(
                            vertical_datum="EGM2008",
                            source_station="SHEK_PIK",
                            event_name=str(row["NAME"]),
                            target_year=str(year),
                            water_level_egm2008_m=f"{water_level:.9f}",
                            boundary_formula=(
                                "H_total_CD + chart_datum_to_egm2008 + AR6_noVLM"
                            ),
                            model="connected_bathtub_8_neighbour",
                        )
                    paths[mask_type] = str(path)
                    exposure[mask_type] = population_exposure(
                        mask,
                        transform,
                        crs,
                        population,
                        window,
                        population_values,
                    )

                record = row.to_dict()
                record.update(
                    {
                        "model": "OLS_linear_subsidence_connected_bathtub",
                        "land_mask": "GSHHG_2.3.7_full_resolution_pixel_center",
                        "base_mask": paths["base"],
                        "ols_mask": paths["ols"],
                        "added_mask": paths["added"],
                        "base_flood_area_km2": mask_area_km2(base, row_areas),
                        "ols_flood_area_km2": mask_area_km2(ols, row_areas),
                        "added_by_ols_subsidence_area_km2": mask_area_km2(added, row_areas),
                        "population_year": 2025,
                        "population_method": "fractional_flooded_area_per_population_cell",
                        "base_exposed_population": exposure["base"][0],
                        "ols_exposed_population": exposure["ols"][0],
                        "added_exposed_population": exposure["added"][0],
                        "nearest_base_population_sensitivity": exposure["base"][1],
                        "nearest_ols_population_sensitivity": exposure["ols"][1],
                        "nearest_added_population_sensitivity": exposure["added"][1],
                    }
                )
                records.append(record)
                print(f"{number}/44 {name} H={water_level:.6f} m", flush=True)

    summary = pd.DataFrame(records)
    summary["area_closure_error_km2"] = (
        summary["ols_flood_area_km2"]
        - summary["base_flood_area_km2"]
        - summary["added_by_ols_subsidence_area_km2"]
    )
    summary["population_closure_error"] = (
        summary["ols_exposed_population"]
        - summary["base_exposed_population"]
        - summary["added_exposed_population"]
    )
    if (summary["ols_flood_area_km2"] + 1e-10 < summary["base_flood_area_km2"]).any():
        raise AssertionError("OLS flood area is below base area")
    if float(summary["area_closure_error_km2"].abs().max()) > 1e-6:
        raise AssertionError("Area results do not close")
    if float(summary["population_closure_error"].abs().max()) > 0.01:
        raise AssertionError("Population results do not close")

    summary_path = args.out / "absolute_event_inundation_population_summary_shek_pik_egm2008.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    representative = summary[summary["NAME"].isin(EVENT_NAME_ZH)].copy()
    representative["event_name_zh"] = representative["NAME"].map(EVENT_NAME_ZH)
    representative = representative.sort_values(
        ["max_observed_total_level_m_cd", "scenario", "year"],
        ascending=[False, True, True],
    )
    representative_columns = [
        "event_name_zh",
        "NAME",
        "scenario",
        "year",
        "max_observed_total_level_m_cd",
        "future_total_level_egm2008_m",
        "base_flood_area_km2",
        "ols_flood_area_km2",
        "added_by_ols_subsidence_area_km2",
        "base_exposed_population",
        "ols_exposed_population",
        "added_exposed_population",
    ]
    representative_path = args.out / "absolute_event_scenarios_manuscript_table_shek_pik_egm2008.csv"
    representative[representative_columns].to_csv(
        representative_path, index=False, encoding="utf-8-sig"
    )

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "scenario_count": len(summary),
        "event_count": int(summary["NAME"].nunique()),
        "vertical_datum": "EGM2008",
        "dem_vertical_reference": "EGM2008 inferred from Copernicus DEM product lineage",
        "dem_vertical_reference_local_tag_status": "not_explicitly_tagged",
        "chart_datum_to_egm2008_m": datum_offset,
        "station_datum_ci95_m": datum_ci95,
        "boundary_formula": (
            "H_event_EGM2008 = H_observed_total_CD + chart_datum_to_EGM2008 "
            "+ AR6_noVLM_increment"
        ),
        "anti_double_counting": [
            "Observed total water level already contains astronomical tide and storm surge.",
            "Storm surge is not added again.",
            "Subsidence is represented only by lowering the future DEM.",
        ],
        "area_method": "WGS84 ellipsoid exact row-pixel surface area",
        "population_main_method": (
            "WorldPop 2025 cell population multiplied by average flooded-area fraction"
        ),
        "population_sensitivity_method": "nearest-neighbour full-cell counting",
        "max_area_closure_error_km2": float(summary["area_closure_error_km2"].abs().max()),
        "max_population_closure_error": float(summary["population_closure_error"].abs().max()),
        "water_level_range_egm2008_m": [
            float(summary["future_total_level_egm2008_m"].min()),
            float(summary["future_total_level_egm2008_m"].max()),
        ],
        "source_files": {
            str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in inputs
        },
        "outputs": {
            "scenarios": str(scenario_path),
            "summary": str(summary_path),
            "representative_table": str(representative_path),
        },
        "warnings": [
            "Static connected bathtub screening, not a hydrodynamic event simulation.",
            "The Shek Pik peak is applied uniformly across the flood-model domain.",
            "The local DEM lacks an explicit vertical-datum tag.",
            "ERA5 products are not used in this inundation calculation.",
        ],
    }
    (args.out / "metadata_absolute_events_shek_pik_egm2008.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary[[
        "NAME",
        "scenario",
        "year",
        "future_total_level_egm2008_m",
        "base_flood_area_km2",
        "ols_flood_area_km2",
        "base_exposed_population",
        "ols_exposed_population",
    ]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
