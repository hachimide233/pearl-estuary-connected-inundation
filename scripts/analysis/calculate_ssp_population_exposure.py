#!/usr/bin/env python3
"""Area-weighted SSP population exposure for Shek Pik EGM2008 flood masks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ARCHIVE_DIR = REPO_ROOT / "external_data" / "population" / "wang_ssp_archives"
DEFAULT_MASK_DIR = REPO_ROOT / "private_inputs" / "connected_inundation_masks"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "outputs" / "population_exposure"
DEFAULT_EXTRACT_DIR = REPO_ROOT / "external_data" / "population" / "wang_ssp_extracted"
YEARS = (2050, 2100)
POPULATION_SCENARIOS = ("SSP2", "SSP5")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_required(archive: Path, scenario: str, extract_dir: Path) -> dict[int, Path]:
    found: dict[int, Path] = {}
    with zipfile.ZipFile(archive) as zf:
        for member in zf.namelist():
            normalized = member.replace("\\", "/")
            match = re.search(rf"{scenario}_(2050|2100)\.tif$", normalized, re.I)
            if not match:
                continue
            year = int(match.group(1))
            target = extract_dir / scenario / f"{scenario}_{year}.tif"
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists() or target.stat().st_size == 0:
                with zf.open(member) as source, target.open("wb") as destination:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        destination.write(block)
            found[year] = target
    missing = sorted(set(YEARS) - set(found))
    if missing:
        raise FileNotFoundError(f"{scenario} archive is missing years: {missing}")
    return found


def parse_mask_name(path: Path) -> dict[str, str | int | float]:
    match = re.fullmatch(
        r"base_flood_connected_(?P<station>SHEK_PIK)_"
        r"(?P<scenario>SSP(?P<ssp>[25])m\d+p\d+)_(?P<year>\d{4})_"
        r"surge(?P<surge>\d+p\d+)m\.tif",
        path.name,
        flags=re.I,
    )
    if not match:
        raise ValueError(f"Unexpected mask filename: {path.name}")
    values = match.groupdict()
    return {
        "station": values["station"].upper(),
        "water_scenario_file": values["scenario"],
        "population_scenario": f"SSP{values['ssp']}",
        "target_year": int(values["year"]),
        "surge_m": float(values["surge"].replace("p", ".")),
    }


def mask_cases(mask_dir: Path) -> list[dict[str, object]]:
    cases = []
    for base in sorted(mask_dir.glob("base_flood_connected_SHEK_PIK_*.tif")):
        suffix = base.name.removeprefix("base_flood_connected_")
        ols = mask_dir / f"ols_flood_connected_{suffix}"
        added = mask_dir / f"added_flood_connected_ols_{suffix}"
        if not (ols.is_file() and added.is_file()):
            raise FileNotFoundError(f"Incomplete mask triplet for {suffix}")
        cases.append({"base": base, "ols": ols, "added": added, **parse_mask_name(base)})
    if not cases:
        raise RuntimeError(f"No Shek Pik EGM2008 mask triplets found in {mask_dir}")
    return cases


def clamp_window(window: Window, width: int, height: int) -> Window | None:
    col_start = max(0, int(math.floor(window.col_off)))
    row_start = max(0, int(math.floor(window.row_off)))
    col_stop = min(width, int(math.ceil(window.col_off + window.width)))
    row_stop = min(height, int(math.ceil(window.row_off + window.height)))
    if col_stop <= col_start or row_stop <= row_start:
        return None
    return Window(col_start, row_start, col_stop - col_start, row_stop - row_start)


def population_window(population, mask) -> Window | None:
    bounds = mask.bounds
    if mask.crs != population.crs:
        bounds = transform_bounds(mask.crs, population.crs, *bounds, densify_pts=21)
    return clamp_window(
        from_bounds(*bounds, transform=population.transform),
        population.width,
        population.height,
    )


def read_population(population, window: Window) -> np.ndarray:
    values = population.read(1, window=window, masked=True).astype("float64").filled(np.nan)
    values[~np.isfinite(values) | (values < 0)] = np.nan
    return values


def reproject_mask_fraction(
    path: Path, population, window: Window, resampling: Resampling
) -> np.ndarray:
    with rasterio.open(path) as source:
        source_mask = (source.read(1) > 0).astype("float32")
        source_transform = source.transform
        source_crs = source.crs
    destination = np.zeros((int(window.height), int(window.width)), dtype="float32")
    reproject(
        source=source_mask,
        destination=destination,
        src_transform=source_transform,
        src_crs=source_crs,
        src_nodata=None,
        dst_transform=population.window_transform(window),
        dst_crs=population.crs,
        dst_nodata=0.0,
        resampling=resampling,
    )
    return np.clip(destination, 0.0, 1.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive-dir", type=Path, default=DEFAULT_ARCHIVE_DIR)
    parser.add_argument("--mask-dir", type=Path, default=DEFAULT_MASK_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    population_files = {}
    archives = []
    for scenario in POPULATION_SCENARIOS:
        archive = args.archive_dir / f"{scenario}.zip"
        if not archive.is_file():
            raise FileNotFoundError(archive)
        archives.append(archive)
        for year, path in extract_required(archive, scenario, args.extract_dir).items():
            population_files[(scenario, year)] = path

    cases = mask_cases(args.mask_dir)
    rows = []
    for index, case in enumerate(cases, 1):
        key = (str(case["population_scenario"]), int(case["target_year"]))
        population_path = population_files[key]
        with rasterio.open(population_path) as population:
            with rasterio.open(case["base"]) as reference_mask:
                window = population_window(population, reference_mask)
            if window is None:
                raise ValueError(f"No population overlap for {case['base']}")
            population_values = read_population(population, window)
            valid_population = np.isfinite(population_values)
            for mask_type in ("base", "ols", "added"):
                average_fraction = reproject_mask_fraction(
                    Path(case[mask_type]), population, window, Resampling.average
                )
                nearest_fraction = reproject_mask_fraction(
                    Path(case[mask_type]), population, window, Resampling.nearest
                )
                weighted = np.where(valid_population, population_values * average_fraction, 0.0)
                nearest = np.where(valid_population & (nearest_fraction > 0.5), population_values, 0.0)
                rows.append(
                    {
                        "station": case["station"],
                        "water_scenario_file": case["water_scenario_file"],
                        "population_scenario": case["population_scenario"],
                        "target_year": case["target_year"],
                        "surge_m": case["surge_m"],
                        "mask_type": mask_type,
                        "mask_file": Path(case[mask_type]).name,
                        "population_file": str(population_path),
                        "area_weighted_exposed_population": float(np.sum(weighted)),
                        "area_weighted_exposed_population_million": float(np.sum(weighted)) / 1e6,
                        "nearest_exposed_population_sensitivity": float(np.sum(nearest)),
                        "nearest_exposed_population_sensitivity_million": float(np.sum(nearest)) / 1e6,
                        "nearest_minus_area_weighted_population": float(np.sum(nearest) - np.sum(weighted)),
                        "fractional_population_cells": int(
                            np.count_nonzero(
                                valid_population
                                & (average_fraction > 0.0)
                                & (average_fraction < 1.0)
                            )
                        ),
                    }
                )
        print(f"{index}/{len(cases)} {Path(case['base']).name}", flush=True)

    long = pd.DataFrame(rows)
    long_path = args.out_dir / "population_exposure_shek_pik_egm2008_long.csv"
    long.to_csv(long_path, index=False, encoding="utf-8-sig")
    index_columns = [
        "station",
        "water_scenario_file",
        "population_scenario",
        "target_year",
        "surge_m",
    ]
    summary = long.pivot_table(
        index=index_columns,
        columns="mask_type",
        values=[
            "area_weighted_exposed_population_million",
            "nearest_exposed_population_sensitivity_million",
        ],
        aggfunc="first",
    )
    summary.columns = [f"{metric}_{mask}" for metric, mask in summary.columns]
    summary = summary.reset_index()
    summary["area_weighted_added_closure_error_million"] = (
        summary["area_weighted_exposed_population_million_ols"]
        - summary["area_weighted_exposed_population_million_base"]
        - summary["area_weighted_exposed_population_million_added"]
    )
    if float(summary["area_weighted_added_closure_error_million"].abs().max()) > 1e-5:
        raise AssertionError("Area-weighted base/OLS/added population does not close")
    summary_path = args.out_dir / "population_exposure_shek_pik_egm2008_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")

    metadata = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "main_method": (
            "Flood-mask area fraction in each 1-km population cell using GDAL average resampling; "
            "cell population multiplied by that fraction."
        ),
        "sensitivity_method": "Nearest-neighbour full-cell counting retained only as a sensitivity comparison.",
        "scenario_pairing": "SSP2 population with SSP2-4.5 water; SSP5 population with SSP5-8.5 water; same target year.",
        "mask_dir": str(args.mask_dir),
        "row_counts": {"cases": len(cases), "long": len(long), "summary": len(summary)},
        "max_population_closure_error_million": float(
            summary["area_weighted_added_closure_error_million"].abs().max()
        ),
        "source_archives": {
            str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in archives
        },
        "outputs": {"long": str(long_path), "summary": str(summary_path)},
    }
    (args.out_dir / "metadata_population_exposure_shek_pik_egm2008.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
