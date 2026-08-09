#!/usr/bin/env python3
"""Extract verified IPCC AR6 no-VLM projections for Shek Pik.

The script reads the official regional no-VLM, medium-confidence NetCDF
files for SSP2-4.5 and SSP5-8.5. It extracts the median projection for
location 1902 (Shek Pik) in 2050 and 2100, adds the requested surge
increments, and writes a provenance-complete CSV without modifying the
legacy scenario table.

The resulting water_level_m is a relative scenario increment only:

    water_level_m = AR6 no-VLM sea-level increment + surge increment

It is not an absolute EGM2008 water elevation until the pointwise station
datum conversion has been independently verified and applied.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE_ROOT = REPO_ROOT / "external_data" / "ar6" / "selected_netcdf"
DEFAULT_LOCATION_LIST = REPO_ROOT / "external_data" / "ar6" / "location_list.lst"
DEFAULT_ZENODO_JSON = REPO_ROOT / "external_data" / "ar6" / "zenodo_record_6382554.json"
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "vertical_datum"

LOCATION_ID = 1902
LOCATION_NAME = "SHEK_PIK"
REFERENCE_PERIOD = "1995-2014"
SOURCE_VERSION = "20210809"
SOURCE_RECORD_DOI = "10.5281/zenodo.6382554"
SOURCE_CONCEPT_DOI = "10.5281/zenodo.5914709"
SOURCE_DATASET = "IPCC AR6 Sea Level Projections regional no-VLM confidence outputs"
REFERENCE_PERIOD_SOURCE = "IPCC AR6 WGI Chapter 9 / NASA-IPCC AR6 projection convention"
REFERENCE_PERIOD_URL = "https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-9/"

SCENARIOS = {
    "ssp245": {
        "label": "SSP2-4.5",
        "filename": "total_ssp245_medium_confidence_values.nc",
    },
    "ssp585": {
        "label": "SSP5-8.5",
        "filename": "total_ssp585_medium_confidence_values.nc",
    },
}
YEARS = (2050, 2100)
QUANTILE_FRACTION = 0.5
SURGES_M = (0.0, 1.0, 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--location-list", type=Path, default=DEFAULT_LOCATION_LIST)
    parser.add_argument("--zenodo-json", type=Path, default=DEFAULT_ZENODO_JSON)
    parser.add_argument(
        "--old-csv",
        type=Path,
        help="Optional legacy scenario CSV used only for a numerical comparison.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace outputs that already exist. The legacy input is never changed.",
    )
    return parser.parse_args()


def load_dependencies():
    try:
        import numpy as np
        from netCDF4 import Dataset
    except ImportError as exc:
        raise SystemExit(
            "This script requires numpy and netCDF4. Install the packages listed "
            "in the repository environment file.\n"
            f"Original import error: {exc}"
        ) from exc
    return np, Dataset


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def find_one(root: Path, filename: str) -> Path:
    matches = list(root.rglob(filename))
    if len(matches) != 1:
        raise SystemExit(
            f"Expected exactly one {filename!r} below {root}, found {len(matches)}"
        )
    return matches[0]


def verify_location_list(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"AR6 location list not found: {path}")
    matches = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as stream:
        for line in stream:
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) >= 4 and fields[1].strip() == str(LOCATION_ID):
                matches.append(fields)
    if len(matches) != 1:
        raise SystemExit(
            f"Expected one location-list record for {LOCATION_ID}, found {len(matches)}"
        )
    name, location_id, lat, lon = matches[0][:4]
    if name.strip().upper() != LOCATION_NAME:
        raise SystemExit(f"Location {LOCATION_ID} is {name!r}, not {LOCATION_NAME}")
    return {
        "name": name.strip(),
        "location_id": int(location_id),
        "lat": float(lat),
        "lon": float(lon),
        "source_file": str(path),
        "source_sha256": sha256_file(path),
    }


def verify_zenodo_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"Zenodo record metadata not found: {path}")
    record = json.loads(path.read_text(encoding="utf-8-sig"))
    observed = {
        "id": int(record.get("id")),
        "doi": str(record.get("doi")),
        "conceptdoi": str(record.get("conceptdoi")),
        "version": str(record.get("metadata", {}).get("version")),
        "title": str(record.get("metadata", {}).get("title")),
    }
    expected = {
        "id": 6382554,
        "doi": SOURCE_RECORD_DOI,
        "conceptdoi": SOURCE_CONCEPT_DOI,
        "version": SOURCE_VERSION,
    }
    for key, expected_value in expected.items():
        if observed[key] != expected_value:
            raise SystemExit(
                f"Zenodo metadata mismatch for {key}: "
                f"expected {expected_value!r}, found {observed[key]!r}"
            )
    observed["source_file"] = str(path)
    observed["source_sha256"] = sha256_file(path)
    return observed


def exact_index(values, target: float, np, label: str) -> int:
    matches = np.flatnonzero(np.isclose(values, target, rtol=0.0, atol=1e-12))
    if matches.size != 1:
        raise SystemExit(
            f"Expected one {label} coordinate matching {target}, found {matches.size}"
        )
    return int(matches[0])


def extract_scenario(
    scenario_code: str,
    source_path: Path,
    location_record: dict[str, Any],
    np,
    Dataset,
) -> tuple[dict[int, float], dict[str, Any]]:
    with Dataset(source_path, "r") as dataset:
        required = {
            "lat",
            "lon",
            "locations",
            "quantiles",
            "years",
            "sea_level_change",
        }
        missing = sorted(required.difference(dataset.variables))
        if missing:
            raise SystemExit(f"Missing variables in {source_path}: {missing}")

        location_ids = dataset.variables["locations"][:]
        quantiles = dataset.variables["quantiles"][:]
        years = dataset.variables["years"][:]
        location_index = exact_index(location_ids, LOCATION_ID, np, "location")
        quantile_index = exact_index(
            quantiles, QUANTILE_FRACTION, np, "quantile"
        )

        source_lat = float(dataset.variables["lat"][location_index])
        source_lon = float(dataset.variables["lon"][location_index])
        if not (
            math.isclose(source_lat, location_record["lat"], abs_tol=1e-4)
            and math.isclose(source_lon, location_record["lon"], abs_tol=1e-4)
        ):
            raise SystemExit(
                "NetCDF and location-list coordinates disagree for Shek Pik: "
                f"NetCDF=({source_lat}, {source_lon}), "
                f"list=({location_record['lat']}, {location_record['lon']})"
            )

        variable = dataset.variables["sea_level_change"]
        units = str(getattr(variable, "units", ""))
        if units.lower() != "mm":
            raise SystemExit(f"Expected sea_level_change units='mm', found {units!r}")
        if variable.dimensions != ("quantiles", "years", "locations"):
            raise SystemExit(
                "Unexpected sea_level_change dimensions: "
                f"{variable.dimensions!r}"
            )

        extracted_m: dict[int, float] = {}
        for year in YEARS:
            year_index = exact_index(years, year, np, "year")
            value_mm = variable[quantile_index, year_index, location_index]
            if np.ma.is_masked(value_mm) or not np.isfinite(float(value_mm)):
                raise SystemExit(
                    f"Invalid projection for {scenario_code}, {year}, location {LOCATION_ID}"
                )
            extracted_m[year] = float(value_mm) / 1000.0

        metadata = {
            "scenario_code": scenario_code,
            "source_file": str(source_path),
            "source_file_size_bytes": source_path.stat().st_size,
            "source_file_sha256": sha256_file(source_path),
            "global_attributes": {
                name: str(getattr(dataset, name)) for name in dataset.ncattrs()
            },
            "variable": "sea_level_change",
            "variable_units": units,
            "variable_dimensions": list(variable.dimensions),
            "variable_shape": list(variable.shape),
            "location_index": location_index,
            "location_id": int(location_ids[location_index]),
            "location_lat": source_lat,
            "location_lon": source_lon,
            "quantile_index": quantile_index,
            "quantile": float(quantiles[quantile_index]),
            "years": [int(value) for value in years.tolist()],
            "extracted_m": {str(key): value for key, value in extracted_m.items()},
        }
    return extracted_m, metadata


def output_rows(extracted: dict[str, dict[int, float]], extraction_date: str):
    rows = []
    for scenario_code, details in SCENARIOS.items():
        for year in YEARS:
            slr_m = extracted[scenario_code][year]
            for surge_m in SURGES_M:
                rows.append(
                    {
                        "station": LOCATION_NAME,
                        "psmsl_id": LOCATION_ID,
                        "station_lat": 22.22,
                        "station_lon": 113.89,
                        "scenario": details["label"],
                        "scenario_code": scenario_code,
                        "year": year,
                        "quantile": 50,
                        "total_slr_m": slr_m,
                        "vlm_m": 0.0,
                        "slr_no_vlm_m": slr_m,
                        "surge_m": surge_m,
                        "water_level_m": slr_m + surge_m,
                        "slr_reference_period": REFERENCE_PERIOD,
                        "source_dataset": SOURCE_DATASET,
                        "source_version": SOURCE_VERSION,
                        "source_record_doi": SOURCE_RECORD_DOI,
                        "source_concept_doi": SOURCE_CONCEPT_DOI,
                        "source_file": details["filename"],
                        "source_variable": "sea_level_change",
                        "source_location_id": LOCATION_ID,
                        "source_confidence": "medium_confidence",
                        "source_quantile": QUANTILE_FRACTION,
                        "source_unit": "mm",
                        "background_vlm_included": "false",
                        "total_slr_field_semantics": (
                            "direct_AR6_regional_total_excluding_background_VLM"
                        ),
                        "vlm_field_semantics": (
                            "zero_because_background_VLM_is_excluded_not_measured_zero"
                        ),
                        "reference_period_source": REFERENCE_PERIOD_SOURCE,
                        "reference_period_url": REFERENCE_PERIOD_URL,
                        "water_level_semantics": (
                            "AR6_noVLM_increment_plus_surge_not_absolute_datum"
                        ),
                        "station_datum_offset_applied": "false",
                        "extraction_date": extraction_date,
                    }
                )
    return rows


def compare_with_legacy(
    old_csv: Path | None, new_rows: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if old_csv is None:
        return []
    if not old_csv.is_file():
        raise SystemExit(f"Legacy comparison CSV not found: {old_csv}")
    with old_csv.open("r", encoding="utf-8-sig", newline="") as stream:
        legacy_rows = list(csv.DictReader(stream))
    old_by_key = {}
    for row in legacy_rows:
        if row.get("station", "").strip().upper() != LOCATION_NAME:
            continue
        key = (
            row.get("scenario_code", "").strip().lower(),
            int(float(row["year"])),
            float(row["surge_m"]),
        )
        old_by_key[key] = row

    comparison = []
    for row in new_rows:
        key = (row["scenario_code"], int(row["year"]), float(row["surge_m"]))
        old = old_by_key.get(key)
        old_slr = float(old["slr_no_vlm_m"]) if old else None
        old_water = float(old["water_level_m"]) if old else None
        comparison.append(
            {
                "station": LOCATION_NAME,
                "scenario_code": key[0],
                "year": key[1],
                "surge_m": key[2],
                "legacy_slr_no_vlm_m": old_slr,
                "official_slr_no_vlm_m": row["slr_no_vlm_m"],
                "slr_difference_official_minus_legacy_m": (
                    None
                    if old_slr is None
                    else round(row["slr_no_vlm_m"] - old_slr, 6)
                ),
                "legacy_water_level_m": old_water,
                "official_water_level_m": row["water_level_m"],
                "water_level_difference_official_minus_legacy_m": (
                    None
                    if old_water is None
                    else round(row["water_level_m"] - old_water, 6)
                ),
            }
        )
    return comparison


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    np, Dataset = load_dependencies()

    output_csv = args.out_dir / "sea_level_scenarios_shekpik_ar6_novlm_verified.csv"
    comparison_csv = args.out_dir / "ar6_shekpik_legacy_comparison.csv"
    provenance_json = args.out_dir / "ar6_shekpik_extraction_provenance.json"
    output_paths = [output_csv, provenance_json]
    if args.old_csv:
        output_paths.append(comparison_csv)
    existing = [path for path in output_paths if path.exists()]
    if existing and not args.force:
        joined = "\n".join(str(path) for path in existing)
        raise SystemExit(
            "Refusing to overwrite existing outputs. Use --force after review:\n" + joined
        )

    location_record = verify_location_list(args.location_list)
    zenodo_record = verify_zenodo_record(args.zenodo_json)
    extracted: dict[str, dict[int, float]] = {}
    netcdf_metadata: dict[str, Any] = {}
    for scenario_code, details in SCENARIOS.items():
        source_path = find_one(args.source_root, details["filename"])
        extracted[scenario_code], netcdf_metadata[scenario_code] = extract_scenario(
            scenario_code, source_path, location_record, np, Dataset
        )

    extraction_date = datetime.now(timezone.utc).date().isoformat()
    rows = output_rows(extracted, extraction_date)
    comparison = compare_with_legacy(args.old_csv, rows)
    differences = [
        abs(float(row["slr_difference_official_minus_legacy_m"]))
        for row in comparison
        if row["slr_difference_official_minus_legacy_m"] is not None
    ]

    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "scientific_selection": {
            "station": LOCATION_NAME,
            "source_location_id": LOCATION_ID,
            "scenarios": list(SCENARIOS),
            "years": list(YEARS),
            "quantile": QUANTILE_FRACTION,
            "confidence": "medium_confidence",
            "variable": "sea_level_change",
            "reference_period": REFERENCE_PERIOD,
            "background_vlm_included": False,
            "surges_m": list(SURGES_M),
        },
        "location_list_validation": location_record,
        "zenodo_record_validation": zenodo_record,
        "netcdf_validation": netcdf_metadata,
        "outputs": {
            "verified_scenario_csv": str(output_csv),
            "legacy_comparison_csv": str(comparison_csv) if comparison else None,
        },
        "legacy_comparison": {
            "source": str(args.old_csv) if args.old_csv else None,
            "row_count": len(comparison),
            "maximum_absolute_slr_difference_m": max(differences) if differences else None,
        },
        "datum_warning": (
            "water_level_m is a relative AR6 no-VLM increment plus surge. "
            "It is not an absolute EGM2008 boundary until the verified pointwise "
            "station-datum offset is applied."
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_csv, rows)
    write_csv(comparison_csv, comparison)
    provenance_json.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("Verified AR6 Shek Pik extraction")
    print("=================================")
    for scenario_code in SCENARIOS:
        for year in YEARS:
            print(f"{scenario_code} {year}: {extracted[scenario_code][year]:.3f} m")
    if differences:
        print(f"Maximum absolute difference from legacy table: {max(differences):.3f} m")
    print(f"Written: {output_csv}")
    if comparison:
        print(f"Written: {comparison_csv}")
    print(f"Written: {provenance_json}")
    print("Datum status: station conversion still pending; do not build final rasters yet.")


if __name__ == "__main__":
    main()
