#!/usr/bin/env python3
"""Build Shek Pik EGM2008 water levels from station data and AR6 increments.

The primary mean-sea-level baseline is the PSMSL metric series, independently
checked against HKO predicted tides. The MDT-derived baseline is retained only
as a sensitivity comparison because its reference-potential normalization is
not demonstrably compatible with the EGM2008 DEM.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window
from scipy import stats


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sample_bilinear(path: Path, lon: float, lat: float) -> tuple[float, dict[str, str]]:
    with rasterio.open(path) as src:
        col_corner, row_corner = (~src.transform) * (lon, lat)
        col = col_corner - 0.5
        row = row_corner - 0.5
        c0, r0 = math.floor(col), math.floor(row)
        block = src.read(1, window=Window(c0, r0, 2, 2), masked=True).astype(float)
        if block.shape != (2, 2) or np.any(np.ma.getmaskarray(block)):
            raise ValueError(f"Cannot bilinearly sample H0 at lon={lon}, lat={lat}")
        dx, dy = col - c0, row - r0
        value = (
            float(block[0, 0]) * (1 - dx) * (1 - dy)
            + float(block[0, 1]) * dx * (1 - dy)
            + float(block[1, 0]) * (1 - dx) * dy
            + float(block[1, 1]) * dx * dy
        )
        return value, src.tags()


def read_psmsl_monthly(path: Path) -> list[dict[str, float | int]]:
    records: list[dict[str, float | int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        fields = [field.strip() for field in line.split(";")]
        if len(fields) < 2:
            continue
        try:
            decimal_year = float(fields[0])
            value_mm = float(fields[1])
        except ValueError:
            continue
        if value_mm <= -99900 or not math.isfinite(value_mm):
            continue
        records.append(
            {
                "decimal_year": decimal_year,
                "year": int(math.floor(decimal_year)),
                "value_mm": value_mm,
            }
        )
    if not records:
        raise ValueError(f"No valid PSMSL monthly records in {path}")
    return records


def annual_means(monthly: list[dict[str, float | int]]) -> list[dict[str, float | int]]:
    grouped: dict[int, list[float]] = {}
    for row in monthly:
        grouped.setdefault(int(row["year"]), []).append(float(row["value_mm"]))
    rows = []
    for year, values in sorted(grouped.items()):
        if len(values) >= 10:
            rows.append(
                {
                    "year": year,
                    "months": len(values),
                    "mean_mm": float(np.mean(values)),
                    "std_monthly_mm": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                }
            )
    if len(rows) < 10:
        raise ValueError("Fewer than ten complete annual means are available")
    return rows


def merge_annual_means(
    rlr_rows: list[dict[str, float | int]], metric_rows: list[dict[str, float | int]]
) -> list[dict[str, object]]:
    rlr = {int(row["year"]): row for row in rlr_rows}
    metric = {int(row["year"]): row for row in metric_rows}
    output = []
    for year in sorted(set(rlr) & set(metric)):
        rr, mr = rlr[year], metric[year]
        output.append(
            {
                "year": year,
                "months": min(int(rr["months"]), int(mr["months"])),
                "mean_rlr_mm": float(rr["mean_mm"]),
                "mean_metric_chart_datum_mm": float(mr["mean_mm"]),
                "rlr_minus_metric_mm": float(rr["mean_mm"]) - float(mr["mean_mm"]),
                "rlr_std_monthly_mm": float(rr["std_monthly_mm"]),
                "metric_std_monthly_mm": float(mr["std_monthly_mm"]),
            }
        )
    return output


def period_mean(
    monthly: list[dict[str, float | int]], start_year: int, end_year: int
) -> tuple[float, dict[str, float | int]]:
    grouped: dict[int, list[float]] = {year: [] for year in range(start_year, end_year + 1)}
    for row in monthly:
        year = int(row["year"])
        if start_year <= year <= end_year:
            grouped[year].append(float(row["value_mm"]))
    values = [value for year_values in grouped.values() for value in year_values]
    expected = (end_year - start_year + 1) * 12
    minimum_yearly_months = min(len(year_values) for year_values in grouped.values())
    coverage = len(values) / expected
    if minimum_yearly_months < 6 or coverage < 0.95:
        raise ValueError(
            f"Insufficient monthly coverage for {start_year}-{end_year}: "
            f"{len(values)}/{expected}, minimum yearly months={minimum_yearly_months}"
        )
    return float(np.mean(values)), {
        "valid_months": len(values),
        "expected_months": expected,
        "coverage_fraction": coverage,
        "minimum_yearly_months": minimum_yearly_months,
        "missing_months": expected - len(values),
        "valid_months_by_year": {
            str(year): len(year_values) for year, year_values in grouped.items()
        },
    }


def check_psmsl_pair(
    rlr: list[dict[str, float | int]], metric: list[dict[str, float | int]]
) -> dict[str, float | int]:
    rlr_by_date = {float(row["decimal_year"]): float(row["value_mm"]) for row in rlr}
    metric_by_date = {float(row["decimal_year"]): float(row["value_mm"]) for row in metric}
    dates = sorted(set(rlr_by_date) & set(metric_by_date))
    if len(dates) != len(rlr) or len(dates) != len(metric):
        raise ValueError("RLR and metric PSMSL monthly records do not have identical dates")
    differences = np.asarray([rlr_by_date[date] - metric_by_date[date] for date in dates])
    median = float(np.median(differences))
    max_residual = float(np.max(np.abs(differences - median)))
    if max_residual > 1e-6:
        raise ValueError("PSMSL RLR-to-metric offset is not constant")
    return {
        "paired_months": len(dates),
        "rlr_minus_metric_median_mm": median,
        "max_residual_from_constant_offset_mm": max_residual,
    }


def read_hko_predicted_tides(path: Path) -> dict[str, float | int | str]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    values = []
    dates = []
    for row in rows:
        try:
            value = float(row["predicted_m"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            values.append(value)
            dates.append(str(row.get("datetime_hkt", "")))
    if not values:
        raise ValueError(f"No valid HKO predicted tides in {path}")
    return {
        "valid_hourly_values": len(values),
        "start": min(dates),
        "end": max(dates),
        "mean_chart_datum_m": float(np.mean(values)),
    }


def alignment_interval_m(theil, shift_years: float) -> tuple[float, float, float]:
    central = float(theil.slope) * shift_years / 1000.0
    endpoints = sorted(
        [
            float(theil.low_slope) * shift_years / 1000.0,
            float(theil.high_slope) * shift_years / 1000.0,
        ]
    )
    return central, endpoints[0], endpoints[1]


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0])
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    psmsl_root = repo_root / "external_data" / "psmsl" / "SHEK_PIK_1902"
    parser.add_argument(
        "--psmsl-rlr",
        type=Path,
        default=psmsl_root / "SHEK_PIK_1902_rlr_monthly.txt",
    )
    parser.add_argument(
        "--psmsl-metric",
        type=Path,
        default=psmsl_root / "SHEK_PIK_1902_metric_monthly.txt",
    )
    parser.add_argument(
        "--hko-predicted",
        type=Path,
        default=repo_root / "external_data" / "coastal_hazard" / "processed" / "hko_spw_predicted_tide_long.csv",
    )
    parser.add_argument(
        "--h0-raster",
        type=Path,
        default=repo_root / "private_inputs" / "vertical_datum" / "h0_msl_egm2008_1993_2012.tif",
    )
    parser.add_argument(
        "--scenarios",
        type=Path,
        default=repo_root / "data" / "derived_tables" / "vertical_datum" / "sea_level_scenarios_shekpik_ar6_novlm_verified.csv",
    )
    parser.add_argument(
        "--datum-calibration",
        type=Path,
        default=repo_root / "outputs" / "vertical_datum" / "shek_pik_datum_calibration" / "shek_pik_hkpd_egm2008_provenance.json",
    )
    parser.add_argument("--lon", type=float, default=113.894)
    parser.add_argument("--lat", type=float, default=22.209)
    parser.add_argument("--station-period-start", type=int, default=1998)
    parser.add_argument("--station-period-end", type=int, default=2014)
    parser.add_argument("--mdt-period-start", type=int, default=1993)
    parser.add_argument("--mdt-period-end", type=int, default=2012)
    parser.add_argument("--ar6-period-start", type=int, default=1995)
    parser.add_argument("--ar6-period-end", type=int, default=2014)
    parser.add_argument(
        "--out",
        type=Path,
        default=repo_root / "outputs" / "vertical_datum" / "shek_pik_boundary_egm2008",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    required = [
        args.psmsl_rlr,
        args.psmsl_metric,
        args.hko_predicted,
        args.h0_raster,
        args.scenarios,
        args.datum_calibration,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing input(s): " + "; ".join(missing))

    rlr_monthly = read_psmsl_monthly(args.psmsl_rlr)
    metric_monthly = read_psmsl_monthly(args.psmsl_metric)
    psmsl_pair_check = check_psmsl_pair(rlr_monthly, metric_monthly)
    rlr_annual = annual_means(rlr_monthly)
    metric_annual = annual_means(metric_monthly)
    annual = merge_annual_means(rlr_annual, metric_annual)

    years = np.asarray([float(row["year"]) + 0.5 for row in rlr_annual])
    levels = np.asarray([float(row["mean_mm"]) for row in rlr_annual])
    ols = stats.linregress(years, levels)
    theil = stats.theilslopes(levels, years, alpha=0.95)

    metric_mean_mm, metric_coverage = period_mean(
        metric_monthly, args.station_period_start, args.station_period_end
    )
    station_midpoint = (args.station_period_start + args.station_period_end + 1.0) / 2.0
    ar6_midpoint = (args.ar6_period_start + args.ar6_period_end + 1.0) / 2.0
    station_shift_years = ar6_midpoint - station_midpoint
    station_shift_m, station_shift_low_m, station_shift_high_m = alignment_interval_m(
        theil, station_shift_years
    )

    datum_calibration = json.loads(args.datum_calibration.read_text(encoding="utf-8"))
    datum_result = datum_calibration["result"]
    chart_datum_to_egm2008_m = float(datum_result["chart_datum_to_egm2008_m"])
    datum_ci95_m = float(datum_result["empirical_interpolation_ci95_m"])
    metric_mean_chart_datum_m = metric_mean_mm / 1000.0
    station_h0_observed_period_m = metric_mean_chart_datum_m + chart_datum_to_egm2008_m
    station_h0_ar6_period_m = station_h0_observed_period_m + station_shift_m

    hko = read_hko_predicted_tides(args.hko_predicted)
    hko_minus_psmsl_m = float(hko["mean_chart_datum_m"]) - metric_mean_chart_datum_m
    if abs(hko_minus_psmsl_m) > 0.05:
        raise ValueError(
            "PSMSL metric mean is not consistent with the independent HKO Chart Datum mean: "
            f"difference={hko_minus_psmsl_m:.6f} m"
        )

    mdt_h0_m, h0_tags = sample_bilinear(args.h0_raster, args.lon, args.lat)
    expected_quantity = "1993-2012 mean sea surface in EGM2008 heights"
    if h0_tags.get("quantity") != expected_quantity:
        raise ValueError(f"Unexpected H0 raster quantity tag: {h0_tags.get('quantity')!r}")
    mdt_midpoint = (args.mdt_period_start + args.mdt_period_end + 1.0) / 2.0
    mdt_shift_years = ar6_midpoint - mdt_midpoint
    mdt_shift_m, mdt_shift_low_m, mdt_shift_high_m = alignment_interval_m(
        theil, mdt_shift_years
    )
    mdt_h0_ar6_period_m = mdt_h0_m + mdt_shift_m
    mdt_minus_station_m = mdt_h0_ar6_period_m - station_h0_ar6_period_m

    with args.scenarios.open("r", encoding="utf-8-sig", newline="") as stream:
        scenario_rows = list(csv.DictReader(stream))
    if not scenario_rows:
        raise ValueError("Scenario CSV is empty")

    output_rows: list[dict[str, object]] = []
    for source_row in scenario_rows:
        row: dict[str, object] = dict(source_row)
        slr = float(source_row["slr_no_vlm_m"])
        surge = float(source_row["surge_m"])
        relative = float(source_row["water_level_m"])
        if not math.isclose(relative, slr + surge, abs_tol=1e-12):
            raise AssertionError("The source water_level_m is not AR6 no-VLM increment plus surge")
        absolute = station_h0_ar6_period_m + slr + surge
        row.update(
            {
                "primary_baseline_source": "PSMSL_1902_metric_mean_validated_by_HKO_predicted_tides",
                "psmsl_metric_baseline_period": (
                    f"{args.station_period_start}-{args.station_period_end}"
                ),
                "psmsl_metric_mean_chart_datum_m": f"{metric_mean_chart_datum_m:.9f}",
                "chart_datum_to_egm2008_m": f"{chart_datum_to_egm2008_m:.9f}",
                "station_epoch_alignment_method": (
                    "PSMSL_RLR_Theil_Sen_trend_times_reference_midpoint_shift"
                ),
                "station_epoch_midpoint_shift_years": f"{station_shift_years:.3f}",
                "station_epoch_alignment_m": f"{station_shift_m:.9f}",
                "station_epoch_alignment_ci95_low_m": f"{station_shift_low_m:.9f}",
                "station_epoch_alignment_ci95_high_m": f"{station_shift_high_m:.9f}",
                "h0_egm2008_ar6_1995_2014_m": f"{station_h0_ar6_period_m:.9f}",
                "station_datum_ci95_m": f"{datum_ci95_m:.9f}",
                "absolute_water_level_egm2008_m": f"{absolute:.9f}",
                "absolute_water_level_egm2008_datum_ci95_low_m": (
                    f"{absolute - datum_ci95_m:.9f}"
                ),
                "absolute_water_level_egm2008_datum_ci95_high_m": (
                    f"{absolute + datum_ci95_m:.9f}"
                ),
                "absolute_water_level_formula": (
                    "station_H0_EGM2008_1995_2014 + slr_no_vlm_m + surge_m"
                ),
                "chart_datum_offset_applied_to_ar6_increment": "false",
                "h0_egm2008_mdt_1993_2012_m_sensitivity": f"{mdt_h0_m:.9f}",
                "h0_egm2008_mdt_aligned_ar6_m_sensitivity": (
                    f"{mdt_h0_ar6_period_m:.9f}"
                ),
                "mdt_minus_station_h0_m": f"{mdt_minus_station_m:.9f}",
            }
        )
        output_rows.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "shek_pik_psmsl_annual_means.csv", annual)
    output_path = args.out / "sea_level_scenarios_shek_pik_absolute_egm2008.csv"
    write_csv(output_path, output_rows)

    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "target": {"station": "SHEK_PIK", "lon": args.lon, "lat": args.lat},
        "primary_baseline": {
            "source": "PSMSL station 1902 metric monthly series",
            "interpretation": (
                "Local metric values are treated as metres above Chart Datum after "
                "independent agreement with the HKO predicted-tide mean."
            ),
            "period": f"{args.station_period_start}-{args.station_period_end}",
            "period_midpoint": station_midpoint,
            "monthly_coverage": metric_coverage,
            "mean_chart_datum_m": metric_mean_chart_datum_m,
            "chart_datum_to_egm2008_m": chart_datum_to_egm2008_m,
            "h0_egm2008_observed_period_m": station_h0_observed_period_m,
            "h0_egm2008_ar6_1995_2014_m": station_h0_ar6_period_m,
            "empirical_station_datum_ci95_m": datum_ci95_m,
        },
        "independent_hko_check": {
            **hko,
            "hko_mean_minus_psmsl_1998_2014_mean_m": hko_minus_psmsl_m,
            "acceptance_tolerance_m": 0.05,
            "status": "pass",
        },
        "psmsl_pair_check": psmsl_pair_check,
        "reference_periods": {
            "station_observed": f"{args.station_period_start}-{args.station_period_end}",
            "ar6": f"{args.ar6_period_start}-{args.ar6_period_end}",
            "station_midpoint": station_midpoint,
            "ar6_midpoint": ar6_midpoint,
            "station_to_ar6_midpoint_shift_years": station_shift_years,
        },
        "psmsl_trend": {
            "station_id": 1902,
            "trend_series": "RLR annual means with at least 10 months",
            "monthly_start": min(float(row["decimal_year"]) for row in rlr_monthly),
            "monthly_end": max(float(row["decimal_year"]) for row in rlr_monthly),
            "valid_months": len(rlr_monthly),
            "complete_years": len(rlr_annual),
            "ols_slope_mm_yr": float(ols.slope),
            "ols_slope_standard_error_mm_yr": float(ols.stderr),
            "ols_r_squared": float(ols.rvalue**2),
            "theil_sen_slope_mm_yr": float(theil.slope),
            "theil_sen_slope_ci95_low_mm_yr": float(theil.low_slope),
            "theil_sen_slope_ci95_high_mm_yr": float(theil.high_slope),
        },
        "station_epoch_alignment": {
            "chosen_method": "Theil-Sen trend multiplied by reference-period midpoint shift",
            "shift_m": station_shift_m,
            "ci95_low_m": station_shift_low_m,
            "ci95_high_m": station_shift_high_m,
        },
        "mdt_sensitivity_only": {
            "h0_egm2008_1993_2012_m": mdt_h0_m,
            "h0_egm2008_aligned_ar6_m": mdt_h0_ar6_period_m,
            "midpoint_shift_years": mdt_shift_years,
            "epoch_shift_m": mdt_shift_m,
            "epoch_shift_ci95_low_m": mdt_shift_low_m,
            "epoch_shift_ci95_high_m": mdt_shift_high_m,
            "mdt_minus_station_h0_m": mdt_minus_station_m,
            "role": "sensitivity_comparison_not_primary_boundary",
        },
        "boundary": {
            "h0_egm2008_1995_2014_m": station_h0_ar6_period_m,
            "formula": (
                "absolute_water_level_EGM2008 = station_H0_EGM2008_1995_2014 "
                "+ AR6_noVLM_increment + surge"
            ),
            "chart_datum_offset_is_not_applied_to_relative_ar6_increment": True,
        },
        "independent_station_datum_validation": datum_result,
        "limitations": [
            "PSMSL marks metric series as requiring caution; its use here is supported by the independent HKO Chart Datum mean check.",
            "PSMSL Shek Pik begins in 1998, so alignment to 1995-2014 uses a trend-based midpoint correction.",
            "The local MDT file contains no MDT error variable and its raw geoid/MDT normalization differs materially from the station chain.",
            "The station datum uncertainty is empirical and excludes every possible systematic geoid or tide-system error.",
            "Future land-motion effects must be applied separately to the DEM or water-land elevation difference.",
        ],
        "source_files": {
            str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
            for path in required
        },
        "output_csv": str(output_path),
    }
    provenance_path = args.out / "shek_pik_boundary_egm2008_provenance.json"
    provenance_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
