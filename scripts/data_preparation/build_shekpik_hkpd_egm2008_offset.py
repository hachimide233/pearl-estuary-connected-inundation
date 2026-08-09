#!/usr/bin/env python3
"""Derive a Shek Pik Chart Datum to EGM2008 offset from official HK controls."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import rasterio
from rasterio.windows import Window


HKO_CD_BELOW_HKPD_M = 0.146
HKO_TIDE_NOTES_URL = "https://www.hko.gov.hk/en/tide/enotes.htm"
HK_CONTROL_DATABASE_URL = "https://www.geodetic.gov.hk/en/gi/search.asp"
HK_CONTROL_FRAME = "ITRF96 epoch 1998:121"
KYC1_SATREF_ELLIPSOIDAL_HEIGHT_M = 116.3830


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def as_float(value: str | None) -> float | None:
    try:
        number = float((value or "").strip())
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def dms_to_decimal(deg: str, minute: str, second: str) -> float | None:
    parts = [as_float(deg), as_float(minute), as_float(second)]
    if any(value is None for value in parts):
        return None
    d, m, s = parts
    if d == 0.0 and m == 0.0 and s == 0.0:
        return None
    sign = -1.0 if d < 0 else 1.0
    return sign * (abs(d) + m / 60.0 + s / 3600.0)


def class_number(text: str, prefix: str) -> int | None:
    match = re.search(rf"(?:^|,)\s*{re.escape(prefix)}(\d+)\s*(?:,|$)", text or "")
    return int(match.group(1)) if match else None


def iter_xml_records(path: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for element in ET.parse(path).getroot().iter():
        children = list(element)
        if children and all(len(child) == 0 for child in children):
            records.append(
                {child.tag.split("}")[-1]: (child.text or "").strip() for child in children}
            )
    return records


def read_controls(paths: list[tuple[str, Path]]) -> list[dict[str, object]]:
    controls: list[dict[str, object]] = []
    for source_type, path in paths:
        for row in iter_xml_records(path):
            hkpd = as_float(row.get("HKPD_m"))
            ellipsoidal = as_float(row.get("WGS_LEVEL"))
            lat = dms_to_decimal(row.get("LAT_DEG", ""), row.get("LAT_MIN", ""), row.get("LAT_SEC", ""))
            lon = dms_to_decimal(row.get("LONG_DEG", ""), row.get("LONG_MIN", ""), row.get("LONG_SEC", ""))
            if (
                hkpd is None
                or ellipsoidal is None
                or lat is None
                or lon is None
                or hkpd == 0.0
                or ellipsoidal == 0.0
                or row.get("FRAME") != "ITRF96"
            ):
                continue
            controls.append(
                {
                    "source_type": source_type,
                    "source_file": str(path),
                    "station_id": row.get("STN_NO", ""),
                    "station_name": row.get("STN_NAME") or row.get("LOCALITY", ""),
                    "locality": row.get("LOCALITY", ""),
                    "lon": lon,
                    "lat": lat,
                    "ellipsoidal_height_itrf96_m": ellipsoidal,
                    "hkpd_height_m": hkpd,
                    "gps_accuracy": row.get("GPS_ACURCY", ""),
                    "gps_vertical_class": class_number(row.get("GPS_ACURCY", ""), "GV"),
                    "levelling_accuracy": row.get("LEV_ACURCY", ""),
                    "levelling_class": class_number(row.get("LEV_ACURCY", ""), "V"),
                    "by_transform": row.get("BY_TRANSFO", ""),
                    "hkpd_text_precision_decimals": len(row.get("HKPD_m", "").partition(".")[2]),
                }
            )
    return controls


def haversine_km(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    radius_km = 6371.0088
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lon2 - lon1)
    value = math.sin(dphi / 2.0) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    return 2.0 * radius_km * math.asin(min(1.0, math.sqrt(value)))


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlambda = math.radians(lon2 - lon1)
    y = math.sin(dlambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlambda)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def sample_bilinear(src: rasterio.io.DatasetReader, lon: float, lat: float) -> float:
    col_corner, row_corner = (~src.transform) * (lon, lat)
    col = col_corner - 0.5
    row = row_corner - 0.5
    c0 = math.floor(col)
    r0 = math.floor(row)
    if c0 < 0 or r0 < 0 or c0 + 1 >= src.width or r0 + 1 >= src.height:
        raise ValueError(f"Point outside EGM2008 interpolation domain: lon={lon}, lat={lat}")
    block = src.read(1, window=Window(c0, r0, 2, 2), masked=True).astype(float)
    if np.ma.is_masked(block) and np.any(np.ma.getmaskarray(block)):
        raise ValueError(f"Masked EGM2008 sample near lon={lon}, lat={lat}")
    dx = col - c0
    dy = row - r0
    q00, q01 = float(block[0, 0]), float(block[0, 1])
    q10, q11 = float(block[1, 0]), float(block[1, 1])
    return (
        q00 * (1.0 - dx) * (1.0 - dy)
        + q01 * dx * (1.0 - dy)
        + q10 * (1.0 - dx) * dy
        + q11 * dx * dy
    )


def idw_predict(
    rows: list[dict[str, object]], lon: float, lat: float, k: int, power: float
) -> tuple[float, list[dict[str, object]], np.ndarray]:
    ranked = sorted(rows, key=lambda row: haversine_km(lon, lat, float(row["lon"]), float(row["lat"])))
    selected = ranked[: min(k, len(ranked))]
    distances = np.array(
        [haversine_km(lon, lat, float(row["lon"]), float(row["lat"])) for row in selected],
        dtype=float,
    )
    values = np.array([float(row["hkpd_to_egm2008_m"]) for row in selected], dtype=float)
    if np.any(distances < 1e-9):
        index = int(np.argmin(distances))
        weights = np.zeros_like(distances)
        weights[index] = 1.0
        return float(values[index]), selected, weights
    weights = 1.0 / np.power(distances, power)
    weights /= weights.sum()
    return float(np.dot(weights, values)), selected, weights


def cross_validate(rows: list[dict[str, object]], k: int, power: float) -> dict[str, float]:
    residuals: list[float] = []
    for index, held_out in enumerate(rows):
        training = rows[:index] + rows[index + 1 :]
        prediction, _, _ = idw_predict(
            training, float(held_out["lon"]), float(held_out["lat"]), k, power
        )
        residuals.append(prediction - float(held_out["hkpd_to_egm2008_m"]))
    array = np.asarray(residuals, dtype=float)
    return {
        "cv_rmse_m": float(np.sqrt(np.mean(array**2))),
        "cv_mae_m": float(np.mean(np.abs(array))),
        "cv_bias_m": float(np.mean(array)),
        "cv_p95_abs_m": float(np.percentile(np.abs(array), 95)),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    repo_root = Path(__file__).resolve().parents[2]
    parser.add_argument("--source-dir", type=Path, default=repo_root / "external_data" / "hkpd")
    parser.add_argument("--egm2008", type=Path, default=repo_root / "external_data" / "geoid" / "us_nga_egm08_25.tif")
    parser.add_argument(
        "--hko-tide-notes",
        type=Path,
        default=repo_root / "external_data" / "coastal_hazard" / "raw" / "official_datum_sources" / "hko_tide_notes.html",
    )
    parser.add_argument("--target-lon", type=float, default=113.894)
    parser.add_argument("--target-lat", type=float, default=22.209)
    parser.add_argument("--radius-km", type=float, default=60.0)
    parser.add_argument("--out", type=Path, default=repo_root / "outputs" / "vertical_datum" / "shek_pik_datum_calibration")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trig = args.source_dir / "Trig.xml"
    traverse = args.source_dir / "Traverse.xml"
    data_dictionary = args.source_dir / "DataDict_DB.pdf"
    satref = args.source_dir / "SatRef_Coord.pdf"
    accuracy_standard = args.source_dir / "Accuracy_Standards_of_Control_Survey_v2.0.pdf"
    required = [trig, traverse, data_dictionary, satref, accuracy_standard, args.egm2008, args.hko_tide_notes]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing required source(s): " + "; ".join(missing))

    # HKO states Chart Datum is 0.146 m below HKPD. The same physical water
    # surface therefore has an HKPD height 0.146 m smaller than its CD height.
    test_cd = 1.000
    test_hkpd = test_cd - HKO_CD_BELOW_HKPD_M
    if not math.isclose(test_hkpd, 0.854, abs_tol=1e-12):
        raise AssertionError("Chart Datum to HKPD sign test failed")

    controls = read_controls([("Trig", trig), ("Traverse", traverse)])
    with rasterio.open(args.egm2008) as egm:
        target_geoid = sample_bilinear(egm, args.target_lon, args.target_lat)
        for row in controls:
            geoid = sample_bilinear(egm, float(row["lon"]), float(row["lat"]))
            h = float(row["ellipsoidal_height_itrf96_m"])
            hkpd = float(row["hkpd_height_m"])
            row["egm2008_geoid_undulation_m"] = geoid
            row["h_minus_hkpd_m"] = h - hkpd
            row["egm2008_orthometric_height_m"] = h - geoid
            row["hkpd_to_egm2008_m"] = h - geoid - hkpd
            row["distance_to_shek_pik_km"] = haversine_km(
                args.target_lon, args.target_lat, float(row["lon"]), float(row["lat"])
            )
            row["bearing_from_shek_pik_deg"] = bearing_deg(
                args.target_lon, args.target_lat, float(row["lon"]), float(row["lat"])
            )

    kyc1 = next(
        (row for row in controls if row["source_type"] == "Trig" and row["station_id"] == "75"),
        None,
    )
    if kyc1 is None:
        raise RuntimeError("KYC1 / Trig 75 was not found for the SatRef cross-check")
    kyc1_difference = float(kyc1["ellipsoidal_height_itrf96_m"]) - KYC1_SATREF_ELLIPSOIDAL_HEIGHT_M
    if abs(kyc1_difference) > 0.0006:
        raise AssertionError("WGS_LEVEL does not agree with official KYC1 SatRef ellipsoidal height")

    local = [
        row
        for row in controls
        if float(row["distance_to_shek_pik_km"]) <= args.radius_km
        and row["gps_vertical_class"] is not None
        and int(row["gps_vertical_class"]) <= 4
        and row["levelling_class"] is not None
        and int(row["levelling_class"]) <= 4
    ]
    if len(local) < 8:
        raise RuntimeError(f"Only {len(local)} preferred controls are available within {args.radius_km} km")

    values = np.asarray([float(row["hkpd_to_egm2008_m"]) for row in local], dtype=float)
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    clip_limit = max(0.10, 4.0 * robust_sigma)
    used = [row for row in local if abs(float(row["hkpd_to_egm2008_m"]) - median) <= clip_limit]
    rejected = [row for row in local if row not in used]
    if len(used) < 8:
        raise RuntimeError("Robust filtering left fewer than eight controls")

    candidates: list[dict[str, object]] = []
    for k in (3, 4, 5, 6, 8, 10, 12):
        if k >= len(used):
            continue
        for power in (1.0, 1.5, 2.0, 2.5, 3.0):
            diagnostics = cross_validate(used, k, power)
            target_estimate, selected, weights = idw_predict(used, args.target_lon, args.target_lat, k, power)
            candidates.append(
                {
                    "method": "IDW",
                    "k": k,
                    "power": power,
                    "target_hkpd_to_egm2008_m": target_estimate,
                    "nearest_distance_km": min(float(row["distance_to_shek_pik_km"]) for row in selected),
                    "farthest_selected_distance_km": max(float(row["distance_to_shek_pik_km"]) for row in selected),
                    **diagnostics,
                }
            )
    candidates.sort(key=lambda row: (float(row["cv_rmse_m"]), float(row["cv_mae_m"]), int(row["k"])))
    best = candidates[0]
    estimate, selected, weights = idw_predict(
        used, args.target_lon, args.target_lat, int(best["k"]), float(best["power"])
    )

    selected_values = np.asarray([float(row["hkpd_to_egm2008_m"]) for row in selected])
    local_weighted_std = float(np.sqrt(np.dot(weights, (selected_values - estimate) ** 2)))
    interpolation_sigma = max(float(best["cv_rmse_m"]), local_weighted_std)
    interpolation_ci95 = 1.96 * interpolation_sigma
    chart_datum_to_egm2008 = estimate - HKO_CD_BELOW_HKPD_M
    bearings = [float(row["bearing_from_shek_pik_deg"]) for row in selected]
    occupied_quadrants = len({int(bearing // 90.0) % 4 for bearing in bearings})
    nearest_distance = min(float(row["distance_to_shek_pik_km"]) for row in selected)
    nearby_validation = sorted(
        [
            row
            for row in controls
            if row not in used and float(row["distance_to_shek_pik_km"]) <= 10.0
        ],
        key=lambda row: float(row["distance_to_shek_pik_km"]),
    )
    shek_pik_control = next(
        (row for row in nearby_validation if str(row["station_name"]).strip().upper() == "SHEK PIK"),
        None,
    )
    if shek_pik_control is None:
        raise RuntimeError("The nearby official Trig 250 / SHEK PIK validation control was not found")
    shek_pik_control_residual = float(shek_pik_control["hkpd_to_egm2008_m"]) - estimate
    confidence = (
        "medium_interpolated_official_controls"
        if nearest_distance <= 15.0 and occupied_quadrants >= 2 and interpolation_sigma <= 0.15
        else "low_interpolated_official_controls"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    controls_sorted = sorted(controls, key=lambda row: float(row["distance_to_shek_pik_km"]))
    used_ids = {(str(row["source_type"]), str(row["station_id"])) for row in used}
    selected_weight_by_id = {
        (str(row["source_type"]), str(row["station_id"])): float(weight)
        for row, weight in zip(selected, weights)
    }
    for row in controls_sorted:
        key = (str(row["source_type"]), str(row["station_id"]))
        row["passes_preferred_local_filter"] = key in used_ids
        row["selected_best_model"] = key in selected_weight_by_id
        row["best_model_weight"] = selected_weight_by_id.get(key, 0.0)
    write_csv(args.out / "hkpd_egm2008_control_points.csv", controls_sorted)
    write_csv(args.out / "idw_candidate_cross_validation.csv", candidates)
    write_csv(args.out / "robust_filter_rejected_controls.csv", rejected, list(controls_sorted[0]))

    source_files = {
        str(path): {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        for path in required
    }
    provenance = {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "script": str(Path(__file__).resolve()),
        "target": {"station": "SHEK_PIK", "lon": args.target_lon, "lat": args.target_lat},
        "formula": {
            "control_point": "H_EGM2008 = h_ITRF96 - N_EGM2008; delta_HKPD_to_EGM2008 = H_EGM2008 - H_HKPD",
            "chart_datum": "H_HKPD = H_CD - 0.146; H_EGM2008 = H_CD + (delta_HKPD_to_EGM2008 - 0.146)",
            "offset_sign_rule": "H_EGM2008 = H_CD + offset_chart_datum_to_egm2008_m",
        },
        "official_definitions": {
            "hko_chart_datum_relation": "Chart Datum is 0.146 m below HKPD",
            "hko_url": HKO_TIDE_NOTES_URL,
            "control_database_url": HK_CONTROL_DATABASE_URL,
            "control_frame": HK_CONTROL_FRAME,
            "wgs_level_interpretation": "Ellipsoidal height; verified against KYC1 in the official SatRef coordinate table",
            "kyc1_xml_minus_satref_m": kyc1_difference,
        },
        "control_selection": {
            "all_paired_itrf96_controls": len(controls),
            "radius_km": args.radius_km,
            "preferred_local_before_robust_filter": len(local),
            "robust_median_m": median,
            "robust_mad_m": mad,
            "robust_sigma_m": robust_sigma,
            "clip_limit_m": clip_limit,
            "retained": len(used),
            "rejected": len(rejected),
            "quality_rule": "ITRF96, GNSS vertical class GV1-GV4, levelling class V1-V4",
        },
        "best_model": best,
        "selected_controls": [
            {
                "source_type": row["source_type"],
                "station_id": row["station_id"],
                "station_name": row["station_name"],
                "lon": row["lon"],
                "lat": row["lat"],
                "distance_km": row["distance_to_shek_pik_km"],
                "bearing_deg": row["bearing_from_shek_pik_deg"],
                "hkpd_to_egm2008_m": row["hkpd_to_egm2008_m"],
                "weight": float(weight),
                "gps_accuracy": row["gps_accuracy"],
                "levelling_accuracy": row["levelling_accuracy"],
            }
            for row, weight in zip(selected, weights)
        ],
        "nearby_excluded_validation_controls": [
            {
                "source_type": row["source_type"],
                "station_id": row["station_id"],
                "station_name": row["station_name"],
                "distance_km": row["distance_to_shek_pik_km"],
                "hkpd_to_egm2008_m": row["hkpd_to_egm2008_m"],
                "model_minus_control_m": estimate - float(row["hkpd_to_egm2008_m"]),
                "gps_accuracy": row["gps_accuracy"],
                "levelling_accuracy": row["levelling_accuracy"],
                "exclusion_reason": "Levelling class is missing, V5, or V6; validation only",
            }
            for row in nearby_validation
        ],
        "result": {
            "target_egm2008_geoid_undulation_m": target_geoid,
            "hkpd_to_egm2008_m": estimate,
            "chart_datum_to_egm2008_m": chart_datum_to_egm2008,
            "empirical_interpolation_sigma_m": interpolation_sigma,
            "empirical_interpolation_ci95_m": interpolation_ci95,
            "nearest_control_distance_km": nearest_distance,
            "selected_control_quadrants": occupied_quadrants,
            "nearby_shek_pik_control_id": shek_pik_control["station_id"],
            "nearby_shek_pik_control_distance_km": shek_pik_control["distance_to_shek_pik_km"],
            "nearby_shek_pik_control_hkpd_to_egm2008_m": shek_pik_control["hkpd_to_egm2008_m"],
            "nearby_shek_pik_control_minus_model_m": shek_pik_control_residual,
            "confidence": confidence,
        },
        "limitations": [
            "The result is a local interpolation from official co-located controls, not a direct tide-gauge benchmark tie.",
            "The empirical uncertainty is based on control residuals and cross-validation; it does not include every systematic geoid or epoch error.",
            "ITRF96 epoch 1998:121 coordinates are used as WGS84-compatible horizontal positions for EGM2008 sampling.",
            "Do not reuse the earlier regional-mean -0.582 m sensitivity value as a verified pointwise offset.",
        ],
        "source_files": source_files,
    }
    provenance_path = args.out / "shek_pik_hkpd_egm2008_provenance.json"
    provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    station_row = {
        "station": "SHEK_PIK",
        "lon": args.target_lon,
        "lat": args.target_lat,
        "water_level_datum": "HKO Chart Datum",
        "offset_water_datum_to_egm2008_m": f"{chart_datum_to_egm2008:.6f}",
        "offset_sign_rule": "H_EGM2008 = H_CD + offset_water_datum_to_egm2008_m",
        "source_title": "HKO Tide Notes; Hong Kong Geodetic Survey Control Station Database; NGA EGM2008",
        "source_url_or_file": f"{HKO_TIDE_NOTES_URL}; {HK_CONTROL_DATABASE_URL}; {provenance_path}",
        "confidence": confidence,
        "notes": (
            f"HKPD->EGM2008={estimate:.6f} m by CV-selected IDW(k={int(best['k'])}, p={float(best['power']):g}); "
            f"CD->HKPD=-{HKO_CD_BELOW_HKPD_M:.3f} m; empirical interpolation 95% range +/-{interpolation_ci95:.3f} m; "
            f"independent Trig {shek_pik_control['station_id']} at {float(shek_pik_control['distance_to_shek_pik_km']):.3f} km differs by "
            f"{shek_pik_control_residual:+.3f} m; interpolated from official controls, not a direct tide-gauge benchmark tie"
        ),
    }
    station_path = args.out / "station_datum_offsets_shek_pik_verified.csv"
    write_csv(station_path, [station_row])

    report = [
        "Shek Pik Chart Datum to EGM2008 calibration",
        "============================================",
        "",
        f"Target: {args.target_lon:.6f} E, {args.target_lat:.6f} N",
        f"Paired official ITRF96 controls: {len(controls)}",
        f"Preferred local controls retained: {len(used)} (rejected: {len(rejected)})",
        f"Best IDW: k={int(best['k'])}, power={float(best['power']):g}",
        f"Cross-validation RMSE: {float(best['cv_rmse_m']):.4f} m",
        f"Nearest selected control: {nearest_distance:.3f} km",
        f"Selected directional quadrants: {occupied_quadrants}/4",
        (
            f"Independent nearby validation: Trig {shek_pik_control['station_id']} / "
            f"{shek_pik_control['station_name']} at {float(shek_pik_control['distance_to_shek_pik_km']):.3f} km, "
            f"offset={float(shek_pik_control['hkpd_to_egm2008_m']):+.4f} m, "
            f"control-minus-model={shek_pik_control_residual:+.4f} m"
        ),
        "",
        f"HKPD -> EGM2008 offset: {estimate:+.4f} m",
        f"Chart Datum -> HKPD offset: {-HKO_CD_BELOW_HKPD_M:+.4f} m",
        f"Chart Datum -> EGM2008 offset: {chart_datum_to_egm2008:+.4f} m",
        f"Empirical interpolation uncertainty (1 sigma): {interpolation_sigma:.4f} m",
        f"Empirical interpolation uncertainty (95%): +/-{interpolation_ci95:.4f} m",
        f"Confidence: {confidence}",
        "",
        "Use: H_EGM2008 = H_CD + offset_chart_datum_to_egm2008_m",
        "Do not use the previous -0.582 m regional-mean sensitivity value as a pointwise transform.",
    ]
    (args.out / "shek_pik_hkpd_egm2008_report.txt").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n".join(report))
    print(f"\nWrote: {args.out}")


if __name__ == "__main__":
    main()
