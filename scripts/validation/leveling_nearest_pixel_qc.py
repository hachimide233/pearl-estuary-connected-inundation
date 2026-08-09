# -*- coding: utf-8 -*-
"""Auditable leveling--InSAR trend comparison using nearest quality pixels.

The spatial match is determined before any agreement statistic is calculated:
the nearest candidate satisfying fixed validity and coherence thresholds is
selected. Temporal spikes are flagged with a fixed Hampel-style rule. The
script never moves a pixel or removes a site to improve correlation or RMSE.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import numpy as np
import pandas as pd
import rasterio


plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["font.size"] = 8
plt.rcParams["axes.spines.right"] = False
plt.rcParams["axes.spines.top"] = False
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["legend.frameon"] = False
plt.rcParams["axes.unicode_minus"] = False


LEVEL_COLS = ["h201301", "h201308", "h201403", "h201409", "h201509"]
LEVEL_DATES = pd.to_datetime(
    ["2013-01-01", "2013-08-01", "2014-03-01", "2014-09-01", "2015-09-01"]
)
PALETTE = {
    "level": "#0F4D92",
    "insar": "#B64342",
    "band": "#A9C4E2",
    "outlier": "#D9902F",
    "neutral": "#606060",
}


def read_leveling(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    if frame.shape[1] < 8:
        raise ValueError("Leveling CSV must contain id, five epochs, latitude and longitude.")
    frame = frame.rename(
        columns={
            frame.columns[0]: "id",
            frame.columns[1]: "h201301",
            frame.columns[2]: "h201308",
            frame.columns[3]: "h201403",
            frame.columns[4]: "h201409",
            frame.columns[5]: "h201509",
            frame.columns[6]: "lat",
            frame.columns[7]: "lon",
        }
    )
    frame["id"] = frame["id"].astype(int)
    for col in LEVEL_COLS + ["lat", "lon"]:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame["id"].duplicated().any():
        raise ValueError("Leveling IDs must be unique.")
    return frame.sort_values("id").reset_index(drop=True)


def decode_attr(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def get_attr(file_obj, dataset, name: str):
    value = file_obj.attrs.get(name, dataset.attrs.get(name))
    return decode_attr(value)


def haversine_m(lon1, lat1, lon2, lat2):
    radius = 6371008.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2.0) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2.0) ** 2
    return 2.0 * radius * math.asin(math.sqrt(a))


def find_coherence_tif(value: str | None) -> Path | None:
    if value is None or value.strip().lower() in {"", "none"}:
        return None
    path = Path(value)
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def h5_settlement_factor(file_obj, dataset) -> tuple[float, str]:
    sign = str(get_attr(file_obj, dataset, "mintpy.vertical_sign") or "").strip().lower()
    if sign == "positive_up":
        return -1000.0, sign
    if sign == "positive_down":
        return 1000.0, sign
    raise RuntimeError(
        "HDF5 vertical sign is missing or unsupported. Refusing to guess the sign; "
        "expected mintpy.vertical_sign=positive_up or positive_down."
    )


def hampel_mask(values: np.ndarray, window: int, sigma: float, floor_mm: float) -> tuple[np.ndarray, float]:
    series = pd.Series(np.asarray(values, dtype=float))
    local_median = series.rolling(window, center=True, min_periods=1).median().to_numpy()
    residual = values - local_median
    center = float(np.nanmedian(residual))
    mad = float(np.nanmedian(np.abs(residual - center)))
    threshold = max(sigma * 1.4826 * mad, floor_mm)
    mask = np.isfinite(values) & (np.abs(residual - center) > threshold)
    return mask, threshold


def linear_fit_ci(x: np.ndarray, y: np.ndarray, x_grid: np.ndarray):
    valid = np.isfinite(x) & np.isfinite(y)
    x = np.asarray(x[valid], dtype=float)
    y = np.asarray(y[valid], dtype=float)
    if x.size < 3:
        raise ValueError("At least three observations are required for a linear fit.")
    slope, intercept = np.polyfit(x, y, 1)
    fitted = intercept + slope * x
    residual = y - fitted
    sxx = float(np.sum((x - x.mean()) ** 2))
    if sxx <= 0:
        raise ValueError("Regression dates have zero spread.")
    residual_se = math.sqrt(float(np.sum(residual**2)) / (x.size - 2))
    # Exact 97.5% t critical for n=5 (df=3), otherwise the large-sample value.
    tcrit = 3.182446305 if x.size == 5 else 1.969
    pred = intercept + slope * x_grid
    mean_se = residual_se * np.sqrt(1.0 / x.size + (x_grid - x.mean()) ** 2 / sxx)
    slope_se = residual_se / math.sqrt(sxx)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "slope_ci_low": float(slope - tcrit * slope_se),
        "slope_ci_high": float(slope + tcrit * slope_se),
        "pred": pred,
        "ci_low": pred - tcrit * mean_se,
        "ci_high": pred + tcrit * mean_se,
    }


def normalize_rank(series: pd.Series, ascending: bool) -> pd.Series:
    if series.notna().sum() <= 1:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    ranks = series.rank(ascending=ascending, method="average", na_option="bottom")
    return (ranks - ranks.min()) / max(float(ranks.max() - ranks.min()), 1.0)


def inspect_and_select(
    h5_path: Path,
    leveling: pd.DataFrame,
    coherence_path: Path | None,
    search_radius: int,
    min_valid_fraction: float,
    min_coherence: float,
    hampel_window: int,
    hampel_sigma: float,
    hampel_floor_mm: float,
):
    candidate_rows = []
    selected_rows = []
    series_rows = []
    coherence = rasterio.open(coherence_path) if coherence_path else None
    try:
        with h5py.File(h5_path, "r") as f:
            ds = f["timeseries"]
            dates = pd.to_datetime(
                [decode_attr(v) for v in f["date"][:]], format="%Y%m%d", errors="raise"
            )
            xf = float(get_attr(f, ds, "X_FIRST"))
            yf = float(get_attr(f, ds, "Y_FIRST"))
            xs = float(get_attr(f, ds, "X_STEP"))
            ys = float(get_attr(f, ds, "Y_STEP"))
            factor, vertical_sign = h5_settlement_factor(f, ds)
            unit = str(get_attr(f, ds, "UNIT") or "").lower()
            if unit not in {"m", "meter", "metre", "meters", "metres"}:
                raise RuntimeError(f"Expected HDF5 displacement unit in metres, found {unit!r}.")
            for point in leveling.itertuples(index=False):
                base_row = int(round((float(point.lat) - yf) / ys))
                base_col = int(round((float(point.lon) - xf) / xs))
                candidates = []
                for dr in range(-search_radius, search_radius + 1):
                    for dc in range(-search_radius, search_radius + 1):
                        row = base_row + dr
                        col = base_col + dc
                        if not (0 <= row < ds.shape[1] and 0 <= col < ds.shape[2]):
                            continue
                        raw_m = np.asarray(ds[:, row, col], dtype=np.float64)
                        finite = np.isfinite(raw_m)
                        valid_fraction = float(finite.mean())
                        settlement = raw_m * factor
                        temporal_std = float(np.nanstd(settlement))
                        pixel_lon = xf + col * xs
                        pixel_lat = yf + row * ys
                        distance = haversine_m(
                            float(point.lon), float(point.lat), pixel_lon, pixel_lat
                        )
                        coh_value = np.nan
                        if coherence is not None:
                            cr, cc = coherence.index(pixel_lon, pixel_lat)
                            if 0 <= cr < coherence.height and 0 <= cc < coherence.width:
                                coh_value = float(coherence.read(1, window=((cr, cr + 1), (cc, cc + 1)))[0, 0])
                        eligible = (
                            valid_fraction >= min_valid_fraction
                            and np.isfinite(temporal_std)
                            and temporal_std > 1e-6
                            and (coherence is None or (np.isfinite(coh_value) and coh_value >= min_coherence))
                        )
                        record = {
                            "id": int(point.id),
                            "row": row,
                            "col": col,
                            "row_offset": dr,
                            "col_offset": dc,
                            "pixel_lon": pixel_lon,
                            "pixel_lat": pixel_lat,
                            "distance_m": distance,
                            "valid_fraction": valid_fraction,
                            "temporal_std_mm": temporal_std,
                            "spatial_coherence": coh_value,
                            "eligible": bool(eligible),
                        }
                        candidate_rows.append(record)
                        candidates.append((record, settlement))
                eligible_candidates = [item for item in candidates if item[0]["eligible"]]
                if not eligible_candidates:
                    raise RuntimeError(
                        f"Point {point.id}: no candidate within {search_radius} pixels meets "
                        f"valid_fraction>={min_valid_fraction} and coherence>={min_coherence}."
                    )
                record, settlement = min(
                    eligible_candidates,
                    key=lambda item: (item[0]["distance_m"], -item[0]["valid_fraction"], -item[0]["spatial_coherence"]),
                )
                finite_idx = np.flatnonzero(np.isfinite(settlement))
                if finite_idx.size == 0:
                    raise RuntimeError(f"Point {point.id}: selected series contains no finite sample.")
                relative = settlement - settlement[finite_idx[0]]
                outlier, threshold = hampel_mask(
                    relative, hampel_window, hampel_sigma, hampel_floor_mm
                )
                clean = relative.copy()
                clean[outlier] = np.nan
                selected = dict(record)
                selected.update(
                    {
                        "vertical_source_sign": vertical_sign,
                        "settlement_conversion_factor_mm_per_m": factor,
                        "outlier_count": int(outlier.sum()),
                        "outlier_fraction": float(outlier.mean()),
                        "outlier_threshold_mm": threshold,
                    }
                )
                selected_rows.append(selected)
                for date, raw_value, clean_value, is_outlier in zip(dates, relative, clean, outlier):
                    series_rows.append(
                        {
                            "id": int(point.id),
                            "date": date.strftime("%Y%m%d"),
                            "settlement_mm_positive_down_raw": raw_value,
                            "settlement_mm_positive_down_clean": clean_value,
                            "outlier_removed": bool(is_outlier),
                            "row": record["row"],
                            "col": record["col"],
                            "distance_m": record["distance_m"],
                            "spatial_coherence": record["spatial_coherence"],
                        }
                    )
    finally:
        if coherence is not None:
            coherence.close()
    return pd.DataFrame(candidate_rows), pd.DataFrame(selected_rows), pd.DataFrame(series_rows)


def calculate_stats(leveling: pd.DataFrame, series: pd.DataFrame, selected: pd.DataFrame):
    origin = LEVEL_DATES.min()
    lev_x = (LEVEL_DATES - origin).days.to_numpy(dtype=float) / 365.25
    rows = []
    for point in leveling.itertuples(index=False):
        pid = int(point.id)
        heights = np.asarray([getattr(point, col) for col in LEVEL_COLS], dtype=float)
        lev_y = (heights[0] - heights) * 1000.0
        sub = series[series["id"] == pid].copy()
        sub["date_dt"] = pd.to_datetime(sub["date"].astype(str), format="%Y%m%d")
        clean = sub[~sub["outlier_removed"]].copy()
        ins_x = (clean["date_dt"] - origin).dt.days.to_numpy(dtype=float) / 365.25
        ins_y = clean["settlement_mm_positive_down_clean"].to_numpy(dtype=float)
        lev_fit = linear_fit_ci(lev_x, lev_y, lev_x)
        ins_fit = linear_fit_ci(ins_x, ins_y, ins_x)
        lev_at_ins = lev_fit["intercept"] + lev_fit["slope"] * ins_x
        connected = ins_y + (lev_at_ins[0] - ins_y[0])
        rmse_extrap = float(np.sqrt(np.mean((connected - lev_at_ins) ** 2)))
        time_r = float(np.corrcoef(ins_x, ins_y)[0, 1]) if np.std(ins_y) > 0 else np.nan
        spatial = selected[selected["id"] == pid].iloc[0]
        rows.append(
            {
                "id": pid,
                "lon": float(point.lon),
                "lat": float(point.lat),
                "leveling_slope_mm_yr": lev_fit["slope"],
                "leveling_slope_ci95_low_mm_yr": lev_fit["slope_ci_low"],
                "leveling_slope_ci95_high_mm_yr": lev_fit["slope_ci_high"],
                "insar_slope_mm_yr": ins_fit["slope"],
                "insar_slope_ci95_low_mm_yr": ins_fit["slope_ci_low"],
                "insar_slope_ci95_high_mm_yr": ins_fit["slope_ci_high"],
                "slope_difference_mm_yr": ins_fit["slope"] - lev_fit["slope"],
                "trend_sign_consistent": bool(np.sign(ins_fit["slope"]) == np.sign(lev_fit["slope"])),
                "insar_time_R": time_r,
                "rmse_to_leveling_extrapolation_after_anchor_mm": rmse_extrap,
                "distance_m": float(spatial["distance_m"]),
                "spatial_coherence": float(spatial["spatial_coherence"]),
                "valid_fraction": float(spatial["valid_fraction"]),
                "outlier_count": int(spatial["outlier_count"]),
                "outlier_fraction": float(spatial["outlier_fraction"]),
            }
        )
    stats = pd.DataFrame(rows)
    stats["quality_score"] = (
        normalize_rank(stats["valid_fraction"], ascending=False)
        + normalize_rank(stats["spatial_coherence"], ascending=False)
        + normalize_rank(stats["outlier_fraction"], ascending=True)
        + normalize_rank(stats["distance_m"], ascending=True)
    )
    stats["quality_rank"] = stats["quality_score"].rank(ascending=True, method="first").astype(int)
    return stats.sort_values("id").reset_index(drop=True)


def plot_comparison(
    leveling: pd.DataFrame,
    series: pd.DataFrame,
    stats: pd.DataFrame,
    ids: list[int],
    out_base: Path,
    display_map: dict[int, int] | None = None,
):
    origin = LEVEL_DATES.min()
    lev_x = (LEVEL_DATES - origin).days.to_numpy(dtype=float) / 365.25
    ncols = 3 if len(ids) <= 6 else 4
    nrows = int(math.ceil(len(ids) / ncols))
    # Six-panel manuscript figure is created at final double-column width
    # (~183 mm); the eight-panel diagnostic remains larger for inspection.
    figsize = (7.2, 5.25) if len(ids) <= 6 else (12.0, 5.9)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for panel_index, (ax, pid) in enumerate(zip(axes, ids)):
        point = leveling[leveling["id"] == pid].iloc[0]
        heights = point[LEVEL_COLS].to_numpy(dtype=float)
        lev_y = (heights[0] - heights) * 1000.0
        sub = series[series["id"] == pid].copy()
        sub["date_dt"] = pd.to_datetime(sub["date"].astype(str), format="%Y%m%d")
        clean = sub[~sub["outlier_removed"]].copy()
        rejected = sub[sub["outlier_removed"]].copy()
        ins_x = (clean["date_dt"] - origin).dt.days.to_numpy(dtype=float) / 365.25
        ins_y = clean["settlement_mm_positive_down_clean"].to_numpy(dtype=float)
        full_dates = pd.date_range(LEVEL_DATES.min(), clean["date_dt"].max(), periods=450)
        full_x = (full_dates - origin).days.to_numpy(dtype=float) / 365.25
        fit = linear_fit_ci(lev_x, lev_y, full_x)
        fit_at_start = fit["intercept"] + fit["slope"] * ins_x[0]
        connected = ins_y + (fit_at_start - ins_y[0])
        ax.fill_between(
            full_dates,
            fit["ci_low"],
            fit["ci_high"],
            color=PALETTE["band"],
            alpha=0.35,
            lw=0,
            label="Leveling 95% CI",
        )
        observed = full_dates <= LEVEL_DATES.max()
        ax.plot(full_dates[observed], fit["pred"][observed], color=PALETTE["level"], lw=1.7, label="Leveling fit")
        ax.plot(full_dates[~observed], fit["pred"][~observed], color=PALETTE["level"], lw=1.3, ls="--", label="Leveling extrapolation")
        ax.scatter(LEVEL_DATES, lev_y, color=PALETTE["level"], s=24, zorder=4, label="Leveling")
        ax.plot(clean["date_dt"], connected, color=PALETTE["insar"], lw=0.7, alpha=0.75)
        ax.scatter(clean["date_dt"], connected, color=PALETTE["insar"], s=8, zorder=3, label="InSAR")
        if not rejected.empty:
            rejected_x = (rejected["date_dt"] - origin).dt.days.to_numpy(dtype=float) / 365.25
            rejected_raw = rejected["settlement_mm_positive_down_raw"].to_numpy(dtype=float)
            rejected_connected = rejected_raw + (fit_at_start - ins_y[0])
            ax.scatter(rejected["date_dt"], rejected_connected, marker="x", color=PALETTE["outlier"], s=22, lw=1.0, label="Flagged spike")
        row = stats[stats["id"] == pid].iloc[0]
        label_id = display_map.get(pid, pid) if display_map else pid
        ax.text(
            0.025,
            0.97,
            f"Leveling: {row.leveling_slope_mm_yr:.1f} mm yr$^{{-1}}$\n"
            f"InSAR: {row.insar_slope_mm_yr:.1f} mm yr$^{{-1}}$\n"
            f"d = {row.distance_m:.1f} m; coh. = {row.spatial_coherence:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=5.8 if len(ids) <= 6 else 6.8,
            bbox=dict(boxstyle="round,pad=0.22", facecolor="white", edgecolor="0.65", alpha=0.9),
        )
        ax.set_title(f"Point {label_id}", pad=2, fontsize=8 if len(ids) <= 6 else 9)
        if panel_index % ncols == 0:
            ax.set_ylabel("Settlement (mm, positive down)", fontsize=7 if len(ids) <= 6 else 8)
        else:
            ax.set_ylabel("")
        ax.grid(True, color="0.85", lw=0.6)
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.tick_params(axis="both", labelsize=6.5 if len(ids) <= 6 else 7.5)
        ax.tick_params(axis="x", rotation=0)
        ax.set_xlim(pd.Timestamp("2012-12-01"), pd.Timestamp("2026-04-01"))
    for ax in axes[len(ids):]:
        ax.set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    unique = dict(zip(labels, handles))
    order = [
        "Leveling fit",
        "Leveling extrapolation",
        "Leveling 95% CI",
        "Leveling",
        "InSAR",
        "Flagged spike",
    ]
    fig.legend(
        [unique[x] for x in order if x in unique],
        [x for x in order if x in unique],
        loc="lower center",
        ncol=3 if len(ids) <= 6 else 6,
        bbox_to_anchor=(0.5, 0.012),
        fontsize=6.3 if len(ids) <= 6 else 7.2,
    )
    bottom = 0.10 if len(ids) <= 6 else 0.065
    fig.tight_layout(rect=(0, bottom, 1, 1), pad=0.8, w_pad=0.8, h_pad=1.0)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        ".svg": {},
        ".pdf": {},
        ".png": {"dpi": 400},
        ".tiff": {"dpi": 600},
    }.items():
        fig.savefig(out_base.with_suffix(suffix), bbox_inches="tight", **kwargs)
    plt.close(fig)


def distance_summary(frame: pd.DataFrame) -> dict[str, float]:
    values = frame["distance_m"].to_numpy(dtype=float)
    return {
        "minimum_m": float(np.min(values)),
        "maximum_m": float(np.max(values)),
        "mean_m": float(np.mean(values)),
        "median_m": float(np.median(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", required=True, type=Path)
    parser.add_argument("--era5-h5", required=True, type=Path)
    parser.add_argument(
        "--coherence-tif",
        default=None,
        help="Optional coherence GeoTIFF. No local-drive auto-discovery is performed.",
    )
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--search-radius-px", type=int, default=2)
    parser.add_argument("--min-valid-fraction", type=float, default=0.95)
    parser.add_argument("--min-coherence", type=float, default=0.30)
    parser.add_argument("--hampel-window", type=int, default=7)
    parser.add_argument("--hampel-sigma", type=float, default=4.5)
    parser.add_argument("--hampel-floor-mm", type=float, default=8.0)
    parser.add_argument("--select-count", type=int, default=6)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    leveling = read_leveling(args.points)
    coherence_path = find_coherence_tif(args.coherence_tif)
    candidates, selected, series = inspect_and_select(
        args.era5_h5,
        leveling,
        coherence_path,
        args.search_radius_px,
        args.min_valid_fraction,
        args.min_coherence,
        args.hampel_window,
        args.hampel_sigma,
        args.hampel_floor_mm,
    )
    stats = calculate_stats(leveling, series, selected)
    best = stats.nsmallest(args.select_count, "quality_rank").sort_values("quality_rank")
    best_ids = best["id"].astype(int).tolist()
    display_map = {pid: idx for idx, pid in enumerate(best_ids, 1)}

    candidates.to_csv(args.out / "candidate_pixels.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(args.out / "selected_nearest_quality_pixels.csv", index=False, encoding="utf-8-sig")
    series.to_csv(args.out / "all8_nearest_quality_timeseries.csv", index=False, encoding="utf-8-sig")
    stats.to_csv(args.out / "all8_trend_and_qc_stats.csv", index=False, encoding="utf-8-sig")
    best.assign(new_point=[display_map[x] for x in best_ids]).to_csv(
        args.out / "best6_quality_selected_stats.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        {"new_point": range(1, len(best_ids) + 1), "original_point": best_ids}
    ).to_csv(args.out / "best6_renumbering.csv", index=False, encoding="utf-8-sig")

    all_summary = distance_summary(selected)
    best_summary = distance_summary(best)
    audit = {
        "selection_rule": (
            "nearest pixel within the search window satisfying fixed validity and coherence thresholds; "
            "agreement metrics are calculated only after pixel selection"
        ),
        "point_ranking_rule": (
            "valid fraction, spatial coherence, temporal outlier fraction and pixel distance; "
            "R, RMSE and leveling agreement are excluded from selection"
        ),
        "h5_vertical_sign": str(selected["vertical_source_sign"].iloc[0]),
        "settlement_positive_direction": "down",
        "coherence_tif": str(coherence_path) if coherence_path else None,
        "all8_distance_summary": all_summary,
        "best6_distance_summary": best_summary,
        "best6_original_ids": best_ids,
        "thresholds": {
            "search_radius_px": args.search_radius_px,
            "min_valid_fraction": args.min_valid_fraction,
            "min_coherence": args.min_coherence,
            "hampel_window": args.hampel_window,
            "hampel_sigma": args.hampel_sigma,
            "hampel_floor_mm": args.hampel_floor_mm,
        },
        "limitations": [
            "Leveling (2013-2015) and InSAR (2017-2025) do not overlap in time.",
            "The InSAR series is vertically anchored to the leveling-only extrapolation at its first epoch for display.",
            "RMSE to the extrapolated line is a trend-divergence diagnostic, not an independent accuracy metric.",
        ],
    }
    (args.out / "selection_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(
        [
            {"set": "all8", **all_summary},
            {"set": "best6_quality_selected", **best_summary},
        ]
    ).to_csv(args.out / "pixel_distance_summary.csv", index=False, encoding="utf-8-sig")

    plot_comparison(
        leveling,
        series,
        stats,
        leveling["id"].astype(int).tolist(),
        args.out / "all8_nearest_quality_trend_comparison",
    )
    plot_comparison(
        leveling,
        series,
        stats,
        best_ids,
        args.out / "best6_nearest_quality_trend_comparison",
        display_map,
    )
    print("Written:", args.out)
    print("Best 6 (quality-only selection):", display_map)
    print("All 8 distance summary:", all_summary)
    print("Best 6 distance summary:", best_summary)
    print(stats.to_string(index=False))


if __name__ == "__main__":
    main()
