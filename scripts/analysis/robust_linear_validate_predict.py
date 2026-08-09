# -*- coding: utf-8 -*-
"""
Robust linear validation and prediction for SBAS-InSAR settlement.

Purpose
-------
This script improves the linear settlement prediction workflow by comparing:
  1) ordinary least squares linear trend (OLS)
  2) Huber robust linear trend

Validation split
----------------
Train dates: dates <= TRAIN_END_DATE
Validation dates: dates > TRAIN_END_DATE
Validation pixels are sampled independently from the full valid-pixel set.

Final prediction
----------------
After validation, the script refits trends using all 2017-2025 dates and exports:
  robust_linear_subsidence_2050_m.tif
  robust_linear_subsidence_2100_m.tif
  robust_linear_subsidence_rate_m_per_year.tif
  ols_linear_subsidence_2050_m.tif
  ols_linear_subsidence_2100_m.tif
  ols_linear_subsidence_rate_m_per_year.tif

Positive settlement_m means subsidence. Future subsidence is clipped to >= 0.

The source NPZ contains processed InSAR time-series values and is intentionally
not distributed with this repository. Supply it with --input-npz.
"""
import argparse
from pathlib import Path
import csv
import datetime as dt
import json
import math

import numpy as np
import rasterio
from rasterio.transform import Affine

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


REPO_ROOT = Path(__file__).resolve().parents[2]
NPZ = REPO_ROOT / "private_inputs" / "insar_settlement_timeseries_tc06_20170312_20251231.npz"
OUT_DIR = REPO_ROOT / "outputs" / "robust_linear"

TRAIN_END_DATE = "20231231"
N_VALID_PIXELS = 20000
SEED = 20260622

# Robust regression controls.
HUBER_K = 1.345
HUBER_ITERS = 8

# Final full-raster processing chunk. Increase if your RAM is comfortable.
CHUNK_PIXELS = 100000

# Optional long-horizon clipping for future cumulative subsidence.
# Set to None if you do not want clipping.
CAP_2050_M = 1.5
CAP_2100_M = 3.0


def dec_year(s):
    y = int(s[:4])
    m = int(s[4:6])
    d = int(s[6:8])
    date = dt.date(y, m, d)
    start = dt.date(y, 1, 1)
    end = dt.date(y + 1, 1, 1)
    return y + (date - start).days / ((end - start).days)


def metrics(obs, pred):
    obs = np.asarray(obs, dtype="float64")
    pred = np.asarray(pred, dtype="float64")
    ok = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[ok]
    pred = pred[ok]
    if obs.size == 0:
        return {"n": 0, "rmse_m": np.nan, "mae_m": np.nan, "bias_m": np.nan, "r2": np.nan}
    err = pred - obs
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((obs - np.mean(obs)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return {"n": int(obs.size), "rmse_m": rmse, "mae_m": mae, "bias_m": bias, "r2": r2}


def ols_fit(years, y):
    """
    Fit y = intercept + slope * year for many pixels.

    Parameters
    ----------
    years : (T,) float64
    y : (T, N) float32/float64

    Returns
    -------
    intercept, slope : (N,) float64
    """
    x = np.asarray(years, dtype="float64")
    yy = np.asarray(y, dtype="float64")
    x_mean = float(np.mean(x))
    xc = x - x_mean
    denom = float(np.sum(xc ** 2))
    y_mean = np.nanmean(yy, axis=0)
    slope = np.nansum((yy - y_mean[None, :]) * xc[:, None], axis=0) / denom
    intercept = y_mean - slope * x_mean
    return intercept, slope


def huber_fit(years, y, k=HUBER_K, n_iter=HUBER_ITERS):
    """
    Huber robust linear regression for many pixels with IRLS.

    This is more robust than OLS against intermittent SBAS jumps/outliers while
    remaining fast enough for raster-scale chunk processing.
    """
    x = np.asarray(years, dtype="float64")
    yy = np.asarray(y, dtype="float64")
    n = yy.shape[1]

    intercept, slope = ols_fit(x, yy)
    eps = 1e-9

    for _ in range(n_iter):
        pred = intercept[None, :] + slope[None, :] * x[:, None]
        resid = yy - pred
        med = np.nanmedian(resid, axis=0)
        mad = np.nanmedian(np.abs(resid - med[None, :]), axis=0)
        scale = 1.4826 * mad + eps
        threshold = k * scale
        abs_resid = np.abs(resid)
        w = np.minimum(1.0, threshold[None, :] / (abs_resid + eps))
        w = np.where(np.isfinite(yy), w, 0.0)

        sw = np.sum(w, axis=0) + eps
        sx = np.sum(w * x[:, None], axis=0)
        sy = np.sum(w * yy, axis=0)
        sxx = np.sum(w * (x[:, None] ** 2), axis=0)
        sxy = np.sum(w * x[:, None] * yy, axis=0)
        den = sw * sxx - sx ** 2
        bad = np.abs(den) < eps
        new_slope = (sw * sxy - sx * sy) / np.where(bad, np.nan, den)
        new_intercept = (sy - new_slope * sx) / sw

        slope = np.where(np.isfinite(new_slope), new_slope, slope)
        intercept = np.where(np.isfinite(new_intercept), new_intercept, intercept)

    return intercept, slope


def predict(intercept, slope, years):
    years = np.asarray(years, dtype="float64")
    return intercept[None, :] + slope[None, :] * years[:, None]


def summarize_values(name, arr):
    arr = np.asarray(arr)
    v = arr[np.isfinite(arr)]
    if v.size == 0:
        return {"name": name, "n": 0}
    out = {
        "name": name,
        "n": int(v.size),
        "min": float(np.min(v)),
        "max": float(np.max(v)),
        "mean": float(np.mean(v)),
        "std": float(np.std(v)),
    }
    for q in [1, 5, 25, 50, 75, 95, 99]:
        out[f"p{q}"] = float(np.percentile(v, q))
    return out


def write_sparse_tif(values, rows, cols, height, width, geotransform, path):
    transform = Affine.from_gdal(*geotransform)
    profile = {
        "driver": "GTiff",
        "height": int(height),
        "width": int(width),
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "compress": "lzw",
        "tiled": True,
        "nodata": np.float32(-9999.0),
    }
    arr = np.full((height, width), -9999.0, dtype="float32")
    arr[rows, cols] = np.where(np.isfinite(values), values, -9999.0).astype("float32")
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr, 1)
    print("wrote:", path)


def main():
    global NPZ, OUT_DIR, TRAIN_END_DATE, N_VALID_PIXELS, SEED
    global HUBER_K, HUBER_ITERS, CHUNK_PIXELS, CAP_2050_M, CAP_2100_M

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-npz", type=Path, default=NPZ)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--train-end-date", default=TRAIN_END_DATE)
    parser.add_argument("--validation-pixels", type=int, default=N_VALID_PIXELS)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--huber-k", type=float, default=HUBER_K)
    parser.add_argument("--huber-iters", type=int, default=HUBER_ITERS)
    parser.add_argument("--chunk-pixels", type=int, default=CHUNK_PIXELS)
    parser.add_argument("--cap-2050-m", type=float, default=CAP_2050_M,
                        help="Use a negative value to disable clipping.")
    parser.add_argument("--cap-2100-m", type=float, default=CAP_2100_M,
                        help="Use a negative value to disable clipping.")
    args = parser.parse_args()

    NPZ = args.input_npz
    OUT_DIR = args.out_dir
    TRAIN_END_DATE = args.train_end_date
    N_VALID_PIXELS = args.validation_pixels
    SEED = args.seed
    HUBER_K = args.huber_k
    HUBER_ITERS = args.huber_iters
    CHUNK_PIXELS = args.chunk_pixels
    CAP_2050_M = None if args.cap_2050_m < 0 else args.cap_2050_m
    CAP_2100_M = None if args.cap_2100_m < 0 else args.cap_2100_m
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if not NPZ.is_file():
        raise FileNotFoundError(
            f"Processed InSAR NPZ not found: {NPZ}. "
            "This private input is not included in the public repository."
        )

    print("loading:", NPZ)
    data = np.load(NPZ)
    settlement = data["settlement_m"].astype("float32")
    dates = data["dates"].astype(str)
    rows = data["rows"].astype("int32")
    cols = data["cols"].astype("int32")
    raster_shape = tuple(data["raster_shape"].astype(int))
    geotransform = data["geotransform"].astype(float)
    height, width = raster_shape
    n_dates, n_pixels = settlement.shape
    years = np.array([dec_year(d) for d in dates], dtype="float64")

    print("settlement shape:", settlement.shape)
    print("date range:", dates[0], "to", dates[-1])
    print("valid pixels:", n_pixels)

    train_date_idx = np.where(dates <= TRAIN_END_DATE)[0]
    valid_date_idx = np.where(dates > TRAIN_END_DATE)[0]
    if train_date_idx.size == 0 or valid_date_idx.size == 0:
        raise ValueError("The train-end date must leave at least one date in each split.")
    print("train dates:", dates[train_date_idx[0]], "to", dates[train_date_idx[-1]], "count=", train_date_idx.size)
    print("valid dates:", dates[valid_date_idx[0]], "to", dates[valid_date_idx[-1]], "count=", valid_date_idx.size)

    rng = np.random.default_rng(SEED)
    valid_pix = rng.choice(n_pixels, size=min(N_VALID_PIXELS, n_pixels), replace=False)
    print("validation pixels:", valid_pix.size)

    y_train = settlement[train_date_idx[:, None], valid_pix[None, :]]
    y_valid = settlement[valid_date_idx[:, None], valid_pix[None, :]]
    train_years = years[train_date_idx]
    valid_years = years[valid_date_idx]

    print("fitting validation OLS...")
    ols_intercept, ols_slope = ols_fit(train_years, y_train)
    ols_valid = predict(ols_intercept, ols_slope, valid_years).astype("float32")

    print("fitting validation Huber robust linear...")
    huber_intercept, huber_slope = huber_fit(
        train_years, y_train, k=HUBER_K, n_iter=HUBER_ITERS
    )
    huber_valid = predict(huber_intercept, huber_slope, valid_years).astype("float32")

    overall_ols = metrics(y_valid.ravel(), ols_valid.ravel())
    overall_huber = metrics(y_valid.ravel(), huber_valid.ravel())
    final_ols = metrics(y_valid[-1], ols_valid[-1])
    final_huber = metrics(y_valid[-1], huber_valid[-1])

    print("OLS overall:", overall_ols)
    print("Huber overall:", overall_huber)
    print("OLS final date:", final_ols)
    print("Huber final date:", final_huber)

    per_date_rows = []
    for oi, ti in enumerate(valid_date_idx):
        m_ols = metrics(y_valid[oi], ols_valid[oi])
        m_huber = metrics(y_valid[oi], huber_valid[oi])
        per_date_rows.append({
            "date": dates[ti],
            "decimal_year": float(years[ti]),
            "ols_rmse_m": m_ols["rmse_m"],
            "ols_mae_m": m_ols["mae_m"],
            "ols_bias_m": m_ols["bias_m"],
            "ols_r2": m_ols["r2"],
            "huber_rmse_m": m_huber["rmse_m"],
            "huber_mae_m": m_huber["mae_m"],
            "huber_bias_m": m_huber["bias_m"],
            "huber_r2": m_huber["r2"],
            "n": m_ols["n"],
        })

    with open(OUT_DIR / "per_date_metrics.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_date_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_date_rows)

    summary_rows = [
        {"model": "OLS_linear", "split": "all_validation_dates", **overall_ols},
        {"model": "Huber_robust_linear", "split": "all_validation_dates", **overall_huber},
        {"model": "OLS_linear", "split": f"final_date_{dates[valid_date_idx[-1]]}", **final_ols},
        {"model": "Huber_robust_linear", "split": f"final_date_{dates[valid_date_idx[-1]]}", **final_huber},
    ]
    with open(OUT_DIR / "validation_summary.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    if plt is not None:
        print("plotting validation figures...")
        sample = np.linspace(0, valid_pix.size - 1, min(12, valid_pix.size)).astype(int)
        fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharex=True)
        axes = axes.ravel()
        for ax, j in zip(axes, sample):
            pix = valid_pix[j]
            ax.plot(years[train_date_idx], settlement[train_date_idx, pix], color="0.65", lw=1.0, label="train obs")
            ax.plot(years[valid_date_idx], y_valid[:, j], "ko", ms=2.5, label="valid obs")
            ax.plot(years[valid_date_idx], ols_valid[:, j], color="#1f77b4", lw=1.2, ls="--", label="OLS")
            ax.plot(years[valid_date_idx], huber_valid[:, j], color="#d62728", lw=1.5, label="Huber")
            ax.set_title(f"pixel {pix}", fontsize=8)
            ax.grid(alpha=0.25)
        axes[0].legend(fontsize=7)
        fig.supxlabel("year")
        fig.supylabel("settlement_m")
        fig.tight_layout()
        fig.savefig(OUT_DIR / "validation_sample_curves_ols_vs_huber.png", dpi=220)
        plt.close(fig)

        obs_final = y_valid[-1]
        fig, ax = plt.subplots(figsize=(6.5, 6))
        ax.scatter(obs_final, ols_valid[-1], s=4, alpha=0.18, label="OLS", color="#1f77b4")
        ax.scatter(obs_final, huber_valid[-1], s=4, alpha=0.18, label="Huber", color="#d62728")
        ok = np.isfinite(obs_final)
        if ok.any():
            mn = float(np.nanpercentile(obs_final[ok], 1))
            mx = float(np.nanpercentile(obs_final[ok], 99))
            ax.plot([mn, mx], [mn, mx], "k--", lw=1)
            ax.set_xlim(mn, mx)
            ax.set_ylim(mn, mx)
        ax.set_xlabel("Observed settlement_m")
        ax.set_ylabel("Predicted settlement_m")
        ax.set_title(f"Final validation date: {dates[valid_date_idx[-1]]}")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(OUT_DIR / "final_date_observed_vs_predicted_ols_vs_huber.png", dpi=220)
        plt.close(fig)
    else:
        print("matplotlib is not installed; skipped plots.")

    print("refitting full raster with all dates and exporting future TIFs...")
    ols_rate = np.full(n_pixels, np.nan, dtype="float32")
    huber_rate = np.full(n_pixels, np.nan, dtype="float32")

    all_years = years
    for start in range(0, n_pixels, CHUNK_PIXELS):
        end = min(start + CHUNK_PIXELS, n_pixels)
        print(f"  chunk {start}:{end}")
        y_chunk = settlement[:, start:end]
        _, slope_ols = ols_fit(all_years, y_chunk)
        _, slope_huber = huber_fit(
            all_years, y_chunk, k=HUBER_K, n_iter=HUBER_ITERS
        )
        ols_rate[start:end] = slope_ols.astype("float32")
        huber_rate[start:end] = slope_huber.astype("float32")

    # Positive settlement slope means subsidence rate.
    ols_sub_rate = np.maximum(ols_rate, 0).astype("float32")
    huber_sub_rate = np.maximum(huber_rate, 0).astype("float32")

    ols_sub_2050 = (ols_sub_rate * 25.0).astype("float32")
    ols_sub_2100 = (ols_sub_rate * 75.0).astype("float32")
    huber_sub_2050 = (huber_sub_rate * 25.0).astype("float32")
    huber_sub_2100 = (huber_sub_rate * 75.0).astype("float32")

    if CAP_2050_M is not None:
        ols_sub_2050 = np.minimum(ols_sub_2050, CAP_2050_M)
        huber_sub_2050 = np.minimum(huber_sub_2050, CAP_2050_M)
    if CAP_2100_M is not None:
        ols_sub_2100 = np.minimum(ols_sub_2100, CAP_2100_M)
        huber_sub_2100 = np.minimum(huber_sub_2100, CAP_2100_M)

    write_sparse_tif(huber_sub_rate, rows, cols, height, width, geotransform, OUT_DIR / "robust_linear_subsidence_rate_m_per_year.tif")
    write_sparse_tif(huber_sub_2050, rows, cols, height, width, geotransform, OUT_DIR / "robust_linear_subsidence_2050_m.tif")
    write_sparse_tif(huber_sub_2100, rows, cols, height, width, geotransform, OUT_DIR / "robust_linear_subsidence_2100_m.tif")
    write_sparse_tif(ols_sub_rate, rows, cols, height, width, geotransform, OUT_DIR / "ols_linear_subsidence_rate_m_per_year.tif")
    write_sparse_tif(ols_sub_2050, rows, cols, height, width, geotransform, OUT_DIR / "ols_linear_subsidence_2050_m.tif")
    write_sparse_tif(ols_sub_2100, rows, cols, height, width, geotransform, OUT_DIR / "ols_linear_subsidence_2100_m.tif")

    stats = [
        summarize_values("huber_subsidence_rate_m_per_year", huber_sub_rate),
        summarize_values("huber_subsidence_2050_m", huber_sub_2050),
        summarize_values("huber_subsidence_2100_m", huber_sub_2100),
        summarize_values("ols_subsidence_rate_m_per_year", ols_sub_rate),
        summarize_values("ols_subsidence_2050_m", ols_sub_2050),
        summarize_values("ols_subsidence_2100_m", ols_sub_2100),
    ]
    with open(OUT_DIR / "future_prediction_stats.csv", "w", newline="", encoding="utf-8-sig") as f:
        keys = sorted({k for row in stats for k in row.keys()})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(stats)

    meta = {
        "dataset": str(NPZ),
        "train_end_date_for_validation": TRAIN_END_DATE,
        "validation_pixels": int(valid_pix.size),
        "huber_k": HUBER_K,
        "huber_iters": HUBER_ITERS,
        "cap_2050_m": CAP_2050_M,
        "cap_2100_m": CAP_2100_M,
        "validation_summary": summary_rows,
        "future_definition": "future cumulative subsidence = max(slope, 0) * (target_year - 2025)",
        "note": "Use validation metrics to decide whether OLS or Huber should be the main linear prediction. Long-term 2100 extrapolation is a scenario, not deterministic forecast.",
    }
    (OUT_DIR / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("done. output:", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
