# -*- coding: utf-8 -*-
"""Identify how the original 8-point CSV was sampled from the 249 GeoTIFFs."""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio


def read_points(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, encoding="utf-8-sig", skipinitialspace=True)
    frame = frame.rename(
        columns={frame.columns[0]: "id", frame.columns[6]: "lat", frame.columns[7]: "lon"}
    )
    frame["id"] = frame["id"].astype(int)
    return frame[["id", "lon", "lat"]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--points", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument(
        "--tif-glob",
        required=True,
        help="Glob for the 249 private vertical-displacement GeoTIFFs.",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=2)
    args = parser.parse_args()

    files = sorted(Path(p) for p in glob.glob(args.tif_glob))
    if len(files) != 249:
        raise RuntimeError(f"Expected 249 original TIFFs, found {len(files)}")
    dates = [re.search(r"(\d{8})", p.name).group(1) for p in files]
    points = read_points(args.points)
    original = pd.read_csv(args.series, encoding="utf-8-sig")
    original["id"] = original["id"].astype(int)
    original["date"] = original["date"].astype(str)
    value_col = "vertical_displacement_m"
    if value_col not in original:
        raise ValueError(f"Missing {value_col} in original series CSV")

    with rasterio.open(files[0]) as src:
        base = {
            int(p.id): src.index(float(p.lon), float(p.lat))
            for p in points.itertuples(index=False)
        }
        transform = src.transform

    stacks = {pid: [] for pid in base}
    for tif in files:
        with rasterio.open(tif) as src:
            for pid, (row, col) in base.items():
                r0, c0 = row - args.radius, col - args.radius
                block = src.read(
                    1,
                    window=((r0, row + args.radius + 1), (c0, col + args.radius + 1)),
                    boundless=True,
                    fill_value=np.nan,
                ).astype(float)
                stacks[pid].append(block)

    results = []
    for p in points.itertuples(index=False):
        pid = int(p.id)
        cube = np.stack(stacks[pid], axis=0)
        target_frame = original[original["id"] == pid].set_index("date").reindex(dates)
        target = target_frame[value_col].to_numpy(dtype=float)
        if np.isnan(target).all():
            raise RuntimeError(f"Point {pid}: original CSV dates do not match TIFF dates")
        candidates = []
        for i, dr in enumerate(range(-args.radius, args.radius + 1)):
            for j, dc in enumerate(range(-args.radius, args.radius + 1)):
                values = cube[:, i, j]
                rmse_same = float(np.sqrt(np.nanmean((values - target) ** 2)))
                rmse_flip = float(np.sqrt(np.nanmean((-values - target) ** 2)))
                candidates.append((min(rmse_same, rmse_flip), dr, dc, rmse_flip < rmse_same))
        for method, values in {
            "3x3_mean": np.nanmean(cube[:, 1:4, 1:4], axis=(1, 2)),
            "3x3_median": np.nanmedian(cube[:, 1:4, 1:4], axis=(1, 2)),
            "5x5_mean": np.nanmean(cube, axis=(1, 2)),
            "5x5_median": np.nanmedian(cube, axis=(1, 2)),
        }.items():
            rmse_same = float(np.sqrt(np.nanmean((values - target) ** 2)))
            rmse_flip = float(np.sqrt(np.nanmean((-values - target) ** 2)))
            candidates.append((min(rmse_same, rmse_flip), method, method, rmse_flip < rmse_same))
        candidates.sort(key=lambda item: item[0])
        best_rmse, a, b, sign_flipped = candidates[0]
        base_row, base_col = base[pid]
        if isinstance(a, int):
            selected_row, selected_col = base_row + a, base_col + b
            pixel_lon = transform.c + (selected_col + 0.5) * transform.a
            pixel_lat = transform.f + (selected_row + 0.5) * transform.e
            method = "single_pixel"
            dr, dc = a, b
        else:
            selected_row = selected_col = np.nan
            pixel_lon = pixel_lat = np.nan
            method = str(a)
            dr = dc = np.nan
        results.append(
            {
                "id": pid,
                "method": method,
                "row_offset_from_rasterio_index": dr,
                "col_offset_from_rasterio_index": dc,
                "base_row": base_row,
                "base_col": base_col,
                "selected_row": selected_row,
                "selected_col": selected_col,
                "pixel_lon_center": pixel_lon,
                "pixel_lat_center": pixel_lat,
                "sign_flipped_relative_to_tiff": bool(sign_flipped),
                "rmse_m": best_rmse,
                "exact_match_within_1e-7_m": bool(best_rmse < 1e-7),
                "second_best_rmse_m": float(candidates[1][0]),
            }
        )

    result = pd.DataFrame(results).sort_values("id")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.out, index=False, encoding="utf-8-sig")
    print(result.to_string(index=False))
    print("written:", args.out)


if __name__ == "__main__":
    main()
