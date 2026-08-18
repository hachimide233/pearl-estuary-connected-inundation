from __future__ import annotations

import argparse
import json
import math
import re
import zipfile
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from rasterio.warp import Resampling, reproject, transform_bounds
from rasterio.windows import Window, from_bounds
from scipy import ndimage


DEM_PATH: Path
LAND_PATH: Path
SCENARIO_PATH: Path
RATE_PATH: Path
SUB_PATHS: dict[int, Path]
POP_ARCHIVES: dict[str, Path]
GADM: Path
AUDIT_DIR: Path
OUT: Path

WGS84_A = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
DECAY_END_YEAR = 2100
DATUM_CI95_M = 0.116600812
TERRAIN_OFFSETS_M = (-1.0, -0.5, 0.0, 0.5, 1.0)


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    }
)


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


def mask_area_km2(mask: np.ndarray, row_area: np.ndarray) -> float:
    return float(np.dot(np.count_nonzero(mask, axis=1), row_area))


def valid_mask(dataset, array: np.ndarray) -> np.ndarray:
    result = np.isfinite(array)
    if dataset.nodata is not None:
        result &= array != dataset.nodata
    return result


def connected_ocean_flood(surface: np.ndarray, water: float, land: np.ndarray, valid_dem: np.ndarray, structure: np.ndarray) -> np.ndarray:
    low_land = land & valid_dem & (surface <= water)
    ocean = (~land) & valid_dem
    domain = ocean | low_land
    seeds = np.zeros_like(domain, dtype=bool)
    seeds[0, :] = ocean[0, :]
    seeds[-1, :] = ocean[-1, :]
    seeds[:, 0] = ocean[:, 0]
    seeds[:, -1] = ocean[:, -1]
    return ndimage.binary_propagation(seeds, structure=structure, mask=domain) & low_land


def write_raster(path: Path, data: np.ndarray, profile: dict, *, dtype: str, nodata, tags: dict[str, str]) -> None:
    target = profile.copy()
    target.update(driver="GTiff", dtype=dtype, count=1, nodata=nodata, compress="deflate", tiled=True)
    with rasterio.open(path, "w", **target) as dst:
        dst.write(data.astype(dtype), 1)
        dst.update_tags(**tags)


def safe_case(row: pd.Series) -> str:
    scenario = str(row["scenario"]).replace(".", "p").replace("-", "m")
    added = float(row["surge_m"])
    return f"{scenario}_{int(row['year'])}_added{added:.1f}m".replace(".", "p")


def load_main_scenarios() -> pd.DataFrame:
    frame = pd.read_csv(SCENARIO_PATH)
    frame = frame[
        frame["scenario"].isin(["SSP2-4.5", "SSP5-8.5"])
        & frame["year"].isin([2050, 2100])
        & frame["surge_m"].round(3).isin([0.0, 1.5, 3.0])
    ].copy()
    frame = frame.sort_values(["scenario", "year", "surge_m"]).drop_duplicates(["scenario", "year", "surge_m"])
    if len(frame) != 12:
        raise RuntimeError(f"Expected 12 main scenarios, found {len(frame)}")
    return frame.reset_index(drop=True)


def decaying_effective_years(target_year: int) -> float:
    elapsed = float(target_year - 2025)
    horizon = float(DECAY_END_YEAR - 2025)
    if elapsed <= 0:
        return 0.0
    elapsed = min(elapsed, horizon)
    return elapsed - elapsed**2 / (2.0 * horizon)


def clamp_window(window: Window, width: int, height: int) -> Window:
    c0 = max(0, int(math.floor(window.col_off)))
    r0 = max(0, int(math.floor(window.row_off)))
    c1 = min(width, int(math.ceil(window.col_off + window.width)))
    r1 = min(height, int(math.ceil(window.row_off + window.height)))
    if c1 <= c0 or r1 <= r0:
        raise ValueError("No overlap with population grid")
    return Window(c0, r0, c1 - c0, r1 - r0)


def extract_population(scenario: str, year: int) -> Path:
    target = OUT / "population_inputs" / f"{scenario}_{year}.tif"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return target
    archive = POP_ARCHIVES[scenario]
    with zipfile.ZipFile(archive) as zf:
        pattern = re.compile(rf"{scenario}_{year}\.tif$", re.I)
        matches = [n for n in zf.namelist() if pattern.search(n.replace("\\", "/"))]
        if len(matches) != 1:
            raise RuntimeError(f"Expected one {scenario}_{year}.tif in {archive}; found {matches}")
        target.write_bytes(zf.read(matches[0]))
    return target


def population_overlap(mask: np.ndarray, reference, population_path: Path) -> float:
    with rasterio.open(population_path) as pop:
        bounds = reference.bounds
        if reference.crs != pop.crs:
            bounds = transform_bounds(reference.crs, pop.crs, *bounds, densify_pts=21)
        window = clamp_window(from_bounds(*bounds, transform=pop.transform), pop.width, pop.height)
        values = pop.read(1, window=window, masked=True).astype("float64").filled(np.nan)
        values[~np.isfinite(values) | (values < 0)] = np.nan
        fraction = np.zeros(values.shape, dtype="float32")
        reproject(
            mask.astype("float32"),
            fraction,
            src_transform=reference.transform,
            src_crs=reference.crs,
            dst_transform=pop.window_transform(window),
            dst_crs=pop.crs,
            resampling=Resampling.average,
            src_nodata=None,
            dst_nodata=0.0,
        )
        return float(np.nansum(values * np.clip(fraction, 0.0, 1.0)))


def population_density_on_reference(reference, population_path: Path) -> np.ndarray:
    with rasterio.open(population_path) as pop:
        bounds = reference.bounds
        if reference.crs != pop.crs:
            bounds = transform_bounds(reference.crs, pop.crs, *bounds, densify_pts=21)
        window = clamp_window(from_bounds(*bounds, transform=pop.transform), pop.width, pop.height)
        values = pop.read(1, window=window, masked=True).astype("float32").filled(np.nan)
        values[~np.isfinite(values) | (values < 0)] = np.nan
        source_transform = pop.window_transform(window)
        source_area = row_pixel_areas_km2(source_transform, values.shape[0])
        density = np.nan_to_num(
            values / source_area[:, None], nan=0.0, posinf=0.0, neginf=0.0
        ).astype("float32")
        destination = np.zeros(reference.shape, dtype="float32")
        reproject(
            density,
            destination,
            src_transform=source_transform,
            src_crs=pop.crs,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            resampling=Resampling.bilinear,
            src_nodata=None,
            dst_nodata=0.0,
        )
        return np.maximum(np.nan_to_num(destination, nan=0.0), 0.0)


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(base.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(base.with_suffix(".tiff"), dpi=600, bbox_inches="tight")


def plot_common_domain(common: np.ndarray, land: np.ndarray, reference, audit: dict) -> None:
    fig = plt.figure(figsize=(7.2, 3.35))
    grid = fig.add_gridspec(1, 3, width_ratios=[1.65, 1.0, 0.9], wspace=0.42)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_admin = fig.add_subplot(grid[0, 1])
    ax_share = fig.add_subplot(grid[0, 2])

    step = 4
    display = np.zeros((land[::step, ::step].shape[0], land[::step, ::step].shape[1]), dtype="uint8")
    display[land[::step, ::step]] = 1
    display[common[::step, ::step]] = 2
    ax_map.imshow(
        display,
        extent=[reference.bounds.left, reference.bounds.right, reference.bounds.bottom, reference.bounds.top],
        origin="upper",
        interpolation="nearest",
        cmap=ListedColormap(["white", "#D9DDE1", "#2B7BBA"]),
        vmin=0,
        vmax=2,
        aspect="auto",
    )
    try:
        import pyogrio

        layer_names = [x[0] for x in pyogrio.list_layers(GADM)]
        layer = next((x for x in layer_names if x.endswith("_2") or x.upper().endswith("ADM_2")), layer_names[-1])
        admin = gpd.read_file(GADM, layer=layer, bbox=tuple(reference.bounds)).to_crs(reference.crs)
        admin.boundary.plot(ax=ax_map, color="#5F666D", linewidth=0.25, alpha=0.65)
    except Exception as exc:
        print("Administrative boundary overlay skipped:", exc)
    ax_map.set_xlabel("Longitude (°E)")
    ax_map.set_ylabel("Latitude (°N)")
    ax_map.set_title("Common analysis domain", loc="left", fontweight="bold")
    ax_map.text(0.01, 0.99, "a", transform=ax_map.transAxes, va="top", ha="left", fontweight="bold", fontsize=8)
    ax_map.scatter([], [], s=30, marker="s", color="#2B7BBA", label="Common land")
    ax_map.scatter([], [], s=30, marker="s", color="#D9DDE1", label="Excluded DEM-valid land")
    ax_map.legend(loc="lower left", fontsize=6.2, handlelength=0.8)
    ax_map.set_xlim(reference.bounds.left, reference.bounds.right)
    ax_map.set_ylim(reference.bounds.bottom, reference.bounds.top)

    admin_df = pd.read_csv(AUDIT_DIR / "common_domain_administrative_coverage.csv")
    admin_df = admin_df[admin_df["common_area_km2"] > 1].sort_values("common_area_km2", ascending=True).tail(6)
    ax_admin.barh(admin_df["administrative_unit"], admin_df["common_share_percent"], color="#4C9B8A")
    for y, value in enumerate(admin_df["common_share_percent"]):
        ax_admin.text(value + 1.2, y, f"{value:.0f}%", va="center", fontsize=6.2)
    ax_admin.set_xlim(0, max(100, admin_df["common_share_percent"].max() + 12))
    ax_admin.set_xlabel("Unit land inside common domain (%)")
    ax_admin.set_title("Administrative coverage", loc="left", fontweight="bold")
    ax_admin.text(-0.15, 0.99, "b", transform=ax_admin.transAxes, va="top", ha="left", fontweight="bold", fontsize=8)
    ax_admin.grid(axis="x", color="#E4E7EA", linewidth=0.5)

    shares = [
        audit["common_domain"]["share_of_dem_valid_land_percent"],
        audit["common_domain"]["share_of_built_up_area_percent"],
        audit["common_domain"]["share_of_population_2020_percent"],
    ]
    labels = ["Land", "Built-up", "Population"]
    bars = ax_share.bar(labels, shares, color=["#7A8793", "#D28E45", "#7B6EB2"], width=0.65)
    ax_share.bar_label(bars, labels=[f"{v:.1f}%" for v in shares], padding=2, fontsize=6.5)
    ax_share.set_ylim(0, max(shares) * 1.3)
    ax_share.set_ylabel("Share of reference domain (%)")
    ax_share.set_title("Representativeness", loc="left", fontweight="bold")
    ax_share.text(-0.17, 0.99, "c", transform=ax_share.transAxes, va="top", ha="left", fontweight="bold", fontsize=8)
    ax_share.tick_params(axis="x", rotation=25)
    ax_share.grid(axis="y", color="#E4E7EA", linewidth=0.5)
    save_figure(fig, OUT / "Figure_Common_analysis_domain")
    plt.close(fig)


def plot_priority(priority: np.ndarray, frequency: np.ndarray, common: np.ndarray, reference, threshold: float) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.05), constrained_layout=True)
    rows, cols = np.where(common)
    pad = 40
    r0, r1 = max(0, int(rows.min()) - pad), min(reference.height, int(rows.max()) + pad + 1)
    c0, c1 = max(0, int(cols.min()) - pad), min(reference.width, int(cols.max()) + pad + 1)
    left, top = reference.transform * (c0, r0)
    right, bottom = reference.transform * (c1, r1)
    extent = [left, right, bottom, top]
    step = 3
    common_show = common[r0:r1:step, c0:c1:step]
    priority_values = priority[r0:r1:step, c0:c1:step]
    frequency_values = frequency[r0:r1:step, c0:c1:step] * 100.0
    priority_show = np.ma.masked_where(~common_show | (priority_values <= 0), priority_values)
    frequency_show = np.ma.masked_where(~common_show | (frequency_values <= 0), frequency_values)
    for ax in axes:
        ax.imshow(
            np.ma.masked_where(~common_show, common_show),
            extent=extent,
            origin="upper",
            cmap=ListedColormap(["#E7E9EC"]),
            vmin=0,
            vmax=1,
            aspect="equal",
        )
    im0 = axes[0].imshow(priority_show, extent=extent, origin="upper", cmap="cividis", vmin=0, vmax=max(threshold, float(np.nanpercentile(priority[common], 99))), aspect="equal")
    recurrence_max = max(11.111, float(np.max(frequency[common]) * 100.0))
    im1 = axes[1].imshow(frequency_show, extent=extent, origin="upper", cmap="Blues", vmin=0, vmax=recurrence_max, aspect="equal")
    for idx, (ax, title) in enumerate(zip(axes, ["Priority index", "Top-decile recurrence across sensitivity cases"])):
        ax.set_xlabel("Longitude (°E)")
        if idx == 0:
            ax.set_ylabel("Latitude (°N)")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.text(0.01, 0.99, chr(97 + idx), transform=ax.transAxes, va="top", ha="left", fontweight="bold", fontsize=8, color="black")
    c0 = fig.colorbar(im0, ax=axes[0], fraction=0.036, pad=0.02)
    c0.set_label("PI (dimensionless)")
    c1 = fig.colorbar(im1, ax=axes[1], fraction=0.036, pad=0.02)
    c1.set_label("Sensitivity cases in top decile (%)")
    if not np.any((frequency >= 2 / 3) & common):
        axes[1].text(
            0.02,
            0.03,
            "No pixel recurred in ≥6 of 9 cases",
            transform=axes[1].transAxes,
            fontsize=6.3,
            bbox={"facecolor": "white", "edgecolor": "#8A9096", "boxstyle": "round,pad=0.25", "alpha": 0.92},
        )
    save_figure(fig, OUT / "Figure_Uncertainty_conditioned_priority_hotspots")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the audited common-domain coastal-inundation scenario analysis."
    )
    parser.add_argument("--dem", type=Path, required=True, help="Aligned Copernicus DEM raster.")
    parser.add_argument("--land-mask", type=Path, required=True, help="Aligned GSHHG land-mask raster.")
    parser.add_argument("--scenario-table", type=Path, required=True, help="Water-level scenario CSV.")
    parser.add_argument("--rate", type=Path, required=True, help="OLS subsidence-rate raster.")
    parser.add_argument("--subsidence-2050", type=Path, required=True, help="OLS 2050 cumulative-subsidence raster.")
    parser.add_argument("--subsidence-2100", type=Path, required=True, help="OLS 2100 cumulative-subsidence raster.")
    parser.add_argument("--ssp2-zip", type=Path, required=True, help="SSP2 population ZIP archive.")
    parser.add_argument("--ssp5-zip", type=Path, required=True, help="SSP5 population ZIP archive.")
    parser.add_argument("--gadm", type=Path, required=True, help="GADM GeoPackage.")
    parser.add_argument("--audit-dir", type=Path, required=True, help="Directory produced by audit_common_domain.py.")
    parser.add_argument("--out", type=Path, default=Path("analysis"), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    global DEM_PATH, LAND_PATH, SCENARIO_PATH, RATE_PATH
    global SUB_PATHS, POP_ARCHIVES, GADM, AUDIT_DIR, OUT
    args = parse_args()
    DEM_PATH = args.dem
    LAND_PATH = args.land_mask
    SCENARIO_PATH = args.scenario_table
    RATE_PATH = args.rate
    SUB_PATHS = {2050: args.subsidence_2050, 2100: args.subsidence_2100}
    POP_ARCHIVES = {"SSP2": args.ssp2_zip, "SSP5": args.ssp5_zip}
    GADM = args.gadm
    AUDIT_DIR = args.audit_dir
    OUT = args.out

    OUT.mkdir(parents=True, exist_ok=True)
    for path in [DEM_PATH, LAND_PATH, SCENARIO_PATH, RATE_PATH, GADM, *SUB_PATHS.values(), *POP_ARCHIVES.values()]:
        if not path.exists():
            raise FileNotFoundError(path)
    audit = json.loads((AUDIT_DIR / "common_domain_audit.json").read_text(encoding="utf-8"))
    scenarios = load_main_scenarios()

    with rasterio.open(DEM_PATH) as reference:
        dem = reference.read(1).astype("float32")
        dem_valid = valid_mask(reference, dem)
        profile = reference.profile.copy()
        row_area = row_pixel_areas_km2(reference.transform, reference.height)
        with rasterio.open(LAND_PATH) as ds:
            land = (ds.read(1) > 0) & dem_valid
        with rasterio.open(RATE_PATH) as ds:
            raw_rate = ds.read(1).astype("float32")
            support = valid_mask(ds, raw_rate)
        rate = np.where(support, np.maximum(raw_rate, 0.0), 0.0).astype("float32")
        continued = {}
        for year, path in SUB_PATHS.items():
            with rasterio.open(path) as ds:
                arr = ds.read(1).astype("float32")
                support &= valid_mask(ds, arr)
                continued[year] = np.where(valid_mask(ds, arr), np.maximum(arr, 0.0), 0.0).astype("float32")
        common = land & support

        write_raster(
            OUT / "common_analysis_land_mask.tif",
            common.astype("uint8"),
            profile,
            dtype="uint8",
            nodata=0,
            tags={"definition": "GSHHG land AND valid Copernicus DEM AND valid OLS rate/2100 support"},
        )

        structures = {
            4: ndimage.generate_binary_structure(2, 1),
            8: ndimage.generate_binary_structure(2, 2),
        }
        persistence_arrays = {}
        for year in (2050, 2100):
            cap = 1.5 if year == 2050 else 3.0
            persistence_arrays[(year, "stabilisation")] = np.zeros_like(rate)
            persistence_arrays[(year, "decaying_trend")] = np.minimum(rate * decaying_effective_years(year), cap).astype("float32")
            persistence_arrays[(year, "continued_trend")] = continued[year]

        central_cache: dict[tuple[str, str], np.ndarray] = {}
        scenario_rows = []
        print("Running persistence scenarios...", flush=True)
        for idx, row in scenarios.iterrows():
            water = float(row["water_level_used_egm2008_m"])
            case = safe_case(row)
            base_full = connected_ocean_flood(dem, water, land, dem_valid, structures[8])
            base = base_full & common
            central_cache[(case, "baseline")] = base_full
            pop_scenario = "SSP2" if str(row["scenario"]).startswith("SSP2") else "SSP5"
            pop_path = extract_population(pop_scenario, int(row["year"]))
            baseline_population = population_overlap(base, reference, pop_path)
            for persistence in ("stabilisation", "decaying_trend", "continued_trend"):
                sub = persistence_arrays[(int(row["year"]), persistence)]
                flood_full = connected_ocean_flood(dem - sub, water, land, dem_valid, structures[8])
                flood = flood_full & common
                added = flood & ~base
                central_cache[(case, persistence)] = flood_full
                population = population_overlap(flood, reference, pop_path)
                scenario_rows.append(
                    {
                        "scenario": row["scenario"],
                        "year": int(row["year"]),
                        "added_water_m": float(row["surge_m"]),
                        "water_level_egm2008_m": water,
                        "persistence": persistence,
                        "common_domain_area_km2": mask_area_km2(common, row_area),
                        "baseline_connected_area_km2": mask_area_km2(base, row_area),
                        "subsidence_adjusted_area_km2": mask_area_km2(flood, row_area),
                        "subsidence_added_area_km2": mask_area_km2(added, row_area),
                        "baseline_population": baseline_population,
                        "subsidence_adjusted_population": population,
                        "subsidence_added_population": population - baseline_population,
                    }
                )
                if persistence == "continued_trend":
                    for label, mask in (("baseline", base), ("continued", flood), ("added", added)):
                        write_raster(
                            OUT / f"common_{label}_{case}.tif",
                            mask.astype("uint8"),
                            profile,
                            dtype="uint8",
                            nodata=0,
                            tags={
                                "domain": "common analysis land",
                                "connectivity": "8-neighbour connection computed on the full valid DEM domain, then intersected with common land",
                                "water_level_egm2008_m": f"{water:.9f}",
                                "persistence": persistence,
                            },
                        )
            print(f"  {idx + 1}/12 {case}", flush=True)
        persistence_df = pd.DataFrame(scenario_rows)
        persistence_df.to_csv(OUT / "main_scenarios_three_persistence_assumptions.csv", index=False, encoding="utf-8-sig")

        print("Running terrain-offset sensitivity...", flush=True)
        terrain_rows = []
        for offset in TERRAIN_OFFSETS_M:
            for _, row in scenarios.iterrows():
                water = float(row["water_level_used_egm2008_m"])
                case = safe_case(row)
                if offset == 0.0:
                    base_full = central_cache[(case, "baseline")]
                    flood_full = central_cache[(case, "continued_trend")]
                else:
                    base_full = connected_ocean_flood(dem + offset, water, land, dem_valid, structures[8])
                    flood_full = connected_ocean_flood(dem + offset - continued[int(row["year"])], water, land, dem_valid, structures[8])
                base = base_full & common
                flood = flood_full & common
                terrain_rows.append(
                    {
                        "scenario": row["scenario"],
                        "year": int(row["year"]),
                        "added_water_m": float(row["surge_m"]),
                        "water_level_egm2008_m": water,
                        "uniform_dem_offset_m": offset,
                        "baseline_area_km2": mask_area_km2(base, row_area),
                        "continued_trend_area_km2": mask_area_km2(flood, row_area),
                        "subsidence_added_area_km2": mask_area_km2(flood & ~base, row_area),
                    }
                )
        pd.DataFrame(terrain_rows).to_csv(OUT / "terrain_representation_sensitivity.csv", index=False, encoding="utf-8-sig")

        print("Running connectivity sensitivity...", flush=True)
        connectivity_rows = []
        for neighbours in (4, 8):
            for _, row in scenarios.iterrows():
                water = float(row["water_level_used_egm2008_m"])
                case = safe_case(row)
                if neighbours == 8:
                    base_full = central_cache[(case, "baseline")]
                    flood_full = central_cache[(case, "continued_trend")]
                else:
                    base_full = connected_ocean_flood(dem, water, land, dem_valid, structures[4])
                    flood_full = connected_ocean_flood(dem - continued[int(row["year"])], water, land, dem_valid, structures[4])
                base = base_full & common
                flood = flood_full & common
                connectivity_rows.append(
                    {
                        "scenario": row["scenario"],
                        "year": int(row["year"]),
                        "added_water_m": float(row["surge_m"]),
                        "neighbours": neighbours,
                        "baseline_area_km2": mask_area_km2(base, row_area),
                        "continued_trend_area_km2": mask_area_km2(flood, row_area),
                        "subsidence_added_area_km2": mask_area_km2(flood & ~base, row_area),
                    }
                )
        connectivity_df = pd.DataFrame(connectivity_rows)
        connectivity_df.to_csv(OUT / "four_vs_eight_neighbour_sensitivity.csv", index=False, encoding="utf-8-sig")

        print("Building uncertainty-conditioned priority index...", flush=True)
        target = scenarios[
            (scenarios["scenario"] == "SSP5-8.5")
            & (scenarios["year"] == 2100)
            & np.isclose(scenarios["surge_m"], 0.0)
        ].iloc[0]
        target_h = float(target["water_level_used_egm2008_m"])
        rate_scale = float(np.percentile(rate[common], 95))
        s_norm = np.clip(rate / rate_scale, 0.0, 1.0)
        pop_density = population_density_on_reference(reference, extract_population("SSP2", 2020))
        pop_log = np.log1p(pop_density)
        pop_scale = float(np.percentile(pop_log[common], 99))
        p_norm = np.clip(pop_log / pop_scale, 0.0, 1.0)
        priority_cases = []
        central_priority = None
        for dem_offset in (-0.5, 0.0, 0.5):
            for datum_offset in (-DATUM_CI95_M, 0.0, DATUM_CI95_M):
                h = target_h + datum_offset
                threshold = np.clip(1.0 - np.abs((dem + dem_offset) - h) / 0.5, 0.0, 1.0)
                connected = connected_ocean_flood(dem + dem_offset, h + 0.5, land, dem_valid, structures[8])
                priority = (s_norm * threshold * connected.astype("float32") * p_norm * common).astype("float32")
                values = priority[common & (priority > 0)]
                cutoff = float(np.percentile(values, 90)) if values.size else 0.0
                hot = priority >= cutoff if cutoff > 0 else np.zeros_like(common)
                priority_cases.append(hot & common)
                if dem_offset == 0.0 and datum_offset == 0.0:
                    central_priority = priority
                    central_cutoff = cutoff
        if central_priority is None:
            raise AssertionError("Central priority case not created")
        frequency = np.mean(np.stack(priority_cases, axis=0), axis=0).astype("float32")
        priority_write = np.where(common, central_priority, -9999.0).astype("float32")
        frequency_write = np.where(common, frequency, -9999.0).astype("float32")
        write_raster(
            OUT / "priority_index_ssp585_2100_no_added_water.tif",
            priority_write,
            profile,
            dtype="float32",
            nodata=-9999.0,
            tags={
                "formula": "PI = S_n * T_n * C * P_n",
                "S_n": "positive OLS subsidence rate scaled to common-domain 95th percentile",
                "T_n": "linear threshold proximity within 0.5 m of SSP5-8.5 2100 no-added-water boundary",
                "C": "8-neighbour ocean connectivity evaluated at H + 0.5 m",
                "P_n": "log1p SSP2-2020 population density scaled to common-domain 99th percentile",
            },
        )
        write_raster(
            OUT / "priority_top_decile_recurrence.tif",
            frequency_write,
            profile,
            dtype="float32",
            nodata=-9999.0,
            tags={"definition": "fraction of nine DEM/datum sensitivity cases in which a pixel is in the PI top decile"},
        )

        priority_summary = {
            "target": "SSP5-8.5 2100, no added water",
            "target_water_level_egm2008_m": target_h,
            "central_top_decile_cutoff": central_cutoff,
            "central_top_decile_area_km2": mask_area_km2((central_priority >= central_cutoff) & common, row_area),
            "robust_hotspot_area_frequency_ge_2_over_3_km2": mask_area_km2((frequency >= 2 / 3) & common, row_area),
            "moderately_recurrent_area_frequency_ge_1_over_3_km2": mask_area_km2((frequency >= 1 / 3) & common, row_area),
            "maximum_top_decile_recurrence_fraction": float(np.max(frequency[common])),
            "rate_normalisation_p95_m_per_year": rate_scale,
            "population_log_normalisation_p99": pop_scale,
            "sensitivity_cases": 9,
            "interpretation": "Relative priority for improved local data acquisition, not flood probability or expected loss.",
        }
        (OUT / "priority_index_summary.json").write_text(json.dumps(priority_summary, indent=2), encoding="utf-8")

        plot_common_domain(common, land, reference, audit)
        plot_priority(central_priority, frequency, common, reference, central_cutoff)

    summary = {
        "analysis_domain": "Full-domain connectivity followed by intersection with common DEM-valid, GSHHG-land and InSAR-support pixels.",
        "common_domain_audit": audit,
        "persistence_definitions": {
            "stabilisation": "No additional post-2025 terrain lowering.",
            "decaying_trend": "Observed 2025 rate declines linearly to zero in 2100; accumulated lowering is the time integral of that rate, capped as in the continued case.",
            "continued_trend": "Observed positive OLS rate persists, using the existing 1.5 m (2050) and 3.0 m (2100) cumulative caps.",
        },
        "dem_sensitivity": list(TERRAIN_OFFSETS_M),
        "connectivity_sensitivity": [4, 8],
        "reporting_precision": {"area_km2": 1, "population_million": 0.01, "water_level_m": 0.01},
        "figures": {
            "common_domain": "Common-domain spatial support and representativeness.",
            "priority": "Uncertainty-conditioned location ranking for improved data acquisition.",
        },
    }
    (OUT / "reanalysis_method_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Written:", OUT)


if __name__ == "__main__":
    main()
