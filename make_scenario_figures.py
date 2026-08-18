from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from PIL import Image
from scipy import ndimage


ANALYSIS: Path
OUT: Path
DEM_PATH: Path
LAND_PATH: Path
SUB_PATH: Path
EXTENT: tuple[float, float, float, float]
DATUM_CI95_M = 0.116600812
BASE_COLOR = "#E4B852"
ADDED_COLOR = "#C8514B"
COMMON_COLOR = "#E2E5E7"
EXCLUDED_COLOR = "#F4F5F5"
COAST_COLOR = "#697176"
TEXT_COLOR = "#202428"


mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7.2,
        "axes.labelsize": 7.2,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 6.7,
        "ytick.labelsize": 6.7,
        "axes.linewidth": 0.65,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
    }
)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / stem
    common = {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(base.with_suffix(".svg"), **common)
    fig.savefig(base.with_suffix(".pdf"), **common)
    fig.savefig(base.with_suffix(".png"), dpi=500, **common)
    with Image.open(base.with_suffix(".png")) as image:
        image.save(base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)


def safe_case(scenario: str, year: int, added: float) -> str:
    scenario_tag = scenario.replace(".", "p").replace("-", "m")
    return f"{scenario_tag}_{year}_added{added:.1f}m".replace(".", "p")


def window_for_extent(dataset: rasterio.io.DatasetReader):
    from rasterio.windows import from_bounds

    window = from_bounds(EXTENT[0], EXTENT[2], EXTENT[1], EXTENT[3], transform=dataset.transform)
    col0 = max(0, int(math.floor(window.col_off)))
    row0 = max(0, int(math.floor(window.row_off)))
    col1 = min(dataset.width, int(math.ceil(window.col_off + window.width)))
    row1 = min(dataset.height, int(math.ceil(window.row_off + window.height)))
    return rasterio.windows.Window(col0, row0, col1 - col0, row1 - row0)


def read_window(path: Path):
    with rasterio.open(path) as dataset:
        window = window_for_extent(dataset)
        array = dataset.read(1, window=window)
        left, bottom, right, top = rasterio.windows.bounds(window, dataset.transform)
    return array, (left, right, bottom, top)


def edge(mask: np.ndarray) -> np.ndarray:
    structure = ndimage.generate_binary_structure(2, 2)
    return mask & ~ndimage.binary_erosion(mask, structure=structure)


def configure_map(ax: plt.Axes, show_x: bool, show_y: bool) -> None:
    ax.set_xlim(EXTENT[0], EXTENT[1])
    ax.set_ylim(EXTENT[2], EXTENT[3])
    ax.set_aspect(1.0 / np.cos(np.deg2rad(np.mean(EXTENT[2:]))))
    ax.set_xticks([112.5, 113.0, 113.5])
    ax.set_yticks([22.0, 22.5])
    ax.set_xticklabels([r"112.5$^\circ$E", r"113.0$^\circ$E", r"113.5$^\circ$E"])
    ax.set_yticklabels([r"22.0$^\circ$N", r"22.5$^\circ$N"])
    ax.tick_params(length=1.8, width=0.55, pad=1.1, labelbottom=show_x, labelleft=show_y)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#687074")
        spine.set_linewidth(0.55)


def make_year_map(frame: pd.DataFrame, year: int) -> None:
    land, land_extent = read_window(LAND_PATH)
    common, common_extent = read_window(ANALYSIS / "common_analysis_land_mask.tif")
    land = land > 0
    common = common > 0
    coast = edge(land)

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.05), sharex=True, sharey=True)
    letters = iter("abcdef")
    for row_index, scenario in enumerate(["SSP2-4.5", "SSP5-8.5"]):
        for col_index, added in enumerate([0.0, 1.5, 3.0]):
            ax = axes[row_index, col_index]
            record = frame[
                frame["scenario"].eq(scenario)
                & frame["year"].eq(year)
                & np.isclose(frame["added_water_m"], added)
                & frame["persistence"].eq("continued_trend")
            ].iloc[0]
            case = safe_case(scenario, year, added)
            baseline, mask_extent = read_window(ANALYSIS / f"common_baseline_{case}.tif")
            added_mask, _ = read_window(ANALYSIS / f"common_added_{case}.tif")
            ax.set_facecolor("white")
            ax.imshow(
                np.ma.masked_where(~land, land),
                extent=land_extent,
                origin="upper",
                cmap=ListedColormap([EXCLUDED_COLOR]),
                interpolation="nearest",
                zorder=0,
            )
            ax.imshow(
                np.ma.masked_where(~common, common),
                extent=common_extent,
                origin="upper",
                cmap=ListedColormap([COMMON_COLOR]),
                interpolation="nearest",
                zorder=1,
            )
            ax.imshow(
                np.ma.masked_where(baseline <= 0, baseline),
                extent=mask_extent,
                origin="upper",
                cmap=ListedColormap([BASE_COLOR]),
                interpolation="nearest",
                alpha=0.95,
                zorder=2,
            )
            ax.imshow(
                np.ma.masked_where(added_mask <= 0, added_mask),
                extent=mask_extent,
                origin="upper",
                cmap=ListedColormap([ADDED_COLOR]),
                interpolation="nearest",
                alpha=0.98,
                zorder=3,
            )
            ax.imshow(
                np.ma.masked_where(~coast, coast),
                extent=land_extent,
                origin="upper",
                cmap=ListedColormap([COAST_COLOR]),
                interpolation="nearest",
                alpha=0.62,
                zorder=4,
            )
            configure_map(ax, show_x=row_index == 1, show_y=col_index == 0)
            letter = next(letters)
            ax.text(
                0.018,
                0.975,
                f"{letter}  {scenario}",
                transform=ax.transAxes,
                va="top",
                ha="left",
                fontsize=7.7,
                fontweight="bold",
                color=TEXT_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
                zorder=8,
            )
            base_area = float(record["baseline_connected_area_km2"])
            delta_area = float(record["subsidence_added_area_km2"])
            water = float(record["water_level_egm2008_m"])
            ax.text(
                0.018,
                0.035,
                rf"H = {water:.2f} m | {base_area:.0f} + {delta_area:.0f} km$^2$",
                transform=ax.transAxes,
                va="bottom",
                ha="left",
                fontsize=6.1,
                color=TEXT_COLOR,
                bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.88, "pad": 1.2},
                zorder=8,
            )
            if row_index == 0:
                ax.set_title(f"Added water: {added:.1f} m", pad=3)
    fig.legend(
        handles=[
            Patch(facecolor=BASE_COLOR, label="Baseline connected area"),
            Patch(facecolor=ADDED_COLOR, label="Added by continued-trend subsidence"),
            Patch(facecolor=COMMON_COLOR, label="Common-domain land"),
            Patch(facecolor=EXCLUDED_COLOR, edgecolor="#B8BDC1", label="Land outside common domain"),
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.008),
        ncol=4,
        fontsize=6.3,
    )
    fig.subplots_adjust(left=0.07, right=0.995, top=0.93, bottom=0.115, wspace=0.035, hspace=0.055)
    save_figure(fig, f"Figure_Common_domain_connected_inundation_{year}")


def make_persistence_sensitivity(frame: pd.DataFrame) -> None:
    target = frame[
        frame["scenario"].eq("SSP5-8.5")
        & frame["year"].eq(2100)
        & np.isclose(frame["added_water_m"], 0.0)
    ].copy()
    order = ["stabilisation", "decaying_trend", "continued_trend"]
    target["persistence"] = pd.Categorical(target["persistence"], order, ordered=True)
    target = target.sort_values("persistence")
    terrain = pd.read_csv(ANALYSIS / "terrain_representation_sensitivity.csv")
    terrain = terrain[
        terrain["scenario"].eq("SSP5-8.5")
        & terrain["year"].eq(2100)
        & np.isclose(terrain["added_water_m"], 0.0)
    ].sort_values("uniform_dem_offset_m")
    connectivity = pd.read_csv(ANALYSIS / "four_vs_eight_neighbour_sensitivity.csv")
    connectivity = connectivity[
        connectivity["scenario"].eq("SSP5-8.5")
        & connectivity["year"].eq(2100)
        & np.isclose(connectivity["added_water_m"], 0.0)
    ].sort_values("neighbours")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.55), gridspec_kw={"wspace": 0.45})
    ax = axes[0]
    labels = ["Stabilisation", "Decaying\ntrend", "Continued\ntrend"]
    values = target["subsidence_adjusted_area_km2"].to_numpy(float)
    bars = ax.bar(labels, values, color=["#7B8790", "#4C9B8A", "#C8514B"], width=0.68)
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in values], padding=2, fontsize=6.3)
    ax.set_ylabel(r"Connected area (km$^2$)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.set_title("Persistence assumption", loc="left", fontweight="bold")
    ax.text(-0.13, 1.02, "a", transform=ax.transAxes, fontweight="bold")
    ax.grid(axis="y", color="#E2E5E7", lw=0.5)

    ax = axes[1]
    ax.plot(terrain["uniform_dem_offset_m"], terrain["baseline_area_km2"], "o-", color="#5B7895", label="Baseline")
    ax.plot(terrain["uniform_dem_offset_m"], terrain["continued_trend_area_km2"], "o-", color="#C8514B", label="Continued trend")
    ax.set_xlabel("Uniform DEM offset (m)")
    ax.set_ylabel(r"Connected area (km$^2$)")
    ax.set_title("Terrain representation", loc="left", fontweight="bold")
    ax.text(-0.13, 1.02, "b", transform=ax.transAxes, fontweight="bold")
    ax.legend(fontsize=6.2, loc="best")
    ax.grid(color="#E2E5E7", lw=0.5)

    ax = axes[2]
    x = np.arange(len(connectivity))
    width = 0.34
    ax.bar(x - width / 2, connectivity["baseline_area_km2"], width, color="#5B7895", label="Baseline")
    ax.bar(x + width / 2, connectivity["continued_trend_area_km2"], width, color="#C8514B", label="Continued trend")
    ax.set_xticks(x, [f"{int(v)}-neighbour" for v in connectivity["neighbours"]])
    ax.set_ylabel(r"Connected area (km$^2$)")
    ax.set_title("Connectivity rule", loc="left", fontweight="bold")
    ax.text(-0.13, 1.02, "c", transform=ax.transAxes, fontweight="bold")
    ax.legend(fontsize=6.2, loc="best")
    ax.grid(axis="y", color="#E2E5E7", lw=0.5)
    fig.suptitle("SSP5-8.5, 2100, no added water", y=1.015, fontsize=8.2)
    save_figure(fig, "Figure_Persistence_terrain_connectivity_sensitivity")


def ellipsoid_strip_primitive(latitude_rad: np.ndarray) -> np.ndarray:
    a = 6378137.0
    f = 1.0 / 298.257223563
    e2 = f * (2.0 - f)
    u = np.sin(latitude_rad)
    e = math.sqrt(e2)
    primitive = (1.0 - e2) * (u / (2.0 * (1.0 - e2 * u * u)) + np.arctanh(e * u) / (2.0 * e))
    return a * a * primitive


def row_areas_km2(transform, height: int) -> np.ndarray:
    edges = transform.f + transform.e * np.arange(height + 1, dtype=float)
    strips = np.abs(np.diff(ellipsoid_strip_primitive(np.deg2rad(edges))))
    return strips * abs(math.radians(transform.a)) / 1e6


def mask_area(mask: np.ndarray, row_area: np.ndarray) -> float:
    return float(np.dot(np.count_nonzero(mask, axis=1), row_area))


def valid_mask(dataset, values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values)
    if dataset.nodata is not None:
        valid &= values != dataset.nodata
    return valid


def connected(surface: np.ndarray, water: float, land: np.ndarray, valid: np.ndarray) -> np.ndarray:
    low_land = land & valid & (surface <= water)
    ocean = (~land) & valid
    domain = ocean | low_land
    seeds = np.zeros_like(domain, dtype=bool)
    seeds[0, :] = ocean[0, :]
    seeds[-1, :] = ocean[-1, :]
    seeds[:, 0] = ocean[:, 0]
    seeds[:, -1] = ocean[:, -1]
    return ndimage.binary_propagation(seeds, structure=ndimage.generate_binary_structure(2, 2), mask=domain) & low_land


def calculate_datum_sensitivity(frame: pd.DataFrame) -> pd.DataFrame:
    cached = ANALYSIS / "vertical_datum_sensitivity_target_scenario.csv"
    if cached.exists():
        return pd.read_csv(cached)
    target = frame[
        frame["scenario"].eq("SSP5-8.5")
        & frame["year"].eq(2100)
        & np.isclose(frame["added_water_m"], 0.0)
        & frame["persistence"].eq("continued_trend")
    ].iloc[0]
    central_h = float(target["water_level_egm2008_m"])
    with rasterio.open(DEM_PATH) as reference, rasterio.open(LAND_PATH) as land_ds, rasterio.open(SUB_PATH) as sub_ds, rasterio.open(ANALYSIS / "common_analysis_land_mask.tif") as common_ds:
        dem = reference.read(1).astype("float32")
        valid = valid_mask(reference, dem)
        land = (land_ds.read(1) > 0) & valid
        subsidence = sub_ds.read(1).astype("float32")
        sub_valid = valid_mask(sub_ds, subsidence)
        subsidence = np.where(sub_valid, np.maximum(subsidence, 0.0), 0.0)
        common = common_ds.read(1) > 0
        row_area = row_areas_km2(reference.transform, reference.height)
        rows = []
        for offset in [-DATUM_CI95_M, 0.0, DATUM_CI95_M]:
            water = central_h + offset
            baseline = connected(dem, water, land, valid) & common
            adjusted = connected(dem - subsidence, water, land, valid) & common
            rows.append(
                {
                    "datum_offset_m": offset,
                    "water_level_egm2008_m": water,
                    "baseline_area_km2": mask_area(baseline, row_area),
                    "continued_trend_area_km2": mask_area(adjusted, row_area),
                    "subsidence_added_area_km2": mask_area(adjusted & ~baseline, row_area),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(cached, index=False, encoding="utf-8-sig")
    return result


def make_vertical_reference_figure(frame: pd.DataFrame) -> None:
    datum = calculate_datum_sensitivity(frame)
    terrain = pd.read_csv(ANALYSIS / "terrain_representation_sensitivity.csv")
    terrain = terrain[
        terrain["scenario"].eq("SSP5-8.5")
        & terrain["year"].eq(2100)
        & np.isclose(terrain["added_water_m"], 0.0)
    ].sort_values("uniform_dem_offset_m")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.75), gridspec_kw={"width_ratios": [0.9, 1.15, 1.15], "wspace": 0.48})
    ax = axes[0]
    ax.axis("off")
    boxes = [
        (0.83, "PSMSL mean sea level\n1.460 m above CD", "#E7BCB8"),
        (0.51, "CD to EGM2008\n-0.534 m", "#A9C8DD"),
        (0.19, "AR6-reference $H_0$\n0.927 $\\pm$ 0.117 m", "#9B90C0"),
    ]
    for y, label, color in boxes:
        ax.text(0.5, y, label, ha="center", va="center", transform=ax.transAxes, fontsize=7.0, bbox={"boxstyle": "round,pad=0.45", "facecolor": color, "edgecolor": "none"})
    for y0, y1 in [(0.72, 0.62), (0.40, 0.30)]:
        ax.annotate("", xy=(0.5, y1), xytext=(0.5, y0), xycoords="axes fraction", arrowprops={"arrowstyle": "-|>", "lw": 1.1, "color": "#4F5559"})
    ax.text(0.02, 0.98, "a", transform=ax.transAxes, va="top", fontweight="bold")
    ax.set_title("Traceable datum chain", loc="left", fontweight="bold")

    ax = axes[1]
    ax.plot(datum["datum_offset_m"], datum["baseline_area_km2"], "o-", color="#5B7895", label="Baseline")
    ax.plot(datum["datum_offset_m"], datum["continued_trend_area_km2"], "o-", color="#C8514B", label="Continued trend")
    ax.axvline(0, color="#8D9499", lw=0.7, ls="--")
    ax.set_xlabel("Water-boundary offset (m)")
    ax.set_ylabel(r"Connected area (km$^2$)")
    ax.set_title("Datum sensitivity", loc="left", fontweight="bold")
    ax.text(-0.15, 1.02, "b", transform=ax.transAxes, fontweight="bold")
    ax.legend(fontsize=6.2)
    ax.grid(color="#E2E5E7", lw=0.5)

    ax = axes[2]
    ax.plot(terrain["uniform_dem_offset_m"], terrain["baseline_area_km2"], "o-", color="#5B7895", label="Baseline")
    ax.plot(terrain["uniform_dem_offset_m"], terrain["continued_trend_area_km2"], "o-", color="#C8514B", label="Continued trend")
    ax.axvline(0, color="#8D9499", lw=0.7, ls="--")
    ax.set_xlabel("Uniform DEM offset (m)")
    ax.set_ylabel(r"Connected area (km$^2$)")
    ax.set_title("Terrain sensitivity", loc="left", fontweight="bold")
    ax.text(-0.15, 1.02, "c", transform=ax.transAxes, fontweight="bold")
    ax.legend(fontsize=6.2)
    ax.grid(color="#E2E5E7", lw=0.5)
    fig.suptitle("SSP5-8.5, 2100, no added water", y=1.005, fontsize=8.2)
    save_figure(fig, "Figure_Vertical_reference_and_terrain_sensitivity")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate scenario and sensitivity figures from audited outputs.")
    parser.add_argument("--analysis", type=Path, required=True, help="Directory produced by build_common_domain_reanalysis.py.")
    parser.add_argument("--dem", type=Path, required=True, help="Aligned Copernicus DEM raster.")
    parser.add_argument("--land-mask", type=Path, required=True, help="Aligned GSHHG land-mask raster.")
    parser.add_argument("--subsidence-2100", type=Path, required=True, help="OLS 2100 cumulative-subsidence raster.")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="Output directory.")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("MIN_LON", "MAX_LON", "MIN_LAT", "MAX_LAT"),
        default=(112.30, 113.82, 21.70, 22.62),
        help="Map extent in geographic coordinates.",
    )
    return parser.parse_args()


def main() -> None:
    global ANALYSIS, OUT, DEM_PATH, LAND_PATH, SUB_PATH, EXTENT
    args = parse_args()
    ANALYSIS = args.analysis
    OUT = args.out
    DEM_PATH = args.dem
    LAND_PATH = args.land_mask
    SUB_PATH = args.subsidence_2100
    EXTENT = tuple(args.extent)

    for path in [ANALYSIS / "main_scenarios_three_persistence_assumptions.csv", DEM_PATH, LAND_PATH, SUB_PATH]:
        if not path.exists():
            raise FileNotFoundError(path)
    frame = pd.read_csv(ANALYSIS / "main_scenarios_three_persistence_assumptions.csv")
    make_year_map(frame, 2050)
    make_year_map(frame, 2100)
    make_persistence_sensitivity(frame)
    make_vertical_reference_figure(frame)
    manifest = {
        "reporting_precision": {"area_km2": 1, "population_million": 0.01, "water_level_m": 0.01},
        "map_encoding": "baseline connected area plus the increment added by continued-trend subsidence, restricted to the common analysis domain",
        "figures": sorted(str(path.name) for path in OUT.glob("Figure_*.png")),
    }
    (OUT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("Written:", OUT)


if __name__ == "__main__":
    main()
