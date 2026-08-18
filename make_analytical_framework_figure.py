import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image


OUT: Path

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "font.size": 7,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "figure.facecolor": "white",
    }
)


def rounded_box(ax, x, y, w, h, title, body, edge, face, stack=False):
    if stack:
        for dx, alpha in [(0.013, 0.32), (0.0065, 0.58)]:
            ax.add_patch(
                FancyBboxPatch(
                    (x + dx, y - dx), w, h,
                    boxstyle="round,pad=0.006,rounding_size=0.010",
                    linewidth=0.7, edgecolor=edge, facecolor=face, alpha=alpha, zorder=1,
                )
            )
    ax.add_patch(
        FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.006,rounding_size=0.010",
            linewidth=1.05, edgecolor=edge, facecolor=face, zorder=2,
        )
    )
    ax.text(
        x + w / 2, y + h * 0.66, title,
        ha="center", va="center", weight="bold", fontsize=6.25,
        color="#263238", linespacing=1.06, zorder=3,
    )
    ax.text(
        x + w / 2, y + h * 0.27, body,
        ha="center", va="center", fontsize=5.15,
        color="#536B79", linespacing=1.08, zorder=3,
    )


def horizontal_arrow(ax, x0, x1, y, color="#667D89"):
    ax.add_patch(
        FancyArrowPatch(
            (x0, y), (x1, y), arrowstyle="-|>", mutation_scale=8,
            linewidth=1.1, color=color, connectionstyle="arc3,rad=0", zorder=4,
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate the analytical-framework figure.")
    parser.add_argument("--out", type=Path, default=Path("figures"), help="Output directory.")
    return parser.parse_args()


def main():
    global OUT
    OUT = parse_args().out

    fig, ax = plt.subplots(figsize=(7.2, 4.85))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    blue, gold, purple, green = "#5B8DB8", "#D69A32", "#8270B2", "#4F9A82"
    columns = [
        (0.025, 0.285, "1  SOURCE DATA", blue, "#F3F8FB"),
        (0.357, 0.285, "2  TRACEABLE\nTRANSFORMATION", gold, "#FFFAF0"),
        (0.689, 0.285, "3  ANALYSIS\nOUTPUT", purple, "#F8F5FC"),
    ]
    for x, w, heading, edge, face in columns:
        ax.add_patch(
            FancyBboxPatch(
                (x, 0.145), w, 0.82,
                boxstyle="round,pad=0.006,rounding_size=0.014",
                linewidth=0.9, edgecolor=edge, facecolor=face, alpha=0.95,
            )
        )
        ax.text(
            x + w / 2, 0.925, heading,
            ha="center", va="center", weight="bold", fontsize=6.9,
            color=edge, linespacing=1.02,
        )

    rows = [0.76, 0.59, 0.42, 0.25]
    height = 0.125
    left = [
        ("Deformation\nobservations", "249 Sentinel-1 scenes\n2017-2025", True),
        ("Reference terrain\nand coast", "Copernicus DEM GLO-30\nGSHHG land-ocean mask", True),
        ("Water-level\ncomponents", "PSMSL/HKO datum chain\nAR6 no-VLM + 0/1.5/3 m", True),
        ("Exposure and context", "SSP population grids\nadministrative reference layers", True),
    ]
    middle = [
        ("SBAS-InSAR and\ntemporal assessment", "recent-period holdout; three\npersistence assumptions"),
        ("Spatial and vertical\nharmonisation", "common raster geometry + EGM2008;\ncommon-domain paired support"),
        ("Connected-inundation\nscreening", "full-domain 8-neighbour propagation;\ncommon-domain intersection"),
        ("Weighting and priority\ntransformation", "ellipsoidal area; population weighting;\nPI = S_n x T_n x C x P_n"),
    ]
    right = [
        ("Conditional\nfuture terrain", "three persistence assumptions;\nnot forecasts", True),
        ("Paired spatial masks", "baseline, subsidence-adjusted\nand subsidence-added", True),
        ("Area and exposure", "scenario tables and spatial\nresponse maps", True),
        ("Priority and\nrecurrence", "uncertainty-conditioned hotspots;\ntargeted survey and model escalation", True),
    ]

    for y, lspec, mspec, rspec in zip(rows, left, middle, right):
        rounded_box(ax, 0.052, y, 0.225, height, lspec[0], lspec[1], blue, "#FFFFFF", stack=lspec[2])
        rounded_box(ax, 0.384, y, 0.225, height, mspec[0], mspec[1], gold if y > 0.42 else green, "#FFF4D6" if y > 0.42 else "#E4F1EC")
        rounded_box(ax, 0.716, y, 0.225, height, rspec[0], rspec[1], purple, "#FFFFFF" if y > 0.59 else "#F0EBF8", stack=rspec[2])
        horizontal_arrow(ax, 0.278, 0.383, y + height / 2)
        horizontal_arrow(ax, 0.610, 0.715, y + height / 2)

    ax.add_patch(
        FancyBboxPatch(
            (0.025, 0.025), 0.95, 0.085,
            boxstyle="round,pad=0.006,rounding_size=0.012",
            linewidth=1.0, edgecolor="#607D8B", facecolor="#EEF3F5",
        )
    )
    ax.text(0.047, 0.077, "PROVENANCE", color="#416F94", weight="bold", fontsize=6.4, va="center")
    ax.text(0.165, 0.077, "source IDs -> transformations -> intermediate products -> final masks", color="#455A64", fontsize=5.45, va="center")
    ax.text(0.047, 0.048, "SENSITIVITY", color="#4F9A82", weight="bold", fontsize=6.4, va="center")
    ax.text(0.165, 0.048, "datum interval | DEM +/-0.5 and +/-1 m | 4/8-neighbour connectivity", color="#455A64", fontsize=5.45, va="center")

    OUT.mkdir(parents=True, exist_ok=True)
    base = OUT / "Figure_Analytical_framework"
    common = {"bbox_inches": "tight", "pad_inches": 0.03}
    fig.savefig(base.with_suffix(".svg"), **common)
    fig.savefig(base.with_suffix(".pdf"), **common)
    fig.savefig(base.with_suffix(".png"), dpi=500, **common)
    with Image.open(base.with_suffix(".png")) as image:
        image.save(base.with_suffix(".tiff"), dpi=(600, 600), compression="tiff_lzw")
    plt.close(fig)
    print("Written:", base)


if __name__ == "__main__":
    main()
