"""Create a publication-quality line plot for confidence-only selective gains.

This figure isolates the two hardest evidence conditions and plots the
held-out selective-accuracy gain of the confidence-only baseline relative to
the 100% coverage baseline. The plotted x-axis omits the trivial 100%
coverage point and only shows the selective operating points. The script
expects matplotlib and seaborn to be installed and writes a vector PDF
suitable for inclusion in the paper.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import seaborn as sns
except ModuleNotFoundError as exc:
    raise SystemExit(
        "matplotlib and seaborn are required for this figure. "
        "Install the project requirements first, e.g. `pip install -r requirements.txt`."
    ) from exc

from plot_selective_prediction_summary import MODELS

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / ".dr-claw" / "chat-attachments" / "1775532891437"
DEFAULT_OUT_PATH = OUT_DIR / "confidence_only_selective_gain_curves.pdf"

CONDITIONS = [
    ("IC", "IC"),
    ("ICC", "ICC"),
]

MODE_TO_KEY = {
    "threshold_transfer": "selective_prediction",
    "exact_topk": "selective_prediction_exact_topk",
}

EXCLUDED_COVERAGES = {"1.0"}
PALETTE = [
    "#d73027",
    "#fc8d59",
    "#fee090",
    "#ffffbf",
    "#e0f3f8",
    "#91bfdb",
    "#4575b4",
]
MODEL_SERIES = [
    (MODELS[idx][0], MODELS[idx][1], color)
    for idx, color in enumerate(PALETTE[:-1])
]
MEAN_COLOR = PALETTE[-1]
ZERO_LINE_COLOR = "#6B7280"
GRID_COLOR = "#D8DDE6"
MARKER_STYLE = "o"
MEAN_MARKER_STYLE = "D"
MARKER_EDGE_COLOR = "#2D3142"
SPREAD_COLOR = "#C7D2FE"
TITLE_BBOX = {
    "facecolor": "#F3F4F6",
    "edgecolor": "none",
    "boxstyle": "round,pad=0.25",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a publication-quality line plot for confidence-only selective gains."
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_KEY),
        default="threshold_transfer",
        help="Which selective-prediction protocol to visualize.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(DEFAULT_OUT_PATH),
        help="Output PDF path.",
    )
    return parser.parse_args()


def load_gains(selective_key: str) -> tuple[dict[str, dict[str, list[float]]], list[int]]:
    gains: dict[str, dict[str, list[float]]] = {}
    coverage_keys: list[str] | None = None
    for model_key, _, _ in MODEL_SERIES:
        data = json.loads(
            (RESULTS_DIR / f"{model_key}_conflict_prediction.json").read_text(encoding="utf-8")
        )
        gains[model_key] = {}
        for cond, _ in CONDITIONS:
            cond_data = data["conditions"][cond]
            sp = cond_data["within_condition"][selective_key]
            if coverage_keys is None:
                coverage_keys = [
                    key for key in sorted(sp.keys(), key=float, reverse=True)
                    if key not in EXCLUDED_COVERAGES
                ]
            baseline = float(cond_data["accuracy"])
            gains[model_key][cond] = [
                (float(sp[c]["test_accuracy_conf_only"]) - baseline) * 100.0
                for c in coverage_keys
            ]
    coverage_levels = [int(round(float(key) * 100)) for key in (coverage_keys or [])]
    return gains, coverage_levels


def compute_bounds(gains: dict[str, dict[str, list[float]]]) -> tuple[float, float]:
    return -10.0, 20.0


def configure_style() -> None:
    sns.set_theme(
        style="ticks",
        context="paper",
        font_scale=1.15,
        rc={
            "axes.spines.right": False,
            "axes.spines.top": False,
            "axes.edgecolor": "#30343F",
            "axes.facecolor": "#FAFAF6",
            "axes.linewidth": 0.9,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        },
    )


def plot_panel(
    ax: plt.Axes,
    condition: str,
    title: str,
    coverage_levels: list[int],
    gains: dict[str, dict[str, list[float]]],
) -> list[float]:
    series: list[list[float]] = []
    for model_key, label, color in MODEL_SERIES:
        y = gains[model_key][condition]
        series.append(y)
        ax.plot(
            coverage_levels,
            y,
            color=color,
            linewidth=1.55,
            marker=MARKER_STYLE,
            markersize=4.6,
            markerfacecolor=color,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=0.7,
            alpha=0.72,
            label=label,
            zorder=2,
        )

    mean_values = [
        sum(values_at_point) / len(values_at_point)
        for values_at_point in zip(*series)
    ]
    lower_values = [
        statistics.quantiles(values_at_point, n=4, method="inclusive")[0]
        for values_at_point in zip(*series)
    ]
    upper_values = [
        statistics.quantiles(values_at_point, n=4, method="inclusive")[2]
        for values_at_point in zip(*series)
    ]
    ax.fill_between(
        coverage_levels,
        lower_values,
        upper_values,
        color=SPREAD_COLOR,
        alpha=0.35,
        zorder=1,
        linewidth=0,
    )
    ax.plot(
        coverage_levels,
        mean_values,
        color=MEAN_COLOR,
        linewidth=3.0,
        linestyle="-",
        marker=MEAN_MARKER_STYLE,
        markersize=6.0,
        markerfacecolor=MEAN_COLOR,
        markeredgecolor=MARKER_EDGE_COLOR,
        markeredgewidth=0.9,
        label="Mean",
        zorder=4,
    )

    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.2, linestyle=":")
    ax.set_title(title, fontsize=10.2, pad=6, weight="semibold", bbox=TITLE_BBOX)
    ax.set_xticks(coverage_levels)
    ax.set_xlim(max(coverage_levels), min(coverage_levels))
    ax.grid(axis="y")
    ax.tick_params(axis="both", labelsize=9.5)
    ax.annotate(
        f"{mean_values[-1]:+.1f}",
        xy=(coverage_levels[-1], mean_values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8.8,
        color=MEAN_COLOR,
        va="center",
        weight="semibold",
    )
    sns.despine(ax=ax)
    return mean_values


def build_legend_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=2.0,
            marker=MARKER_STYLE,
            markersize=5.2,
            markerfacecolor=color,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=0.8,
            label=label,
        )
        for _, label, color in MODEL_SERIES
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            color=MEAN_COLOR,
            linewidth=2.8,
            marker=MEAN_MARKER_STYLE,
            markersize=5.6,
            markerfacecolor=MEAN_COLOR,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=0.9,
            label="Mean",
        )
    )
    handles.append(
        Line2D(
            [0],
            [0],
            color=SPREAD_COLOR,
            linewidth=7,
            alpha=0.6,
            label="Model IQR",
        )
    )
    return handles


def render_with_matplotlib(
    gains: dict[str, dict[str, list[float]]],
    coverage_levels: list[int],
    out_path: Path,
) -> None:
    configure_style()
    y_min, y_max = compute_bounds(gains)

    fig, axes = plt.subplots(
        1,
        len(CONDITIONS),
        figsize=(7.2, 2.7),
        sharey=True,
    )
    if len(CONDITIONS) == 1:
        axes = [axes]

    for ax, (condition, title) in zip(axes, CONDITIONS):
        plot_panel(ax, condition, title, coverage_levels, gains)
        ax.set_xlabel("Coverage (%)", fontsize=9.0)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([-10, 0, 10, 20])

    axes[0].set_ylabel("Gain vs. 100% coverage (pp)", fontsize=9.3)

    legend_handles = build_legend_handles()
    fig.legend(
        handles=legend_handles,
        loc="center left",
        ncol=1,
        bbox_to_anchor=(0.87, 0.5),
        columnspacing=0.8,
        handlelength=1.8,
        handletextpad=0.4,
        fontsize=7.4,
    )

    fig.subplots_adjust(bottom=0.18, right=0.82, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    gains, coverage_levels = load_gains(MODE_TO_KEY[args.mode])
    render_with_matplotlib(gains, coverage_levels, Path(args.out_path))
    print(Path(args.out_path))


if __name__ == "__main__":
    main()
