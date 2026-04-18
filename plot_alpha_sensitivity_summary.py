"""Plot a publication-quality alpha-sensitivity summary for HealthContradict.

This appendix figure mirrors the confidence-only curve layout: two panels
(`IC` and `ICC`), target coverage on the x-axis, and one line per mixing
weight ``alpha``. The script reads the JSON summary written by
``sweep_healthcontradict_alpha.py``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = ROOT / "results" / "alpha_sensitivity" / "hc_alpha_sweep_summary.json"
OUT_DIR = ROOT / ".dr-claw" / "chat-attachments" / "1775532891437"
DEFAULT_OUT_PATH = OUT_DIR / "alpha_sensitivity_selective_gain.pdf"

CONDITIONS = [
    ("IC", "IC"),
    ("ICC", "ICC"),
]
ALPHA_SERIES = [
    (0.25, r"$\alpha=0.25$", "#fef0d9"),
    (0.5, r"$\alpha=0.5$", "#fdcc8a"),
    (0.75, r"$\alpha=0.75$", "#fc8d59"),
    (1.0, r"$\alpha=1.0$", "#d7301f"),
]
GRID_COLOR = "#D8DDE6"
ZERO_LINE_COLOR = "#6B7280"
MARKER_STYLE = "o"
MARKER_EDGE_COLOR = "#2D3142"
REFERENCE_ALPHA = 0.5
REFERENCE_LINE_COLOR = "#B45309"
TITLE_BBOX = {
    "facecolor": "#F3F4F6",
    "edgecolor": "none",
    "boxstyle": "round,pad=0.25",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HealthContradict alpha sensitivity from a saved summary."
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(SUMMARY_PATH),
        help="Path to hc_alpha_sweep_summary.json.",
    )
    parser.add_argument(
        "--out-path",
        type=str,
        default=str(DEFAULT_OUT_PATH),
        help="Output PDF path.",
    )
    parser.add_argument(
        "--metric",
        choices=["mean_lift_over_confidence_pp", "mean_combined_gain_vs_baseline_pp"],
        default="mean_lift_over_confidence_pp",
        help="Which summary metric to plot.",
    )
    return parser.parse_args()


def _load_summary(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _metric_label(metric: str) -> str:
    if metric == "mean_lift_over_confidence_pp":
        return "Mean lift over confidence-only (pp)"
    return "Mean gain over 100% coverage (pp)"


def _collect_series(
    summary: dict,
    metric: str,
) -> tuple[dict[str, dict[float, list[float]]], list[int]]:
    mean_rows = summary["mean_rows"]
    coverage_targets = sorted(
        {float(value) for value in summary["coverage_targets"]},
        reverse=True,
    )
    coverage_levels = [int(round(value * 100)) for value in coverage_targets]
    series: dict[str, dict[float, list[float]]] = {}

    for condition, _title in CONDITIONS:
        series[condition] = {}
        for alpha, _label, _color in ALPHA_SERIES:
            values = []
            for coverage_target in coverage_targets:
                match = next(
                    row for row in mean_rows
                    if abs(float(row["alpha"]) - alpha) < 1e-12
                    and row["condition"] == condition
                    and abs(float(row["coverage_target"]) - coverage_target) < 1e-12
                )
                values.append(float(match[metric]))
            series[condition][alpha] = values

    return series, coverage_levels


def _compute_y_bounds(series: dict[str, dict[float, list[float]]]) -> tuple[float, float]:
    return 0.0, 40.0


def _configure_style() -> None:
    import seaborn as sns

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


def _plot_panel(
    ax,
    condition: str,
    title: str,
    coverage_levels: list[int],
    series: dict[str, dict[float, list[float]]],
) -> None:
    for alpha, label, color in ALPHA_SERIES:
        if alpha not in series[condition]:
            continue
        is_reference = abs(alpha - REFERENCE_ALPHA) < 1e-12
        ax.plot(
            coverage_levels,
            series[condition][alpha],
            color=color,
            linewidth=3.0 if is_reference else 2.0,
            linestyle="-" if is_reference else "--",
            marker=MARKER_STYLE,
            markersize=5.8 if is_reference else 4.8,
            markerfacecolor=color,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=0.8,
            alpha=0.98 if is_reference else 0.82,
            label=label,
            zorder=4 if is_reference else 2,
        )

    ax.axhline(0.0, color=ZERO_LINE_COLOR, linewidth=1.2, linestyle=":")
    ax.set_title(title, fontsize=10.2, pad=6, weight="semibold", bbox=TITLE_BBOX)
    ax.set_xticks(coverage_levels)
    ax.set_xlim(max(coverage_levels), min(coverage_levels))
    ax.grid(axis="y")
    ax.tick_params(axis="both", labelsize=9.5)
    ref_values = series[condition][REFERENCE_ALPHA]
    ax.annotate(
        f"{ref_values[-1]:+.1f}",
        xy=(coverage_levels[-1], ref_values[-1]),
        xytext=(6, 0),
        textcoords="offset points",
        fontsize=8.8,
        color=REFERENCE_LINE_COLOR,
        va="center",
        weight="semibold",
    )


def _build_legend_handles():
    from matplotlib.lines import Line2D

    return [
        Line2D(
            [0],
            [0],
            color=color,
            linewidth=3.0 if abs(_alpha - REFERENCE_ALPHA) < 1e-12 else 2.0,
            linestyle="-" if abs(_alpha - REFERENCE_ALPHA) < 1e-12 else "--",
            marker=MARKER_STYLE,
            markersize=5.8 if abs(_alpha - REFERENCE_ALPHA) < 1e-12 else 4.8,
            markerfacecolor=color,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=0.8,
            label=label,
        )
        for _alpha, label, color in ALPHA_SERIES
    ]


def main() -> None:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "matplotlib and seaborn are required for this figure. "
            "Install the project requirements first, e.g. `pip install -r requirements.txt`."
        ) from exc

    summary_path = Path(args.summary_path)
    out_path = Path(args.out_path)
    summary = _load_summary(summary_path)
    series, coverage_levels = _collect_series(summary, args.metric)
    y_min, y_max = _compute_y_bounds(series)

    _configure_style()
    fig, axes = plt.subplots(
        1,
        len(CONDITIONS),
        figsize=(7.2, 2.7),
        sharey=True,
    )
    if len(CONDITIONS) == 1:
        axes = [axes]

    for ax, (condition, title) in zip(axes, CONDITIONS):
        _plot_panel(ax, condition, title, coverage_levels, series)
        ax.set_xlabel("Coverage (%)", fontsize=9.0)
        ax.set_ylim(y_min, y_max)
        ax.set_yticks([0, 10, 20, 30, 40])
        sns.despine(ax=ax)

    axes[0].set_ylabel(_metric_label(args.metric), fontsize=9.3)

    fig.legend(
        handles=_build_legend_handles(),
        loc="center left",
        ncol=1,
        bbox_to_anchor=(0.79, 0.5),
        columnspacing=0.85,
        handlelength=1.9,
        fontsize=7.3,
    )
    fig.tight_layout(rect=[0.0, 0.0, 0.78, 1.0])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
