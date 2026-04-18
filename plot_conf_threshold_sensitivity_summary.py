"""Plot a publication-quality confidence-threshold sensitivity summary.

This mirrors the alpha-sensitivity figure layout: two panels (`IC` and `ICC`),
target coverage on the x-axis, and one line per confidently-wrong threshold.
The script reads the JSON summary written by
``sweep_healthcontradict_conf_threshold.py``.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
except ModuleNotFoundError:
    plt = None
    sns = None

ROOT = Path(__file__).resolve().parent
SUMMARY_PATH = (
    ROOT / "results" / "conf_threshold_sensitivity" / "hc_conf_threshold_sweep_summary.json"
)
DEFAULT_OUT_PATH = ROOT / "figures" / "conf_threshold_sensitivity_selective_gain.pdf"

CONDITIONS = [
    ("IC", "IC"),
    ("ICC", "ICC"),
]
THRESHOLD_SERIES = [
    (0.5, r"$\tau=0.5$", "#fef0d9"),
    (0.6, r"$\tau=0.6$", "#fdcc8a"),
    (0.7, r"$\tau=0.7$", "#fc8d59"),
    (0.8, r"$\tau=0.8$", "#d7301f"),
]
GRID_COLOR = "#D8DDE6"
ZERO_LINE_COLOR = "#6B7280"
MARKER_STYLE = "o"
MARKER_EDGE_COLOR = "#2D3142"
REFERENCE_THRESHOLD = 0.7
REFERENCE_LINE_COLOR = "#B45309"
TITLE_BBOX = {
    "facecolor": "#F3F4F6",
    "edgecolor": "none",
    "boxstyle": "round,pad=0.25",
}


def _color_tuple(color: str | tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(color, tuple):
        return color
    color = color.lstrip("#")
    if len(color) != 6:
        raise ValueError(f"Unsupported color format: {color}")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


class PdfCanvas:
    def __init__(self, width: float, height: float) -> None:
        self.width = width
        self.height = height
        self.ops: list[str] = []

    def line(self, x1: float, y1: float, x2: float, y2: float, color=(0, 0, 0), width=1.0) -> None:
        r, g, b = color
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} RG {width:.3f} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")

    def polyline(self, points: list[tuple[float, float]], color=(0, 0, 0), width=1.0) -> None:
        if not points:
            return
        r, g, b = color
        cmds = [f"{r:.3f} {g:.3f} {b:.3f} RG {width:.3f} w", f"{points[0][0]:.2f} {points[0][1]:.2f} m"]
        cmds.extend(f"{x:.2f} {y:.2f} l" for x, y in points[1:])
        cmds.append("S")
        self.ops.append(" ".join(cmds))

    def circle(self, x: float, y: float, radius: float, color=(0, 0, 0), fill=True, width=1.0) -> None:
        k = 0.552284749831 * radius
        r, g, b = color
        op = "f" if fill else "S"
        prefix = "rg" if fill else "RG"
        self.ops.append(
            f"{r:.3f} {g:.3f} {b:.3f} {prefix} {width:.3f} w "
            f"{x + radius:.2f} {y:.2f} m "
            f"{x + radius:.2f} {y + k:.2f} {x + k:.2f} {y + radius:.2f} {x:.2f} {y + radius:.2f} c "
            f"{x - k:.2f} {y + radius:.2f} {x - radius:.2f} {y + k:.2f} {x - radius:.2f} {y:.2f} c "
            f"{x - radius:.2f} {y - k:.2f} {x - k:.2f} {y - radius:.2f} {x:.2f} {y - radius:.2f} c "
            f"{x + k:.2f} {y - radius:.2f} {x + radius:.2f} {y - k:.2f} {x + radius:.2f} {y:.2f} c "
            f"{op}"
        )

    def rect(self, x: float, y: float, w: float, h: float, color=(0, 0, 0), fill=False, width=1.0) -> None:
        r, g, b = color
        op = "f" if fill else "S"
        prefix = "rg" if fill else "RG"
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} {prefix} {width:.3f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}")

    def text(self, x: float, y: float, text: str, size=9, color=(0, 0, 0), bold=False, center=False) -> None:
        r, g, b = color
        font = "F2" if bold else "F1"
        if center:
            x -= len(text) * size * 0.24
        self.ops.append(f"BT /{font} {size:.1f} Tf {r:.3f} {g:.3f} {b:.3f} rg {x:.2f} {y:.2f} Td ({pdf_escape(text)}) Tj ET")

    def write_pdf(self, path: Path) -> None:
        stream = "\n".join(self.ops).encode("ascii")
        objects = [
            b"<< /Type /Catalog /Pages 2 0 R >>",
            b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {self.width:.0f} {self.height:.0f}] /Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> /Contents 4 0 R >>".encode("ascii"),
            b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
            b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>",
        ]
        chunks = [b"%PDF-1.4\n"]
        offsets = [0]
        for i, obj in enumerate(objects, start=1):
            offsets.append(sum(len(c) for c in chunks))
            chunks.append(f"{i} 0 obj\n".encode("ascii") + obj + b"\nendobj\n")
        xref_offset = sum(len(c) for c in chunks)
        xref = [f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii")]
        xref.extend(f"{offset:010d} 00000 n \n".encode("ascii") for offset in offsets[1:])
        trailer = f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref_offset}\n%%EOF\n".encode("ascii")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"".join(chunks + xref + [trailer]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot HealthContradict threshold sensitivity from a saved summary."
    )
    parser.add_argument(
        "--summary-path",
        type=str,
        default=str(SUMMARY_PATH),
        help="Path to hc_conf_threshold_sweep_summary.json.",
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
        for threshold, _label, _color in THRESHOLD_SERIES:
            values = []
            for coverage_target in coverage_targets:
                match = next(
                    row for row in mean_rows
                    if abs(float(row["confidently_wrong_threshold"]) - threshold) < 1e-12
                    and row["condition"] == condition
                    and abs(float(row["coverage_target"]) - coverage_target) < 1e-12
                )
                values.append(float(match[metric]))
            series[condition][threshold] = values

    return series, coverage_levels


def _compute_y_bounds(series: dict[str, dict[float, list[float]]]) -> tuple[float, float]:
    values = [
        value
        for condition_series in series.values()
        for threshold_values in condition_series.values()
        for value in threshold_values
    ]
    if not values:
        return 0.0, 1.0
    ymin = min(values)
    ymax = max(values)
    lower = min(0.0, ymin)
    upper = max(0.0, ymax)
    margin = max(1.0, 0.08 * (upper - lower if upper > lower else 1.0))
    return lower - margin * 0.25, upper + margin


def _nice_ticks(y_min: float, y_max: float) -> list[float]:
    span = max(y_max - y_min, 1.0)
    rough_step = span / 4.0
    if rough_step <= 2:
        step = 2.0
    elif rough_step <= 5:
        step = 5.0
    elif rough_step <= 10:
        step = 10.0
    else:
        step = math.ceil(rough_step / 5.0) * 5.0
    start = math.floor(y_min / step) * step
    end = math.ceil(y_max / step) * step
    ticks = []
    value = start
    while value <= end + 1e-9:
        ticks.append(round(value, 6))
        value += step
    return ticks


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
    for threshold, label, color in THRESHOLD_SERIES:
        if threshold not in series[condition]:
            continue
        is_reference = abs(threshold - REFERENCE_THRESHOLD) < 1e-12
        ax.plot(
            coverage_levels,
            series[condition][threshold],
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
    ref_values = series[condition][REFERENCE_THRESHOLD]
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
            linewidth=3.0 if abs(threshold - REFERENCE_THRESHOLD) < 1e-12 else 2.0,
            linestyle="-" if abs(threshold - REFERENCE_THRESHOLD) < 1e-12 else "--",
            marker=MARKER_STYLE,
            markersize=5.8 if abs(threshold - REFERENCE_THRESHOLD) < 1e-12 else 4.8,
            markerfacecolor=color,
            markeredgecolor=MARKER_EDGE_COLOR,
            markeredgewidth=0.8,
            label=label,
        )
        for threshold, label, color in THRESHOLD_SERIES
    ]


def _draw_pdf_panel(
    canvas: PdfCanvas,
    x0: float,
    y0: float,
    w: float,
    h: float,
    condition: str,
    title: str,
    coverage_levels: list[int],
    series: dict[str, dict[float, list[float]]],
    y_min: float,
    y_max: float,
    y_ticks: list[float],
    show_y_axis: bool,
) -> None:
    bg = (0.980, 0.980, 0.965)
    grid = (0.847, 0.867, 0.902)
    axis = (0.188, 0.204, 0.247)
    zero = (0.420, 0.447, 0.502)

    canvas.rect(x0, y0, w, h, color=bg, fill=True)
    canvas.line(x0, y0, x0, y0 + h, color=axis, width=0.9)
    canvas.line(x0, y0, x0 + w, y0, color=axis, width=0.9)

    def map_x(index: int) -> float:
        if len(coverage_levels) == 1:
            return x0 + w / 2.0
        return x0 + (index / (len(coverage_levels) - 1)) * w

    def map_y(value: float) -> float:
        if abs(y_max - y_min) < 1e-12:
            return y0 + h / 2.0
        return y0 + (value - y_min) / (y_max - y_min) * h

    for tick in y_ticks:
        y = map_y(tick)
        canvas.line(x0, y, x0 + w, y, color=grid, width=0.6)
        if show_y_axis:
            canvas.text(x0 - 22, y - 3, f"{tick:.0f}", size=7.0, color=axis)

    zero_y = map_y(0.0)
    canvas.line(x0, zero_y, x0 + w, zero_y, color=zero, width=1.0)

    for idx, cov in enumerate(coverage_levels):
        x = map_x(idx)
        canvas.line(x, y0, x, y0 - 3, color=axis, width=0.8)
        canvas.text(x, y0 - 17, str(cov), size=7.4, color=axis, center=True)

    for threshold, _label, color in THRESHOLD_SERIES:
        values = series[condition][threshold]
        points = [(map_x(i), map_y(v)) for i, v in enumerate(values)]
        is_reference = abs(threshold - REFERENCE_THRESHOLD) < 1e-12
        pdf_color = _color_tuple(color)
        canvas.polyline(points, color=pdf_color, width=2.1 if is_reference else 1.45)
        for px, py in points:
            canvas.circle(px, py, 2.5 if is_reference else 2.0, color=pdf_color, fill=True)

    ref_values = series[condition][REFERENCE_THRESHOLD]
    canvas.text(
        x0 + w + 5,
        map_y(ref_values[-1]) - 3,
        f"{ref_values[-1]:+.1f}",
        size=7.2,
        color=_color_tuple(REFERENCE_LINE_COLOR),
        bold=True,
    )

    title_w = 28 if title == "IC" else 32
    canvas.rect(x0 + w / 2 - title_w / 2, y0 + h + 6, title_w, 11, color=(0.953, 0.957, 0.965), fill=True)
    canvas.text(x0 + w / 2, y0 + h + 9.5, title, size=9.0, color=axis, bold=True, center=True)


def render_with_pdf_canvas(
    series: dict[str, dict[float, list[float]]],
    coverage_levels: list[int],
    out_path: Path,
    metric_label: str,
) -> None:
    y_min, y_max = _compute_y_bounds(series)
    y_ticks = _nice_ticks(y_min, y_max)
    canvas = PdfCanvas(520, 208)

    left = 54
    right = 112
    bottom = 38
    top = 34
    gap = 32
    panel_w = (canvas.width - left - right - gap) / 2.0
    panel_h = canvas.height - bottom - top

    for idx, (condition, title) in enumerate(CONDITIONS):
        x0 = left + idx * (panel_w + gap)
        _draw_pdf_panel(
            canvas,
            x0=x0,
            y0=bottom,
            w=panel_w,
            h=panel_h,
            condition=condition,
            title=title,
            coverage_levels=coverage_levels,
            series=series,
            y_min=y_min,
            y_max=y_max,
            y_ticks=y_ticks,
            show_y_axis=(idx == 0),
        )
        canvas.text(x0 + panel_w / 2, 12, "Coverage (%)", size=8.0, color=(0.188, 0.204, 0.247), center=True)

    canvas.text(14, bottom + panel_h / 2, metric_label, size=8.0, color=(0.188, 0.204, 0.247))

    legend_x = canvas.width - 88
    legend_y = bottom + panel_h * 0.58
    for row_idx, (_threshold, label, color) in enumerate(THRESHOLD_SERIES):
        y = legend_y - row_idx * 16
        pdf_color = _color_tuple(color)
        canvas.line(legend_x, y, legend_x + 18, y, color=pdf_color, width=1.8)
        canvas.circle(legend_x + 9, y, 2.1, color=pdf_color, fill=True)
        canvas.text(legend_x + 24, y - 3, label.replace("$", ""), size=7.2, color=(0.188, 0.204, 0.247))

    canvas.write_pdf(out_path)


def main() -> None:
    args = parse_args()
    summary_path = Path(args.summary_path)
    out_path = Path(args.out_path)
    summary = _load_summary(summary_path)
    series, coverage_levels = _collect_series(summary, args.metric)
    if plt is not None and sns is not None:
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
    else:
        render_with_pdf_canvas(
            series=series,
            coverage_levels=coverage_levels,
            out_path=out_path,
            metric_label=_metric_label(args.metric),
        )


if __name__ == "__main__":
    main()
