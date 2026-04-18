"""Create a paper-ready selective-prediction summary figure.

The local environment used by the chat agent may not have matplotlib
installed, so this script writes a small vector PDF directly using the
standard library. It reads the saved HealthContradict conflict-prediction
results and plots held-out selective accuracy across coverage targets.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    from matplotlib.colors import LinearSegmentedColormap, Normalize
except ModuleNotFoundError:
    plt = None
    sns = None
    LinearSegmentedColormap = None
    Normalize = None

ROOT = Path(__file__).resolve().parent
RESULTS_DIR = ROOT / "results"
OUT_DIR = ROOT / ".dr-claw" / "chat-attachments" / "1775532891437"
DEFAULT_OUT_PATH = OUT_DIR / "selective_prediction_lift_heatmap.pdf"
EXCLUDED_COVERAGES = {"1.0"}
MODE_TO_KEY = {
    "threshold_transfer": "selective_prediction",
    "exact_topk": "selective_prediction_exact_topk",
}
MODE_TO_FILENAME = {
    "threshold_transfer": "selective_prediction_lift_heatmap.pdf",
    "exact_topk": "selective_prediction_lift_heatmap_exact_topk.pdf",
}

MODELS = [
    ("llama3.1-8b", "Llama-3.1-8B", (0.82, 0.10, 0.12)),
    ("meditron3-8b", "Meditron3-8B", (0.12, 0.47, 0.71)),
    ("phi4-14b", "Phi-4", (0.17, 0.63, 0.17)),
    ("qwen3-4b", "Qwen3-4B-Inst.", (0.95, 0.50, 0.05)),
    ("qwen3-8b", "Qwen3-8B", (0.58, 0.40, 0.74)),
    ("qwen3.5-9b", "Qwen3.5-9B", (0.55, 0.34, 0.29)),
]

CONDITIONS = [
    ("IC", "IC"),
    ("ICC", "ICC"),
]


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

    def rect(self, x: float, y: float, w: float, h: float, color=(0, 0, 0), fill=False, width=1.0) -> None:
        r, g, b = color
        op = "f" if fill else "S"
        prefix = "rg" if fill else "RG"
        self.ops.append(f"{r:.3f} {g:.3f} {b:.3f} {prefix} {width:.3f} w {x:.2f} {y:.2f} {w:.2f} {h:.2f} re {op}")

    def text(self, x: float, y: float, text: str, size=9, color=(0, 0, 0), bold=False, center=False) -> None:
        r, g, b = color
        font = "F2" if bold else "F1"
        if center:
            # Approximate centering is sufficient for short labels in this figure.
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
        description="Create a heatmap summary for selective-prediction lift."
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
        default=None,
        help="Optional output PDF path. Defaults depend on --mode.",
    )
    return parser.parse_args()


def load_values(
    selective_key: str,
) -> tuple[dict[str, dict[str, dict[str, list[float]]]], list[str]]:
    values: dict[str, dict[str, dict[str, list[float]]]] = {}
    coverage_keys: list[str] | None = None
    for model_key, _, _ in MODELS:
        data = json.loads((RESULTS_DIR / f"{model_key}_conflict_prediction.json").read_text(encoding="utf-8"))
        values[model_key] = {}
        for cond, _ in CONDITIONS:
            wc = data["conditions"][cond]["within_condition"]
            if selective_key not in wc:
                raise KeyError(
                    f"Missing selective prediction key '{selective_key}' for "
                    f"model={model_key}, condition={cond}."
                )
            sp = wc[selective_key]
            if coverage_keys is None:
                coverage_keys = [
                    key for key in sorted(sp.keys(), key=float, reverse=True)
                    if key not in EXCLUDED_COVERAGES
                ]
            values[model_key][cond] = {
                "combined": [sp[c]["test_accuracy_combined"] for c in coverage_keys],
                "confidence": [sp[c]["test_accuracy_conf_only"] for c in coverage_keys],
            }
    return values, coverage_keys or []


def compute_lift_matrices(
    values: dict[str, dict[str, dict[str, list[float]]]],
) -> dict[str, list[list[float]]]:
    matrices: dict[str, list[list[float]]] = {}
    for cond, _ in CONDITIONS:
        rows: list[list[float]] = []
        for model_key, _, _ in MODELS:
            combined = values[model_key][cond]["combined"]
            conf = values[model_key][cond]["confidence"]
            rows.append([(c - f) * 100.0 for c, f in zip(combined, conf)])
        matrices[cond] = rows
    return matrices


def heat_color(value: float, vmax: float) -> tuple[float, float, float]:
    """Map lift value to a sequential white-green scale with a soft underflow color."""
    if vmax <= 0.0:
        return (0.96, 0.96, 0.96)
    if value < 0.0:
        return (0.94, 0.86, 0.86)
    low = (0.98, 0.98, 0.98)
    high = (0.21, 0.56, 0.31)
    t = max(0.0, min(1.0, value / vmax))
    return tuple(low[i] + t * (high[i] - low[i]) for i in range(3))


def render_with_matplotlib(
    values: dict[str, dict[str, dict[str, list[float]]]],
    coverage_keys: list[str],
    out_path: Path,
) -> None:
    sns.set_theme(style="white", context="paper", font_scale=1.05)
    coverage_labels = [str(int(round(float(c) * 100))) for c in coverage_keys]
    model_labels = [label for _, label, _ in MODELS]
    matrices = compute_lift_matrices(values)
    all_vals = [v for rows in matrices.values() for row in rows for v in row]
    vmax = max(5.0, math.ceil(max(all_vals) / 5.0) * 5.0)

    cmap = LinearSegmentedColormap.from_list(
        "lift_map",
        ["#fbfbfb", "#2f7f3b"],
    )
    cmap.set_under("#f0d9d9")
    norm = Normalize(vmin=0.0, vmax=vmax)

    fig = plt.figure(figsize=(2.65, 1.75))
    gs = fig.add_gridspec(
        1,
        len(CONDITIONS) + 1,
        width_ratios=[1.0, 1.0, 0.036],
        wspace=0.032,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(len(CONDITIONS))]
    cax = fig.add_subplot(gs[0, len(CONDITIONS)])

    for col, (cond, title) in enumerate(CONDITIONS):
        ax = axes[col]
        data = matrices[cond]
        annot = [[f"{val:+.1f}" for val in row] for row in data]
        hm = sns.heatmap(
            data,
            ax=ax,
            cmap=cmap,
            norm=norm,
            annot=annot,
            fmt="",
            linewidths=0.6,
            linecolor="#ffffff",
            cbar=(col == len(CONDITIONS) - 1),
            cbar_ax=cax if col == len(CONDITIONS) - 1 else None,
            cbar_kws={"extend": "min"},
            xticklabels=coverage_labels,
            yticklabels=model_labels,
            annot_kws={"fontsize": 4.6},
        )
        ax.set_title(title, fontsize=5.0, fontweight="semibold", pad=0.5)
        ax.set_xlabel("Cov. (%)", fontsize=4.4)
        if col == 0:
            ax.set_ylabel("Model", fontsize=4.4, labelpad=2)
        else:
            ax.set_ylabel("")
            ax.set_yticklabels([])
        ax.tick_params(axis="x", labelrotation=0, labelsize=4.2, pad=0.5)
        ax.tick_params(axis="y", labelrotation=0, labelsize=4.2, pad=0.5)
        for spine in ax.spines.values():
            spine.set_visible(False)

        if col == len(CONDITIONS) - 1 and hm.collections:
            cbar = hm.collections[0].colorbar
            if cbar is not None:
                cbar.set_label("")
                cbar.ax.set_title("Lift", fontsize=4.6, pad=0.5, fontweight="semibold")
                cbar.set_ticks(list(range(0, int(vmax) + 1, 10)))
                cbar.ax.tick_params(labelsize=4.2)
    fig.subplots_adjust(left=0.12, right=0.992, top=0.995, bottom=0.16, wspace=0.035)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def render_with_pdf_canvas(
    values: dict[str, dict[str, dict[str, list[float]]]],
    coverage_keys: list[str],
    out_path: Path,
) -> None:
    coverage_labels = [str(int(round(float(c) * 100))) for c in coverage_keys]
    canvas = PdfCanvas(405, 218)
    matrices = compute_lift_matrices(values)
    model_labels = [label for _, label, _ in MODELS]
    all_vals = [v for rows in matrices.values() for row in rows for v in row]
    vmax = max(5.0, math.ceil(max(all_vals) / 5.0) * 5.0)

    left = 68
    top = 188
    right = 32
    gap_x = 8
    panel_w = (canvas.width - left - right - gap_x) / 2
    panel_h = 102
    cell_h = panel_h / len(MODELS)
    cell_w = panel_w / len(coverage_labels)

    for col, (cond, title) in enumerate(CONDITIONS):
        x0 = left + col * (panel_w + gap_x)
        y0 = top - panel_h
        canvas.text(x0 + panel_w / 2, top + 8, title, size=6.2, bold=True, center=True)

        for r, model_label in enumerate(model_labels):
            y = top - (r + 1) * cell_h
            if col == 0:
                canvas.text(10, y + cell_h / 2 - 3, model_label, size=6.0)
            for c, cov_label in enumerate(coverage_labels):
                x = x0 + c * cell_w
                value = matrices[cond][r][c]
                canvas.rect(x, y, cell_w, cell_h, color=heat_color(value, vmax), fill=True)
                canvas.rect(x, y, cell_w, cell_h, color=(1, 1, 1), fill=False, width=0.6)
                canvas.text(
                    x + cell_w / 2,
                    y + cell_h / 2 - 3,
                    f"{value:+.1f}",
                    size=4.9,
                    center=True,
                )

        for c, cov_label in enumerate(coverage_labels):
            canvas.text(x0 + c * cell_w + cell_w / 2, y0 - 8, cov_label, size=5.0, center=True)
        canvas.text(x0 + panel_w / 2, y0 - 17, "Cov. (%)", size=5.1, center=True)

    canvas.text(10, top + 8, "Model", size=5.4, bold=True)

    # Simple color legend
    legend_x = 360
    legend_y = 152
    legend_h = 44
    steps = 8
    for i in range(steps):
        frac = i / (steps - 1)
        value = frac * vmax
        y = legend_y - i * (legend_h / steps)
        canvas.rect(legend_x, y, 15, legend_h / steps, color=heat_color(value, vmax), fill=True)
    canvas.text(380, legend_y + 6, f"{vmax:.0f}", size=5.4)
    canvas.text(380, legend_y - legend_h + 6, "0", size=5.4)
    canvas.text(360, legend_y + 13, "Lift", size=4.8, bold=True)

    canvas.write_pdf(out_path)


def main() -> None:
    args = parse_args()
    selective_key = MODE_TO_KEY[args.mode]
    out_path = Path(args.out_path) if args.out_path else (OUT_DIR / MODE_TO_FILENAME[args.mode])
    values, coverage_keys = load_values(selective_key)
    if plt is not None and sns is not None:
        render_with_matplotlib(values, coverage_keys, out_path)
    else:
        render_with_pdf_canvas(values, coverage_keys, out_path)
    print(out_path)


if __name__ == "__main__":
    main()
