"""
Visualization functions for the Conflict-Aware Selective Prediction analysis.

Generates publication-quality figures:
  1. Conflict score distributions (correct vs. incorrect)
  2. Risk-coverage curves comparing abstention signals
  3. Feature importance bar chart
  4. Multi-condition comparison table
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from conflict_detector import DetectorResult
from selective_prediction import (
    RiskCoveragePoint,
    risk_coverage_curve,
)

sns.set_theme(style="whitegrid", font_scale=1.2)

SIGNAL_COLORS = {
    "Confidence (baseline)": "#8172B3",
    "Conflict score": "#C44E52",
    "Combined": "#55A868",
}

CONDITION_COLORS = {
    "NC": "#4C72B0",
    "CC": "#55A868",
    "IC": "#C44E52",
    "CIC": "#8172B3",
    "ICC": "#CCB974",
}

CONDITION_ORDER = ["NC", "CC", "IC", "CIC", "ICC"]


def _savefig(fig: plt.Figure, name: str, out_dir: Path | None = None) -> Path:
    d = out_dir or config.FIGURES_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


# ── 1. Conflict score distribution ───────────────────────────────────
def plot_conflict_score_distribution(
    results: list[DetectorResult],
    condition: str,
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    """Histogram of conflict scores split by correct vs. incorrect predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    correct_mask = np.array([r.is_wrong == 0 for r in results])
    conflict_scores = np.array([r.conflict_score for r in results])
    confidences = np.array([r.model_confidence for r in results])

    bins = np.linspace(0, 1, 30)

    # Panel 1: Conflict score
    ax = axes[0]
    ax.hist(conflict_scores[correct_mask], bins=bins, alpha=0.6,
            color="#55A868", label="Correct", density=True)
    ax.hist(conflict_scores[~correct_mask], bins=bins, alpha=0.6,
            color="#C44E52", label="Incorrect", density=True)
    ax.set_xlabel("Conflict Score (P(confidently wrong))")
    ax.set_ylabel("Density")
    ax.set_title(f"Conflict Score — {condition}")
    ax.legend()

    # Panel 2: Model confidence
    ax = axes[1]
    ax.hist(confidences[correct_mask], bins=bins, alpha=0.6,
            color="#55A868", label="Correct", density=True)
    ax.hist(confidences[~correct_mask], bins=bins, alpha=0.6,
            color="#C44E52", label="Incorrect", density=True)
    ax.set_xlabel("Model Confidence")
    ax.set_ylabel("Density")
    ax.set_title(f"Model Confidence — {condition}")
    ax.legend()

    fig.suptitle(
        f"Conflict Score vs. Confidence as Error Discriminators — {model_name} ({condition})",
        fontsize=13,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return _savefig(fig, f"conflict_dist_{condition}_{model_name}", out_dir)


# ── 2. Risk-coverage curves comparing abstention signals ─────────────
def plot_conflict_risk_coverage(
    results: list[DetectorResult],
    condition: str,
    model_name: str,
    confidence_scores: np.ndarray,
    conflict_scores: np.ndarray,
    combined_scores: np.ndarray,
    out_dir: Path | None = None,
) -> Path:
    """Risk-coverage curves for three abstention signals."""
    fig, ax = plt.subplots(figsize=(8, 6))

    correct = np.array([1 - r.is_wrong for r in results])
    tau_grid = [i / 100 for i in range(0, 101)]

    signals = {
        "Confidence (baseline)": confidence_scores,
        "Conflict score": conflict_scores,
        "Combined": combined_scores,
    }

    for label, scores in signals.items():
        rc = risk_coverage_curve(scores, correct, tau_grid=tau_grid)
        rc_sorted = sorted(rc, key=lambda p: p.coverage)
        ax.plot(
            [p.coverage for p in rc_sorted],
            [p.risk for p in rc_sorted],
            label=label,
            color=SIGNAL_COLORS[label],
            linewidth=2,
        )

    ax.axhline(0.10, color="gray", linestyle=":", linewidth=0.8, label="10% risk")
    ax.axhline(0.05, color="gray", linestyle="--", linewidth=0.8, label="5% risk")
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (error rate)")
    ax.set_title(f"Risk–Coverage: Conflict Score vs. Confidence — {model_name} ({condition})")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.02, 0.8)

    return _savefig(fig, f"conflict_risk_coverage_{condition}_{model_name}", out_dir)


# ── 3. Multi-condition AUROC comparison ──────────────────────────────
def plot_auroc_comparison(
    condition_metrics: dict[str, dict],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    """Bar chart comparing AUROC of conflict score vs. confidence across conditions."""
    fig, ax = plt.subplots(figsize=(10, 5))

    conditions = list(condition_metrics.keys())
    x = np.arange(len(conditions))
    width = 0.35

    auroc_conf = [condition_metrics[c]["auroc_confidence"] for c in conditions]
    auroc_conflict = [condition_metrics[c]["auroc_conflict_score"] for c in conditions]

    bars1 = ax.bar(x - width / 2, auroc_conf, width, label="Confidence (baseline)",
                   color="#8172B3", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width / 2, auroc_conflict, width, label="Conflict score",
                   color="#C44E52", edgecolor="black", linewidth=0.5)

    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.3f}", ha="center", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=10)
    ax.set_ylabel("AUROC (error detection)")
    ax.set_ylim(0, 1.05)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, label="Random")
    ax.set_title(f"Error Detection AUROC: Conflict Score vs. Confidence — {model_name}")
    ax.legend(fontsize=10)

    return _savefig(fig, f"conflict_auroc_comparison_{model_name}", out_dir)


# ── 4. Summary table ─────────────────────────────────────────────────
def plot_conflict_summary_table(
    condition_metrics: dict[str, dict],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    """Render a summary table of conflict detector performance across conditions."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.axis("off")

    headers = [
        "Condition", "Accuracy", "AUROC\n(confidence)", "AUROC\n(conflict)",
        "AUROC\n(combined)", "AUC-RC\n(confidence)", "AUC-RC\n(conflict)",
        "AUC-RC\n(combined)", "Cov@10%\n(confidence)", "Cov@10%\n(conflict)",
        "Cov@10%\n(combined)",
    ]

    rows = []
    for cond, m in condition_metrics.items():
        rows.append([
            cond,
            f"{m.get('accuracy', 0):.3f}",
            f"{m.get('auroc_confidence', float('nan')):.3f}",
            f"{m.get('auroc_conflict_score', float('nan')):.3f}",
            f"{m.get('auroc_combined', float('nan')):.3f}",
            f"{m.get('auc_rc_confidence', float('nan')):.3f}",
            f"{m.get('auc_rc_conflict', float('nan')):.3f}",
            f"{m.get('auc_rc_combined', float('nan')):.3f}",
            f"{m.get('cov10_confidence', 0):.1%}",
            f"{m.get('cov10_conflict', 0):.1%}",
            f"{m.get('cov10_combined', 0):.1%}",
        ])

    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.5)

    # Style header row
    for j in range(len(headers)):
        cell = table[0, j]
        cell.set_facecolor("#4C72B0")
        cell.set_text_props(color="white", fontweight="bold")

    # Highlight IC row (where conflict detection matters most)
    for cond_idx, cond in enumerate(condition_metrics.keys()):
        if cond == "IC":
            for j in range(len(headers)):
                table[cond_idx + 1, j].set_facecolor("#FFEEDD")

    ax.set_title(
        f"Conflict-Aware Selective Prediction Summary — {model_name}",
        fontsize=13, pad=20,
    )

    return _savefig(fig, f"conflict_summary_table_{model_name}", out_dir)


# ── 5. Selective prediction accuracy bar charts ──────────────────────

def _plot_selective_prediction_bars(
    all_selective: dict[str, dict],
    model_name: str,
    signal_key: str,
    signal_label: str,
    out_dir: Path | None = None,
) -> Path:
    """Grouped bar chart: accuracy at each coverage target across all conditions.

    Parameters
    ----------
    all_selective : dict
        Mapping condition -> selective_prediction dict (keys are coverage
        target strings like "0.9", "0.7", etc.).
    signal_key : str
        "test_accuracy_combined" or "test_accuracy_conf_only".
    signal_label : str
        Human-readable label, e.g. "Combined" or "Confidence-only".
    """
    conditions = [c for c in CONDITION_ORDER if c in all_selective]
    # Collect coverage targets from first condition
    first_sp = all_selective[conditions[0]]
    cov_targets = sorted(first_sp.keys(), key=lambda x: -float(x))

    n_conds = len(conditions)
    n_targets = len(cov_targets)
    x = np.arange(n_conds)
    total_width = 0.75
    bar_width = total_width / (n_targets + 1)  # +1 for baseline

    fig, ax = plt.subplots(figsize=(12, 6))

    # Color palette for coverage targets
    cov_colors = ["#A8D8EA", "#4C72B0", "#C44E52", "#DD8452"]
    if len(cov_colors) < n_targets:
        cov_colors = plt.cm.viridis(np.linspace(0.2, 0.9, n_targets)).tolist()

    # Plot baseline bars first
    baselines = []
    for cond in conditions:
        sp = all_selective[cond]
        any_target = next(iter(sp.values()))
        baselines.append(any_target["baseline_accuracy"])

    offset = -total_width / 2
    bars_base = ax.bar(
        x + offset + bar_width / 2, baselines, bar_width,
        label="Baseline (100%)", color="#CCCCCC", edgecolor="black", linewidth=0.5,
    )

    # Plot each coverage target
    for i, cov_str in enumerate(cov_targets):
        cov_pct = f"{float(cov_str):.0%}"
        accs = []
        for cond in conditions:
            m = all_selective[cond][cov_str]
            accs.append(m[signal_key])

        offset_i = offset + (i + 1) * bar_width + bar_width / 2
        bars = ax.bar(
            x + offset_i, accs, bar_width,
            label=f"Cov {cov_pct}", color=cov_colors[i % len(cov_colors)],
            edgecolor="black", linewidth=0.5,
        )

        # Add value labels on bars
        for bar, acc in zip(bars, accs):
            h = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2, h + 0.008,
                f"{acc:.0%}", ha="center", va="bottom", fontsize=7,
            )

    # Add baseline value labels
    for bar, acc in zip(bars_base, baselines):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2, h + 0.008,
            f"{acc:.0%}", ha="center", va="bottom", fontsize=7,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=11)
    ax.set_ylabel("Accuracy on answered instances")
    ax.set_ylim(0, 1.12)
    ax.set_title(
        f"Selective Prediction Accuracy ({signal_label}) — {model_name}",
        fontsize=13,
    )
    ax.legend(fontsize=9, loc="upper right", ncol=n_targets + 1)

    fig.tight_layout()

    suffix = signal_label.lower().replace(" ", "_").replace("-", "_")
    return _savefig(fig, f"selective_acc_{suffix}_{model_name}", out_dir)


def plot_selective_prediction_combined(
    all_selective: dict[str, dict],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    """Bar chart of selective prediction accuracy using the *combined* signal."""
    return _plot_selective_prediction_bars(
        all_selective, model_name,
        signal_key="test_accuracy_combined",
        signal_label="Combined",
        out_dir=out_dir,
    )


def plot_selective_prediction_conf_only(
    all_selective: dict[str, dict],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    """Bar chart of selective prediction accuracy using *confidence-only*."""
    return _plot_selective_prediction_bars(
        all_selective, model_name,
        signal_key="test_accuracy_conf_only",
        signal_label="Confidence-only",
        out_dir=out_dir,
    )
