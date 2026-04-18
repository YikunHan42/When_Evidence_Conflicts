"""
Plotting utilities for all evaluation results.

Generates publication-quality figures saved to config.FIGURES_DIR.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

import config
from selective_prediction import RiskCoveragePoint
from uncertainty import UncertaintyScores

# ── Style defaults ─────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
CONDITION_COLORS = {
    "NC": "#4C72B0",
    "CC": "#55A868",
    "IC": "#C44E52",
    "CIC": "#8172B3",
    "ICC": "#CCB974",
}
CONDITION_ORDER = config.CONDITIONS


def _savefig(fig: plt.Figure, name: str, out_dir: Path | None = None) -> Path:
    d = out_dir or config.FIGURES_DIR
    d.mkdir(parents=True, exist_ok=True)
    path = d / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)
    return path


# ── 1. Accuracy bar chart per condition ────────────────────────────────
def plot_accuracy_bars(
    acc_by_condition: dict[str, float],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4))
    conds = [c for c in CONDITION_ORDER if c in acc_by_condition]
    vals = [acc_by_condition[c] for c in conds]
    colors = [CONDITION_COLORS[c] for c in conds]
    ax.bar(conds, vals, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Answer Accuracy by Context Condition – {model_name}")
    ax.set_ylim(0, 1.05)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=10)
    return _savefig(fig, f"accuracy_{model_name}", out_dir)


# ── 2. Uncertainty distribution box/violin plots ───────────────────────
def plot_uncertainty_distributions(
    scores_by_condition: dict[str, list[UncertaintyScores]],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    metrics = [
        ("logit_margin", "Logit Margin"),
        ("entropy", "Binary Entropy"),
        ("confidence", "Confidence"),
    ]

    for ax, (attr, label) in zip(axes, metrics):
        data, labels = [], []
        for cond in CONDITION_ORDER:
            if cond not in scores_by_condition:
                continue
            vals = [getattr(s, attr) for s in scores_by_condition[cond]]
            data.append(vals)
            labels.append(cond)
        parts = ax.violinplot(data, showmeans=True, showmedians=True)
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(CONDITION_COLORS[labels[i]])
            pc.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        ax.set_ylabel(label)

    fig.suptitle(f"Uncertainty Distributions – {model_name}", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return _savefig(fig, f"uncertainty_dist_{model_name}", out_dir)


# ── 3. Calibration reliability diagram ─────────────────────────────────
def plot_reliability_diagram(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    condition: str,
    model_name: str,
    n_bins: int = config.ECE_NUM_BINS,
    out_dir: Path | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(5, 5))
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bin_centers, bin_accs, bin_counts = [], [], []
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_centers.append((lo + hi) / 2)
        bin_accs.append(accuracies[mask].mean())
        bin_counts.append(mask.sum())

    ax.bar(bin_centers, bin_accs, width=1.0 / n_bins * 0.8,
           color=CONDITION_COLORS.get(condition, "#888"),
           edgecolor="black", linewidth=0.5, alpha=0.7, label="Accuracy")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability Diagram – {condition} – {model_name}")
    ax.legend(loc="upper left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    return _savefig(fig, f"reliability_{condition}_{model_name}", out_dir)


# ── 4. Risk–coverage curves ───────────────────────────────────────────
def plot_risk_coverage(
    curves_by_condition: dict[str, list[RiskCoveragePoint]],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(7, 5))
    for cond in CONDITION_ORDER:
        if cond not in curves_by_condition:
            continue
        pts = sorted(curves_by_condition[cond], key=lambda p: p.coverage)
        coverages = [p.coverage for p in pts]
        risks = [p.risk for p in pts]
        ax.plot(coverages, risks, marker=".", label=cond,
                color=CONDITION_COLORS[cond], linewidth=1.5)
    ax.set_xlabel("Coverage")
    ax.set_ylabel("Risk (error rate)")
    ax.set_title(f"Risk–Coverage Curves – {model_name}")
    ax.legend()
    ax.set_xlim(0, 1.05)
    ax.set_ylim(-0.02, 1.0)
    ax.axhline(0.10, color="gray", linestyle=":", linewidth=0.8, label="10% risk")
    return _savefig(fig, f"risk_coverage_{model_name}", out_dir)


# ── 5. Order-effect paired scatter (CIC vs ICC margins) ───────────────
def plot_order_effect_scatter(
    cic_scores: list[UncertaintyScores],
    icc_scores: list[UncertaintyScores],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    fig, ax = plt.subplots(figsize=(6, 6))
    cic_m = [s.logit_margin for s in cic_scores]
    icc_m = [s.logit_margin for s in icc_scores]
    ax.scatter(cic_m, icc_m, alpha=0.4, s=15, color="#8172B3")
    lims = [
        min(min(cic_m), min(icc_m)) - 0.5,
        max(max(cic_m), max(icc_m)) + 0.5,
    ]
    ax.plot(lims, lims, "k--", linewidth=0.8)
    ax.set_xlabel("CIC logit margin")
    ax.set_ylabel("ICC logit margin")
    ax.set_title(f"Order Effect: CIC vs. ICC – {model_name}")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    return _savefig(fig, f"order_scatter_{model_name}", out_dir)


# ── 6. Calibration metrics summary heatmap ─────────────────────────────
def plot_calibration_heatmap(
    metrics_by_condition: dict[str, dict[str, float]],
    model_name: str,
    out_dir: Path | None = None,
) -> Path:
    conds = [c for c in CONDITION_ORDER if c in metrics_by_condition]
    metric_names = ["ece", "brier", "auroc", "confidence_at_error"]
    display_names = ["ECE", "Brier", "AUROC", "Conf@Error"]

    data = np.array([
        [metrics_by_condition[c].get(m, np.nan) for m in metric_names]
        for c in conds
    ])

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.heatmap(data, annot=True, fmt=".3f", xticklabels=display_names,
                yticklabels=conds, cmap="YlOrRd", ax=ax, linewidths=0.5)
    ax.set_title(f"Calibration Metrics – {model_name}")
    return _savefig(fig, f"calibration_heatmap_{model_name}", out_dir)
