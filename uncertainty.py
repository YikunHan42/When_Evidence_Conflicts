"""
Uncertainty quantification from Yes/No logits (Section 6.2).

Given p(Yes) and p(No) (normalised over the constrained answer set), compute:
  - Logit margin   m = log p(Yes) - log p(No)   (signed toward predicted label)
  - Binary entropy H = -p log p - (1-p) log(1-p)   where p = max(p_yes, p_no)
  - Confidence     c = max(p_yes, p_no)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from inference import PredictionResult


@dataclass
class UncertaintyScores:
    """Uncertainty signals for a single prediction."""
    logit_margin: float   # signed toward predicted label
    entropy: float        # binary entropy
    confidence: float     # max(p_yes, p_no)


def compute_uncertainty(pred: PredictionResult) -> UncertaintyScores:
    """Derive uncertainty signals from a single PredictionResult."""
    p_yes, p_no = pred.p_yes, pred.p_no

    # Logit margin (signed toward predicted label)
    raw_margin = math.log(p_yes + 1e-12) - math.log(p_no + 1e-12)
    logit_margin = raw_margin if pred.predicted_label == "yes" else -raw_margin

    # Confidence
    confidence = max(p_yes, p_no)

    # Binary entropy:  H = -p log p - (1-p) log(1-p)  where p = confidence
    p = confidence
    if p >= 1.0:
        entropy = 0.0
    else:
        entropy = -p * math.log(p + 1e-12) - (1.0 - p) * math.log(1.0 - p + 1e-12)

    return UncertaintyScores(
        logit_margin=logit_margin,
        entropy=entropy,
        confidence=confidence,
    )


def compute_uncertainty_batch(
    preds: list[PredictionResult],
) -> list[UncertaintyScores]:
    """Compute uncertainty for a list of predictions."""
    return [compute_uncertainty(p) for p in preds]


# ── Aggregation helpers (used by analysis module) ──────────────────────
def summarise_uncertainty(
    scores: list[UncertaintyScores],
) -> dict[str, float]:
    """Return summary statistics for a list of uncertainty scores."""
    margins = np.array([s.logit_margin for s in scores])
    entropies = np.array([s.entropy for s in scores])
    confs = np.array([s.confidence for s in scores])
    return {
        "margin_mean": float(np.mean(margins)),
        "margin_std": float(np.std(margins)),
        "margin_median": float(np.median(margins)),
        "entropy_mean": float(np.mean(entropies)),
        "entropy_std": float(np.std(entropies)),
        "entropy_median": float(np.median(entropies)),
        "confidence_mean": float(np.mean(confs)),
        "confidence_std": float(np.std(confs)),
        "confidence_median": float(np.median(confs)),
    }
