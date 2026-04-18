"""
Calibration and error-detection metrics (Section 6.3).

Metrics implemented:
  - Expected Calibration Error (ECE)
  - Brier score
  - AUROC for error detection (confidence as score for correctness)
  - Confidence-at-error (mean confidence on incorrect predictions)
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

import config


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    n_bins: int = config.ECE_NUM_BINS,
) -> float:
    """Compute ECE (equal-width binning).

    Parameters
    ----------
    confidences : array of shape (N,)
        max(p_yes, p_no) per sample.
    accuracies : array of shape (N,)
        1 if predicted == gold, 0 otherwise.
    n_bins : int
        Number of equal-width bins.

    Returns
    -------
    float  ECE value.
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(confidences)
    for lo, hi in zip(bin_boundaries[:-1], bin_boundaries[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = accuracies[mask].mean()
        ece += mask.sum() / n * abs(bin_acc - bin_conf)
    return float(ece)


def brier_score(
    confidences: np.ndarray,
    accuracies: np.ndarray,
) -> float:
    """Brier score: mean squared error between confidence and correctness.

    Uses confidence as p(correct) so Brier = mean((c - correct)^2).
    """
    return float(np.mean((confidences - accuracies) ** 2))


def auroc_error_detection(
    confidences: np.ndarray,
    accuracies: np.ndarray,
) -> float:
    """AUROC for separating correct vs. incorrect predictions.

    We treat confidence (or -entropy) as a score for correctness.
    Returns NaN when only one class is present (e.g. all correct).
    """
    if len(np.unique(accuracies)) < 2:
        return float("nan")
    return float(roc_auc_score(accuracies, confidences))


def confidence_at_error(
    confidences: np.ndarray,
    accuracies: np.ndarray,
) -> float:
    """Mean confidence on incorrect predictions ("confidently wrong" indicator).

    Returns NaN when all predictions are correct.
    """
    wrong = accuracies == 0
    if wrong.sum() == 0:
        return float("nan")
    return float(confidences[wrong].mean())


# ── Convenience wrapper ────────────────────────────────────────────────
def compute_all_calibration_metrics(
    confidences: np.ndarray,
    accuracies: np.ndarray,
) -> dict[str, float]:
    """Return a dict with all calibration metrics."""
    return {
        "ece": expected_calibration_error(confidences, accuracies),
        "brier": brier_score(confidences, accuracies),
        "auroc": auroc_error_detection(confidences, accuracies),
        "confidence_at_error": confidence_at_error(confidences, accuracies),
    }
