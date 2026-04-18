"""
Selective prediction / risk–coverage analysis (Section 6.3).

Simulates a deployment policy: answer only if confidence >= tau,
otherwise abstain.  For varying tau we compute:
  - Coverage: fraction of samples answered.
  - Risk: error rate among answered samples.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import config

# NumPy compat: trapezoid was added in NumPy 2.0; older versions use trapz
_trapz = getattr(np, "trapezoid", None) or np.trapz


@dataclass
class RiskCoveragePoint:
    """A single (tau, coverage, risk) triple."""
    tau: float
    coverage: float
    risk: float


def risk_coverage_curve(
    confidences: np.ndarray,
    accuracies: np.ndarray,
    tau_grid: list[float] | None = None,
) -> list[RiskCoveragePoint]:
    """Sweep confidence thresholds and compute (coverage, risk) pairs.

    Parameters
    ----------
    confidences : shape (N,)
    accuracies  : shape (N,), 1=correct, 0=incorrect
    tau_grid    : list of threshold values (default from config)

    Returns
    -------
    List of RiskCoveragePoint, one per tau.
    """
    if tau_grid is None:
        tau_grid = config.TAU_GRID

    curve: list[RiskCoveragePoint] = []
    n = len(confidences)
    for tau in tau_grid:
        mask = confidences >= tau
        coverage = mask.sum() / n if n > 0 else 0.0
        if mask.sum() == 0:
            risk = 0.0  # nothing answered → no errors
        else:
            risk = 1.0 - accuracies[mask].mean()
        curve.append(RiskCoveragePoint(tau=tau, coverage=coverage, risk=risk))
    return curve


def auc_risk_coverage(curve: list[RiskCoveragePoint]) -> float:
    """Area under the risk–coverage curve (lower is better).

    Uses the trapezoidal rule on sorted (coverage, risk) pairs.
    """
    pts = sorted(curve, key=lambda p: p.coverage)
    coverages = np.array([p.coverage for p in pts])
    risks = np.array([p.risk for p in pts])
    return float(_trapz(risks, coverages))


def coverage_at_risk(
    curve: list[RiskCoveragePoint],
    max_risk: float = 0.10,
) -> float:
    """Maximum coverage achievable while keeping risk <= max_risk.

    Returns 0.0 if no threshold satisfies the constraint.
    """
    feasible = [p for p in curve if p.risk <= max_risk]
    if not feasible:
        return 0.0
    return max(p.coverage for p in feasible)
