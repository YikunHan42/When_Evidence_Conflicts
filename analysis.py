"""
Cross-condition analysis (Section 7).

Addresses the three research questions:
  RQ1  Conflict sensitivity  – IC vs. CIC/ICC uncertainty shift
  RQ2  Order effects         – CIC vs. ICC margin / prediction shifts
  RQ3  Abstention utility    – risk–coverage comparison
"""

from __future__ import annotations

import numpy as np
from scipy import stats

from uncertainty import UncertaintyScores


# ── RQ1: Conflict sensitivity ──────────────────────────────────────────
def conflict_sensitivity(
    ic_scores: list[UncertaintyScores],
    cic_scores: list[UncertaintyScores],
    icc_scores: list[UncertaintyScores],
) -> dict:
    """Compare uncertainty distributions: IC vs. CIC and IC vs. ICC.

    Reports mean shifts and paired Wilcoxon signed-rank test p-values
    for logit margin, entropy, and confidence.
    """
    results: dict = {}
    for tag, conflict_scores in [("IC_vs_CIC", cic_scores), ("IC_vs_ICC", icc_scores)]:
        r: dict = {}
        for metric, getter in [
            ("logit_margin", lambda s: s.logit_margin),
            ("entropy", lambda s: s.entropy),
            ("confidence", lambda s: s.confidence),
        ]:
            ic_vals = np.array([getter(s) for s in ic_scores])
            cf_vals = np.array([getter(s) for s in conflict_scores])
            diff = cf_vals - ic_vals
            stat, p = stats.wilcoxon(diff, alternative="two-sided")
            r[metric] = {
                "ic_mean": float(ic_vals.mean()),
                "conflict_mean": float(cf_vals.mean()),
                "mean_shift": float(diff.mean()),
                "wilcoxon_stat": float(stat),
                "wilcoxon_p": float(p),
            }
        results[tag] = r
    return results


# ── RQ2: Order effects ─────────────────────────────────────────────────
def order_effects(
    cic_scores: list[UncertaintyScores],
    icc_scores: list[UncertaintyScores],
    cic_preds: list[str],
    icc_preds: list[str],
) -> dict:
    """Quantify CIC vs. ICC differences (recency / position bias).

    Reports:
      - margin shift (paired Wilcoxon)
      - prediction flip rate (fraction of samples that change answer)
    """
    n = len(cic_scores)
    cic_margins = np.array([s.logit_margin for s in cic_scores])
    icc_margins = np.array([s.logit_margin for s in icc_scores])
    diff = icc_margins - cic_margins

    stat, p = stats.wilcoxon(diff, alternative="two-sided")
    flip_rate = sum(1 for a, b in zip(cic_preds, icc_preds) if a != b) / n

    cic_conf = np.array([s.confidence for s in cic_scores])
    icc_conf = np.array([s.confidence for s in icc_scores])
    conf_stat, conf_p = stats.wilcoxon(icc_conf - cic_conf, alternative="two-sided")

    return {
        "margin_shift_mean": float(diff.mean()),
        "margin_wilcoxon_stat": float(stat),
        "margin_wilcoxon_p": float(p),
        "prediction_flip_rate": float(flip_rate),
        "confidence_shift_mean": float((icc_conf - cic_conf).mean()),
        "confidence_wilcoxon_stat": float(conf_stat),
        "confidence_wilcoxon_p": float(conf_p),
    }


# ── RQ3: Abstention utility (thin wrapper) ────────────────────────────
def abstention_summary(
    rc_curves: dict[str, list],
    auc_func,
    cov_func,
) -> dict:
    """Summarise risk–coverage per condition.

    Parameters
    ----------
    rc_curves : {condition: list[RiskCoveragePoint]}
    auc_func  : callable(curve) -> float   (auc_risk_coverage)
    cov_func  : callable(curve, max_risk) -> float   (coverage_at_risk)
    """
    summary: dict = {}
    for cond, curve in rc_curves.items():
        summary[cond] = {
            "auc_risk_coverage": auc_func(curve),
            "coverage_at_10pct_risk": cov_func(curve, 0.10),
            "coverage_at_5pct_risk": cov_func(curve, 0.05),
        }
    return summary
