"""
Main pipeline: Uncertainty-Aware Evaluation of Biomedical LLMs
under Conflicting Evidence Contexts.

Usage
-----
  python main.py                            # run all models, all conditions
  python main.py --models llama3.1-8b       # single model
  python main.py --conditions NC IC CIC ICC # subset of conditions
  python main.py --max-samples 20           # quick debug run
  python main.py --use-prebuilt-prompts     # use HealthContradict prompts_N.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

import config
from analysis import abstention_summary, conflict_sensitivity, order_effects
from calibration import compute_all_calibration_metrics
from data_loader import (
    HealthContradictSample,
    load_all_prompts,
    load_dataset,
)
from inference import LogitExtractor, PredictionResult
from prompts import build_prompt
from selective_prediction import (
    auc_risk_coverage,
    coverage_at_risk,
    risk_coverage_curve,
)
from uncertainty import (
    UncertaintyScores,
    compute_uncertainty_batch,
    summarise_uncertainty,
)
from visualization import (
    plot_accuracy_bars,
    plot_calibration_heatmap,
    plot_order_effect_scatter,
    plot_reliability_diagram,
    plot_risk_coverage,
    plot_uncertainty_distributions,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


# ── Argument parsing ───────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Uncertainty-aware biomedical LLM evaluation")
    p.add_argument("--models", nargs="+", default=list(config.MODELS.keys()),
                    help="Model keys to evaluate (default: all)")
    p.add_argument("--model-root", type=str, default=None,
                    help="Optional local directory containing one subfolder per model key")
    p.add_argument("--conditions", nargs="+", default=config.CONDITIONS,
                    help="Context conditions (default: NC CC IC CIC ICC)")
    p.add_argument("--use-prebuilt-prompts", action="store_true",
                    help="Use pre-built prompts_N.jsonl from HealthContradict repo")
    p.add_argument("--dataset-path", type=str, default=None,
                    help="Path to dataset_ready.jsonl (default: auto-detect)")
    p.add_argument("--max-samples", type=int, default=None,
                    help="Cap the number of samples (for debugging)")
    p.add_argument("--max-input-tokens", type=int, default=None,
                    help="Optional truncation cap applied before model forward pass")
    p.add_argument("--out-dir", type=str, default=str(config.RESULTS_DIR),
                    help="Directory for result JSON files")
    return p.parse_args()


def _maybe_use_local_model_paths(model_root: str | None, model_keys: list[str]) -> None:
    if not model_root:
        return
    root = Path(model_root)
    for model_key in model_keys:
        local_path = root / model_key
        if local_path.exists():
            config.MODELS[model_key] = str(local_path.resolve())
            logger.info("Using local checkpoint for %s -> %s", model_key, local_path.resolve())
        else:
            logger.warning("Local checkpoint not found for %s under %s", model_key, root)


# ── Build prompt text for a condition ──────────────────────────────────
def get_prompts_for_condition(
    condition: str,
    samples: list[HealthContradictSample],
    prebuilt: dict | None,
) -> list[str]:
    """Return a list of prompt strings aligned with `samples`.

    Uses pre-built prompts if available, otherwise builds them on the fly.
    """
    if prebuilt and condition in prebuilt:
        prompt_instances = prebuilt[condition]
        # Ensure alignment by instance_id
        by_id = {p.instance_id: p.prompt for p in prompt_instances}
        return [by_id[s.instance_id] for s in samples]
    return [build_prompt(condition, s) for s in samples]


# ── Per-model evaluation ───────────────────────────────────────────────
def evaluate_model(
    model_key: str,
    samples: list[HealthContradictSample],
    conditions: list[str],
    prebuilt: dict | None,
    out_dir: Path,
    max_input_tokens: int | None = None,
) -> dict:
    """Run full evaluation for one model across all requested conditions."""
    extractor = LogitExtractor(model_key, max_input_tokens=max_input_tokens)

    # Storage keyed by condition
    preds_by_cond: dict[str, list[PredictionResult]] = {}
    unc_by_cond: dict[str, list[UncertaintyScores]] = {}
    acc_by_cond: dict[str, float] = {}
    cal_by_cond: dict[str, dict[str, float]] = {}
    rc_by_cond: dict[str, list] = {}

    for cond in conditions:
        logger.info("── %s | condition=%s (%d samples) ──",
                     model_key, cond, len(samples))

        # Get prompt texts
        prompt_texts = get_prompts_for_condition(cond, samples, prebuilt)

        # Inference
        preds: list[PredictionResult] = []
        for prompt in tqdm(prompt_texts, desc=f"{model_key}/{cond}"):
            preds.append(extractor.predict(prompt))
        preds_by_cond[cond] = preds

        # Uncertainty
        unc = compute_uncertainty_batch(preds)
        unc_by_cond[cond] = unc

        # Accuracy
        correct = np.array([
            int(p.predicted_label == s.answer)
            for p, s in zip(preds, samples)
        ])
        acc_by_cond[cond] = float(correct.mean())
        logger.info("  accuracy = %.4f", acc_by_cond[cond])

        # Calibration
        confs = np.array([u.confidence for u in unc])
        cal_by_cond[cond] = compute_all_calibration_metrics(confs, correct)
        logger.info("  ECE=%.4f  Brier=%.4f  AUROC=%.4f",
                     cal_by_cond[cond]["ece"],
                     cal_by_cond[cond]["brier"],
                     cal_by_cond[cond]["auroc"])

        # Risk–coverage
        rc_by_cond[cond] = risk_coverage_curve(confs, correct)

        # Reliability diagram
        plot_reliability_diagram(confs, correct, cond, model_key,
                                 out_dir=config.FIGURES_DIR / model_key)

        # Save per-condition raw predictions
        _save_predictions(preds, unc, samples, cond, model_key, out_dir)

    # ── Cross-condition analyses ───────────────────────────────────────
    result: dict = {
        "model": model_key,
        "n_samples": len(samples),
        "accuracy": acc_by_cond,
        "calibration": cal_by_cond,
        "uncertainty_summary": {
            cond: summarise_uncertainty(unc_by_cond[cond])
            for cond in conditions
        },
    }

    # RQ1: conflict sensitivity
    if all(c in unc_by_cond for c in ("IC", "CIC", "ICC")):
        result["conflict_sensitivity"] = conflict_sensitivity(
            unc_by_cond["IC"], unc_by_cond["CIC"], unc_by_cond["ICC"],
        )
        logger.info("  RQ1 conflict-sensitivity computed")

    # RQ2: order effects
    if all(c in unc_by_cond for c in ("CIC", "ICC")):
        result["order_effects"] = order_effects(
            unc_by_cond["CIC"], unc_by_cond["ICC"],
            [p.predicted_label for p in preds_by_cond["CIC"]],
            [p.predicted_label for p in preds_by_cond["ICC"]],
        )
        logger.info("  RQ2 order-effects computed")

    # RQ3: abstention
    result["abstention"] = abstention_summary(
        rc_by_cond, auc_risk_coverage, coverage_at_risk,
    )

    # ── Plots ──────────────────────────────────────────────────────────
    fig_dir = config.FIGURES_DIR / model_key
    plot_accuracy_bars(acc_by_cond, model_key, out_dir=fig_dir)
    plot_uncertainty_distributions(unc_by_cond, model_key, out_dir=fig_dir)
    plot_risk_coverage(rc_by_cond, model_key, out_dir=fig_dir)
    plot_calibration_heatmap(cal_by_cond, model_key, out_dir=fig_dir)
    if all(c in unc_by_cond for c in ("CIC", "ICC")):
        plot_order_effect_scatter(
            unc_by_cond["CIC"], unc_by_cond["ICC"], model_key, out_dir=fig_dir,
        )

    # ── Save JSON results ──────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    result_path = out_dir / f"{model_key}_results.json"
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Results saved → %s", result_path)

    return result


def _save_predictions(
    preds: list[PredictionResult],
    unc: list[UncertaintyScores],
    samples: list[HealthContradictSample],
    condition: str,
    model_key: str,
    out_dir: Path,
) -> None:
    """Save per-sample predictions as JSONL (compatible with HC repo format)."""
    d = out_dir / model_key
    d.mkdir(parents=True, exist_ok=True)
    tid = config.CONDITION_TO_TEMPLATE[condition]
    path = d / f"responses_{tid}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for s, p, u in zip(samples, preds, unc):
            entry = {
                "instance_id": s.instance_id,
                "template_id": tid,
                "query_stance": s.answer,
                "model_response": p.predicted_label.upper(),
                "softmax_prob": {"YES": p.p_yes, "NO": p.p_no},
                "logits": {"YES": p.logit_yes, "NO": p.logit_no},
                "logit_margin": u.logit_margin,
                "entropy": u.entropy,
                "confidence": u.confidence,
            }
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


# ── Entry point ────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()
    _maybe_use_local_model_paths(args.model_root, args.models)

    # Load dataset
    logger.info("Loading HealthContradict dataset …")
    samples = load_dataset(args.dataset_path)
    if args.max_samples:
        samples = samples[: args.max_samples]
    logger.info("  %d samples loaded", len(samples))

    # Optionally load pre-built prompts
    prebuilt = None
    if args.use_prebuilt_prompts:
        logger.info("Loading pre-built prompts from %s …", config.PROMPTS_DIR)
        prebuilt = load_all_prompts()
        # Trim to match max_samples if needed
        if args.max_samples:
            sample_ids = {s.instance_id for s in samples}
            for cond in prebuilt:
                prebuilt[cond] = [p for p in prebuilt[cond]
                                  if p.instance_id in sample_ids]
        logger.info("  pre-built prompts loaded for %s", list(prebuilt.keys()))

    out_dir = Path(args.out_dir)
    all_results: dict = {}

    for model_key in args.models:
        if model_key not in config.MODELS:
            logger.error("Unknown model key: %s (available: %s)",
                         model_key, list(config.MODELS.keys()))
            sys.exit(1)
        all_results[model_key] = evaluate_model(
            model_key, samples, args.conditions, prebuilt, out_dir,
            max_input_tokens=args.max_input_tokens,
        )
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Cross-model summary ────────────────────────────────────────────
    summary_path = out_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info("All results saved → %s", summary_path)


if __name__ == "__main__":
    main()
