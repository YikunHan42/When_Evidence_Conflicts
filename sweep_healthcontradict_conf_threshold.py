"""Sweep HealthContradict selective-prediction performance over confidence thresholds.

This script varies the confidence cutoff used to define the within-condition
"confidently wrong" supervision label. It reuses the saved default
HealthContradict conflict-prediction outputs when available, reruns only the
missing threshold values, and writes a compact appendix-ready summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import config

MODE_TO_KEY = {
    "threshold_transfer": "selective_prediction",
    "exact_topk": "selective_prediction_exact_topk",
}

DEFAULT_MODELS = [
    "llama3.1-8b",
    "meditron3-8b",
    "phi4-14b",
    "qwen3-4b",
    "qwen3-8b",
    "qwen3.5-9b",
]
DEFAULT_THRESHOLDS = [0.5, 0.6, 0.7, 0.8]
DEFAULT_CONDITIONS = ["IC", "ICC"]
DEFAULT_COVERAGES = [0.75, 0.5, 0.25]

SUMMARY_JSON_NAME = "hc_conf_threshold_sweep_summary.json"
SUMMARY_CSV_NAME = "hc_conf_threshold_sweep_summary.csv"
SUMMARY_MD_NAME = "hc_conf_threshold_sweep_summary.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _format_float(value: float) -> str:
    return format(value, ".12g")


def _threshold_tag(value: float) -> str:
    return _format_float(value).replace("-", "m").replace(".", "p")


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _sorted_unique_desc(values: list[float]) -> list[float]:
    return sorted({float(v) for v in values}, reverse=True)


def _sorted_unique_asc(values: list[float]) -> list[float]:
    return sorted({float(v) for v in values})


def _result_path(out_dir: Path, model_key: str, threshold: float) -> Path:
    tag = _threshold_tag(threshold)
    return out_dir / f"{model_key}_conflict_prediction_confthr_{tag}.json"


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    tmp_path.replace(path)


def _load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_runner():
    try:
        from run_conflict_prediction import run_conflict_prediction_for_model
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "HealthContradict threshold sweeps require the project runtime "
            "(notably numpy and scikit-learn). Install the project requirements "
            "and rerun this script in that environment."
        ) from exc
    return run_conflict_prediction_for_model


def _can_reuse_root_result(alpha: float, threshold: float, root_result: dict) -> bool:
    root_alpha = float(root_result.get("alpha", 0.5))
    root_threshold = float(
        root_result.get(
            "confidently_wrong_threshold",
            config.HC_CONFIDENTLY_WRONG_THRESHOLD,
        )
    )
    return abs(root_alpha - alpha) < 1e-12 and abs(root_threshold - threshold) < 1e-12


def _extract_rows(
    result: dict,
    model_key: str,
    threshold: float,
    alpha: float,
    source: str,
    source_path: Path,
    conditions: list[str],
    coverage_targets: list[float],
    selective_key: str,
) -> list[dict]:
    rows = []
    for condition in conditions:
        if condition not in result.get("conditions", {}):
            raise KeyError(
                f"Condition '{condition}' missing from result for model={model_key}, "
                f"threshold={threshold}."
            )
        cond_result = result["conditions"][condition]
        within = cond_result["within_condition"]
        if selective_key not in within:
            raise KeyError(
                f"Selective key '{selective_key}' missing for model={model_key}, "
                f"condition={condition}, threshold={threshold}."
            )
        selective = within[selective_key]
        baseline_accuracy = float(cond_result["accuracy"])

        for coverage_target in coverage_targets:
            coverage_key = str(coverage_target)
            if coverage_key not in selective:
                raise KeyError(
                    f"Coverage target '{coverage_key}' missing for model={model_key}, "
                    f"condition={condition}, threshold={threshold}."
                )
            metrics = selective[coverage_key]
            rows.append(
                {
                    "model": model_key,
                    "alpha": float(alpha),
                    "confidently_wrong_threshold": float(threshold),
                    "condition": condition,
                    "coverage_target": float(coverage_target),
                    "baseline_accuracy": baseline_accuracy,
                    "combined_accuracy": float(metrics["test_accuracy_combined"]),
                    "confidence_accuracy": float(metrics["test_accuracy_conf_only"]),
                    "combined_coverage": float(metrics["test_coverage_combined"]),
                    "confidence_coverage": float(metrics["test_coverage_conf_only"]),
                    "combined_gain_vs_baseline_pp": (
                        float(metrics["test_accuracy_combined"]) - baseline_accuracy
                    ) * 100.0,
                    "confidence_gain_vs_baseline_pp": (
                        float(metrics["test_accuracy_conf_only"]) - baseline_accuracy
                    ) * 100.0,
                    "lift_over_confidence_pp": (
                        float(metrics["test_accuracy_combined"])
                        - float(metrics["test_accuracy_conf_only"])
                    ) * 100.0,
                    "source": source,
                    "source_path": str(source_path),
                }
            )
    return rows


def _mean_rows(
    rows: list[dict],
    thresholds: list[float],
    conditions: list[str],
    coverage_targets: list[float],
) -> list[dict]:
    summary = []
    for threshold in thresholds:
        for condition in conditions:
            for coverage_target in coverage_targets:
                bucket = [
                    row for row in rows
                    if abs(row["confidently_wrong_threshold"] - threshold) < 1e-12
                    and row["condition"] == condition
                    and abs(row["coverage_target"] - coverage_target) < 1e-12
                ]
                summary.append(
                    {
                        "confidently_wrong_threshold": threshold,
                        "condition": condition,
                        "coverage_target": coverage_target,
                        "n_models": len(bucket),
                        "mean_combined_accuracy": _mean(
                            [row["combined_accuracy"] for row in bucket]
                        ),
                        "mean_confidence_accuracy": _mean(
                            [row["confidence_accuracy"] for row in bucket]
                        ),
                        "mean_combined_gain_vs_baseline_pp": _mean(
                            [row["combined_gain_vs_baseline_pp"] for row in bucket]
                        ),
                        "mean_confidence_gain_vs_baseline_pp": _mean(
                            [row["confidence_gain_vs_baseline_pp"] for row in bucket]
                        ),
                        "mean_lift_over_confidence_pp": _mean(
                            [row["lift_over_confidence_pp"] for row in bucket]
                        ),
                    }
                )
    return summary


def _write_csv(path: Path, rows: list[dict]) -> None:
    fieldnames = [
        "model",
        "alpha",
        "confidently_wrong_threshold",
        "condition",
        "coverage_target",
        "baseline_accuracy",
        "combined_accuracy",
        "confidence_accuracy",
        "combined_coverage",
        "confidence_coverage",
        "combined_gain_vs_baseline_pp",
        "confidence_gain_vs_baseline_pp",
        "lift_over_confidence_pp",
        "source",
        "source_path",
    ]
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    tmp_path.replace(path)


def _write_markdown(
    path: Path,
    mean_rows: list[dict],
    mode: str,
    alpha: float,
    thresholds: list[float],
    conditions: list[str],
    coverage_targets: list[float],
) -> None:
    coverage_targets = _sorted_unique_desc(coverage_targets)
    lines = [
        "# HealthContradict Confidence-Threshold Sweep",
        "",
        f"Protocol: `{mode}`",
        f"Alpha: `{_format_float(alpha)}`",
        "",
        "Each cell reports the mean selective-accuracy lift of CAS over COS, in percentage points, averaged across the six models.",
        "",
    ]

    for condition in conditions:
        lines.append(f"## {condition}")
        lines.append("")
        header = ["threshold"] + [f"lift@{int(round(cov * 100))}" for cov in coverage_targets]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for threshold in thresholds:
            values = []
            for coverage_target in coverage_targets:
                row = next(
                    item for item in mean_rows
                    if abs(item["confidently_wrong_threshold"] - threshold) < 1e-12
                    and item["condition"] == condition
                    and abs(item["coverage_target"] - coverage_target) < 1e-12
                )
                values.append(f"{row['mean_lift_over_confidence_pp']:.2f}")
            lines.append("| " + " | ".join([_format_float(threshold)] + values) + " |")
        lines.append("")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep HealthContradict selective prediction over confidently-wrong thresholds."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys to include in the sweep.",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=DEFAULT_THRESHOLDS,
        help="Confidence thresholds for the confidently-wrong label.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="CAS mixing weight to hold fixed during the threshold sweep.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=DEFAULT_CONDITIONS,
        help="Conditions to include in the summary.",
    )
    parser.add_argument(
        "--coverage-targets",
        nargs="+",
        type=float,
        default=DEFAULT_COVERAGES,
        help="Selective-prediction coverage targets to summarize.",
    )
    parser.add_argument(
        "--mode",
        choices=sorted(MODE_TO_KEY),
        default="threshold_transfer",
        help="Selective-prediction protocol to summarize.",
    )
    parser.add_argument(
        "--reference",
        type=str,
        default="CC",
        help="Reference condition for the cross-condition detector.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(config.RESULTS_DIR / "conf_threshold_sensitivity"),
        help="Directory for threshold-specific JSONs and summaries.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun thresholds even if threshold-specific JSONs already exist.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip per-model figure generation while sweeping.",
    )
    parser.add_argument(
        "--no-reuse-root-default",
        action="store_true",
        help="Do not reuse the existing root-level default JSON when threshold=0.7 and alpha=0.5.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_conflict_prediction_for_model = None

    mode = args.mode
    selective_key = MODE_TO_KEY[mode]
    thresholds = _sorted_unique_asc(args.thresholds)
    coverage_targets = _sorted_unique_desc(args.coverage_targets)
    conditions = list(dict.fromkeys(args.conditions))
    rows = []
    per_run_summary = []

    for threshold in thresholds:
        for model_key in args.models:
            sweep_path = _result_path(out_dir, model_key, threshold)

            if not args.force and sweep_path.exists():
                logger.info(
                    "Using cached sweep result: model=%s threshold=%s",
                    model_key,
                    _format_float(threshold),
                )
                result = _load_json(sweep_path)
                source = "cached_sweep"
                source_path = sweep_path
            else:
                root_result_path = config.RESULTS_DIR / f"{model_key}_conflict_prediction.json"
                can_try_root = (
                    not args.no_reuse_root_default
                    and root_result_path.exists()
                )
                if can_try_root:
                    root_result = _load_json(root_result_path)
                    if _can_reuse_root_result(args.alpha, threshold, root_result):
                        logger.info(
                            "Reusing root result: model=%s threshold=%s",
                            model_key,
                            _format_float(threshold),
                        )
                        result = root_result
                        source = "root_default"
                        source_path = root_result_path
                    else:
                        can_try_root = False

                if not can_try_root:
                    if run_conflict_prediction_for_model is None:
                        run_conflict_prediction_for_model = _load_runner()
                    logger.info(
                        "Running sweep: model=%s threshold=%s",
                        model_key,
                        _format_float(threshold),
                    )
                    result = run_conflict_prediction_for_model(
                        model_key=model_key,
                        conditions=conditions,
                        alpha=args.alpha,
                        reference_condition=args.reference,
                        coverage_targets=coverage_targets,
                        generate_figures=not args.skip_figures,
                        confidently_wrong_threshold=threshold,
                    )
                    _atomic_write_json(sweep_path, result)
                    source = "rerun"
                    source_path = sweep_path

            run_rows = _extract_rows(
                result=result,
                model_key=model_key,
                threshold=threshold,
                alpha=args.alpha,
                source=source,
                source_path=source_path,
                conditions=conditions,
                coverage_targets=coverage_targets,
                selective_key=selective_key,
            )
            rows.extend(run_rows)
            per_run_summary.append(
                {
                    "model": model_key,
                    "alpha": args.alpha,
                    "confidently_wrong_threshold": threshold,
                    "source": source,
                    "source_path": str(source_path),
                }
            )

    mean_rows = _mean_rows(rows, thresholds, conditions, coverage_targets)
    payload = {
        "mode": mode,
        "selective_key": selective_key,
        "reference_condition": args.reference,
        "alpha": float(args.alpha),
        "thresholds": thresholds,
        "models": list(args.models),
        "conditions": conditions,
        "coverage_targets": coverage_targets,
        "runs": per_run_summary,
        "rows": rows,
        "mean_rows": mean_rows,
    }

    _atomic_write_json(out_dir / SUMMARY_JSON_NAME, payload)
    _write_csv(out_dir / SUMMARY_CSV_NAME, rows)
    _write_markdown(
        out_dir / SUMMARY_MD_NAME,
        mean_rows=mean_rows,
        mode=mode,
        alpha=float(args.alpha),
        thresholds=thresholds,
        conditions=conditions,
        coverage_targets=coverage_targets,
    )

    logger.info("Summary saved -> %s", out_dir / SUMMARY_JSON_NAME)
    logger.info("Table saved   -> %s", out_dir / SUMMARY_MD_NAME)


if __name__ == "__main__":
    main()
