"""Sweep HealthContradict selective-prediction performance over alpha.

This script reuses the saved default HealthContradict conflict-prediction
outputs for ``alpha=0.5`` when available, reruns only the missing alpha
values, and writes a compact appendix-ready summary. It focuses on the two
hardest conditions by default (``IC`` and ``ICC``) and reports the selective
accuracy lift of the combined conflict-aware score over the confidence-only
baseline at the target coverages used in the paper.
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
DEFAULT_ALPHAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DEFAULT_CONDITIONS = ["IC", "ICC"]
DEFAULT_COVERAGES = [0.75, 0.5, 0.25]

SUMMARY_JSON_NAME = "hc_alpha_sweep_summary.json"
SUMMARY_CSV_NAME = "hc_alpha_sweep_summary.csv"
SUMMARY_MD_NAME = "hc_alpha_sweep_summary.md"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)


def _format_float(value: float) -> str:
    return format(value, ".12g")


def _alpha_tag(value: float) -> str:
    return _format_float(value).replace("-", "m").replace(".", "p")


def _mean(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def _sorted_unique_desc(values: list[float]) -> list[float]:
    return sorted({float(v) for v in values}, reverse=True)


def _result_path(out_dir: Path, model_key: str, alpha: float) -> Path:
    tag = _alpha_tag(alpha)
    return out_dir / f"{model_key}_conflict_prediction_alpha_{tag}.json"


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
            "HealthContradict alpha sweeps require the project runtime "
            "(notably numpy and scikit-learn). Install the project requirements "
            "and rerun this script in that environment."
        ) from exc
    return run_conflict_prediction_for_model


def _can_reuse_root_result(alpha: float, root_result: dict) -> bool:
    return abs(float(root_result.get("alpha", 0.5)) - alpha) < 1e-12


def _extract_rows(
    result: dict,
    model_key: str,
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
                f"alpha={alpha}."
            )
        cond_result = result["conditions"][condition]
        within = cond_result["within_condition"]
        if selective_key not in within:
            raise KeyError(
                f"Selective key '{selective_key}' missing for model={model_key}, "
                f"condition={condition}, alpha={alpha}."
            )
        selective = within[selective_key]
        baseline_accuracy = float(cond_result["accuracy"])

        for coverage_target in coverage_targets:
            coverage_key = str(coverage_target)
            if coverage_key not in selective:
                raise KeyError(
                    f"Coverage target '{coverage_key}' missing for model={model_key}, "
                    f"condition={condition}, alpha={alpha}."
                )
            metrics = selective[coverage_key]
            rows.append(
                {
                    "model": model_key,
                    "alpha": float(alpha),
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
    alphas: list[float],
    conditions: list[str],
    coverage_targets: list[float],
) -> list[dict]:
    summary = []
    for alpha in alphas:
        for condition in conditions:
            for coverage_target in coverage_targets:
                bucket = [
                    row for row in rows
                    if abs(row["alpha"] - alpha) < 1e-12
                    and row["condition"] == condition
                    and abs(row["coverage_target"] - coverage_target) < 1e-12
                ]
                summary.append(
                    {
                        "alpha": alpha,
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
    alphas: list[float],
    conditions: list[str],
    coverage_targets: list[float],
) -> None:
    coverage_targets = _sorted_unique_desc(coverage_targets)
    lines = [
        "# HealthContradict Alpha Sweep",
        "",
        f"Protocol: `{mode}`",
        "",
        "Each cell reports the mean selective-accuracy lift of the combined score over the confidence-only baseline, in percentage points, averaged across the six models.",
        "",
    ]

    for condition in conditions:
        lines.append(f"## {condition}")
        lines.append("")
        header = ["alpha"] + [f"lift@{int(round(cov * 100))}" for cov in coverage_targets]
        lines.append("| " + " | ".join(header) + " |")
        lines.append("| " + " | ".join(["---"] * len(header)) + " |")
        for alpha in alphas:
            values = []
            for coverage_target in coverage_targets:
                row = next(
                    item for item in mean_rows
                    if abs(item["alpha"] - alpha) < 1e-12
                    and item["condition"] == condition
                    and abs(item["coverage_target"] - coverage_target) < 1e-12
                )
                values.append(f"{row['mean_lift_over_confidence_pp']:.2f}")
            lines.append("| " + " | ".join([_format_float(alpha)] + values) + " |")
        lines.append("")

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    tmp_path.replace(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sweep HealthContradict selective prediction over alpha."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=DEFAULT_MODELS,
        help="Model keys to include in the sweep.",
    )
    parser.add_argument(
        "--alphas",
        nargs="+",
        type=float,
        default=DEFAULT_ALPHAS,
        help="Alpha values to evaluate.",
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
        default=str(config.RESULTS_DIR / "alpha_sensitivity"),
        help="Directory for alpha-specific JSONs and summaries.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rerun alpha values even if alpha-specific JSONs already exist.",
    )
    parser.add_argument(
        "--skip-figures",
        action="store_true",
        help="Skip per-model figure generation while sweeping.",
    )
    parser.add_argument(
        "--no-reuse-root-alpha05",
        action="store_true",
        help="Do not reuse the existing root-level alpha=0.5 JSON.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_conflict_prediction_for_model = None

    mode = args.mode
    selective_key = MODE_TO_KEY[mode]
    alphas = sorted({float(alpha) for alpha in args.alphas})
    coverage_targets = _sorted_unique_desc(args.coverage_targets)
    conditions = list(dict.fromkeys(args.conditions))
    rows = []
    per_run_summary = []

    for alpha in alphas:
        for model_key in args.models:
            sweep_path = _result_path(out_dir, model_key, alpha)

            if not args.force and sweep_path.exists():
                logger.info(
                    "Using cached sweep result: model=%s alpha=%s",
                    model_key,
                    _format_float(alpha),
                )
                result = _load_json(sweep_path)
                source = "cached_sweep"
                source_path = sweep_path
            else:
                root_result_path = config.RESULTS_DIR / f"{model_key}_conflict_prediction.json"
                reuse_root_alpha05 = (
                    not args.no_reuse_root_alpha05
                    and abs(alpha - 0.5) < 1e-12
                    and root_result_path.exists()
                )
                if reuse_root_alpha05:
                    root_result = _load_json(root_result_path)
                    if _can_reuse_root_result(alpha, root_result):
                        logger.info(
                            "Reusing root result: model=%s alpha=%s",
                            model_key,
                            _format_float(alpha),
                        )
                        result = root_result
                        source = "root_alpha_0.5"
                        source_path = root_result_path
                    else:
                        reuse_root_alpha05 = False

                if not reuse_root_alpha05:
                    if run_conflict_prediction_for_model is None:
                        run_conflict_prediction_for_model = _load_runner()
                    logger.info(
                        "Running sweep: model=%s alpha=%s",
                        model_key,
                        _format_float(alpha),
                    )
                    result = run_conflict_prediction_for_model(
                        model_key=model_key,
                        conditions=conditions,
                        alpha=alpha,
                        reference_condition=args.reference,
                        coverage_targets=coverage_targets,
                        generate_figures=not args.skip_figures,
                    )
                    _atomic_write_json(sweep_path, result)
                    source = "rerun"
                    source_path = sweep_path

            run_rows = _extract_rows(
                result=result,
                model_key=model_key,
                alpha=alpha,
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
                    "alpha": alpha,
                    "source": source,
                    "source_path": str(source_path),
                }
            )

    mean_rows = _mean_rows(rows, alphas, conditions, coverage_targets)
    payload = {
        "mode": mode,
        "selective_key": selective_key,
        "reference_condition": args.reference,
        "alphas": alphas,
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
        alphas=alphas,
        conditions=conditions,
        coverage_targets=coverage_targets,
    )

    logger.info("Summary saved -> %s", out_dir / SUMMARY_JSON_NAME)
    logger.info("Table saved   -> %s", out_dir / SUMMARY_MD_NAME)


if __name__ == "__main__":
    main()
