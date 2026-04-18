"""
Prompt templates for each context condition.

Matches the exact format from HealthContradict/scripts/a3_create_prompt.py.
Prompts are raw text (no chat template) — model is asked to answer YES or NO.

These builders are used only when generating prompts from dataset_ready.jsonl.
If the pre-built prompts_N.jsonl files are available they can be used directly
via data_loader.load_prompts().
"""

from __future__ import annotations

from data_loader import HealthContradictSample


# ── Condition-specific prompt builders ─────────────────────────────────

def build_nc_prompt(sample: HealthContradictSample) -> str:
    """Template 1 — NC: No context (parametric knowledge only)."""
    return (
        f"Answer the following question with only YES or NO "
        f"based on your parametric knowledge.\n"
        f"Question: {sample.question} \n"
        f"Answer:"
    )


def build_cc_prompt(sample: HealthContradictSample) -> str:
    """Template 2 — CC: Correct context only."""
    return (
        f"Answer the following question with only YES or NO "
        f"based on the given contextual knowledge.\n"
        f"Question: {sample.question}\n"
        f"Context: {sample.correct_doc}  \n"
        f"Answer:"
    )


def build_ic_prompt(sample: HealthContradictSample) -> str:
    """Template 3 — IC: Incorrect context only."""
    return (
        f"Answer the following question with only YES or NO "
        f"based on the given contextual knowledge.\n"
        f"Question: {sample.question}\n"
        f"Context: {sample.incorrect_doc} \n"
        f"Answer:"
    )


def build_cic_prompt(sample: HealthContradictSample) -> str:
    """Template 4 — CIC: Correct-first, then incorrect."""
    return (
        f"Answer the following question with only YES or NO "
        f"based on the given contextual knowledge.\n"
        f"Question: {sample.question}\n"
        f"Context:\n{sample.correct_doc}\n\n{sample.incorrect_doc} \n"
        f"Answer:"
    )


def build_icc_prompt(sample: HealthContradictSample) -> str:
    """Template 5 — ICC: Incorrect-first, then correct."""
    return (
        f"Answer the following question with only YES or NO "
        f"based on the given contextual knowledge.\n"
        f"Question: {sample.question}\n"
        f"Context:\n{sample.incorrect_doc}\n\n{sample.correct_doc} \n"
        f"Answer:"
    )


# ── Dispatcher ─────────────────────────────────────────────────────────
CONDITION_BUILDERS = {
    "NC": build_nc_prompt,
    "CC": build_cc_prompt,
    "IC": build_ic_prompt,
    "CIC": build_cic_prompt,
    "ICC": build_icc_prompt,
}


def build_prompt(condition: str, sample: HealthContradictSample) -> str:
    """Return the raw prompt for a given condition."""
    return CONDITION_BUILDERS[condition](sample)
