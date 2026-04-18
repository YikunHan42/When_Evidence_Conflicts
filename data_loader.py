"""
Load and preprocess the HealthContradict dataset.

Schema of dataset_ready.jsonl (920 instances):
  - instance_id : int
  - topic_id    : int
  - query       : str   (health yes/no question)
  - query_stance: str   ("yes" or "no" — the ground-truth answer)
  - doc_a       : str   (document with YES stance)
  - doc_b       : str   (document with NO stance)

Which document is "correct" depends on query_stance:
  query_stance == "yes" → correct = doc_a, incorrect = doc_b
  query_stance == "no"  → correct = doc_b, incorrect = doc_a

Pre-built prompt files (prompts_1.jsonl … prompts_5.jsonl):
  - instance_id, topic_id, template_id, query, query_stance, prompt
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import config


# ── Data structures ────────────────────────────────────────────────────
@dataclass
class HealthContradictSample:
    """A single HealthContradict instance."""
    instance_id: int
    topic_id: int
    question: str                  # the health question
    answer: str                    # ground-truth: "yes" or "no"
    doc_a: str                     # YES-stance document
    doc_b: str                     # NO-stance document

    @property
    def correct_doc(self) -> str:
        return self.doc_a if self.answer == "yes" else self.doc_b

    @property
    def incorrect_doc(self) -> str:
        return self.doc_b if self.answer == "yes" else self.doc_a


@dataclass
class PromptInstance:
    """A pre-built prompt loaded from prompts_N.jsonl."""
    instance_id: int
    topic_id: int
    template_id: int
    query: str
    query_stance: str
    prompt: str


# ── Loaders ────────────────────────────────────────────────────────────
def load_dataset(path: str | Path | None = None) -> list[HealthContradictSample]:
    """Load dataset_ready.jsonl and return structured samples."""
    path = Path(path) if path else config.DATASET_JSONL
    samples: list[HealthContradictSample] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            samples.append(HealthContradictSample(
                instance_id=obj["instance_id"],
                topic_id=obj["topic_id"],
                question=obj["query"],
                answer=obj["query_stance"].strip().lower(),
                doc_a=obj["doc_a"],
                doc_b=obj["doc_b"],
            ))
    return samples


def load_prompts(template_id: int, prompts_dir: str | Path | None = None) -> list[PromptInstance]:
    """Load a pre-built prompt file (prompts_N.jsonl)."""
    d = Path(prompts_dir) if prompts_dir else config.PROMPTS_DIR
    path = d / f"prompts_{template_id}.jsonl"
    prompts: list[PromptInstance] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            prompts.append(PromptInstance(
                instance_id=obj["instance_id"],
                topic_id=obj["topic_id"],
                template_id=obj["template_id"],
                query=obj["query"],
                query_stance=obj["query_stance"],
                prompt=obj["prompt"],
            ))
    return prompts


def load_all_prompts(
    prompts_dir: str | Path | None = None,
) -> dict[str, list[PromptInstance]]:
    """Load all 5 prompt files, keyed by condition name (NC/CC/IC/CIC/ICC)."""
    result: dict[str, list[PromptInstance]] = {}
    for condition, tid in config.CONDITION_TO_TEMPLATE.items():
        result[condition] = load_prompts(tid, prompts_dir)
    return result
