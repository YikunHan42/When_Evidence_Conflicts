"""
Feature extraction for the Conflict-Aware Selective Prediction detector.

Two feature groups:
  1. **Model uncertainty signals** (available at inference time for a single
     context condition): logit_margin, entropy, confidence.
  2. **Document embeddings** (from the context document(s) shown to the model):
     sentence-transformer embeddings of the incorrect document, the question,
     and optionally the correct document (for CIC/ICC conditions).

The features are combined into a single numpy array per instance.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

import config
from data_loader import HealthContradictSample

logger = logging.getLogger(__name__)

# ── Embedding cache ───────────────────────────────────────────────────
_EMBED_CACHE: dict[str, np.ndarray] = {}
_MODEL: SentenceTransformer | None = None

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast, good quality
EMBED_DIM = 384


def _get_embed_model() -> SentenceTransformer:
    global _MODEL
    if _MODEL is None:
        logger.info("Loading sentence-transformer: %s", EMBED_MODEL_NAME)
        _MODEL = SentenceTransformer(EMBED_MODEL_NAME)
    return _MODEL


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """Encode texts to dense vectors, caching results.

    Returns array of shape (len(texts), EMBED_DIM).
    """
    model = _get_embed_model()
    # Check cache
    uncached_idx = [i for i, t in enumerate(texts) if t not in _EMBED_CACHE]
    if uncached_idx:
        uncached_texts = [texts[i] for i in uncached_idx]
        logger.info("  Embedding %d texts ...", len(uncached_texts))
        vecs = model.encode(
            uncached_texts,
            batch_size=batch_size,
            show_progress_bar=len(uncached_texts) > 100,
            normalize_embeddings=True,
        )
        for i, idx in enumerate(uncached_idx):
            _EMBED_CACHE[texts[idx]] = vecs[i]

    return np.array([_EMBED_CACHE[t] for t in texts])


# ── Load model uncertainty signals from response JSONL ────────────────
def load_uncertainty_signals(
    model_key: str,
    template_id: int,
) -> dict[str, np.ndarray]:
    """Load logit_margin, entropy, confidence arrays from a response file.

    Returns dict with keys: logit_margin, entropy, confidence, p_yes, p_no,
    predicted_label, correct (shape (N,) each).
    """
    path = config.RESULTS_DIR / model_key / f"responses_{template_id}.jsonl"
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    rows.sort(key=lambda r: r["instance_id"])

    return {
        "logit_margin": np.array([r["logit_margin"] for r in rows]),
        "entropy": np.array([r["entropy"] for r in rows]),
        "confidence": np.array([r["confidence"] for r in rows]),
        "p_yes": np.array([r["softmax_prob"]["YES"] for r in rows]),
        "p_no": np.array([r["softmax_prob"]["NO"] for r in rows]),
        "predicted_label": np.array([r["model_response"].lower() for r in rows]),
    }


# ── Build feature matrices ────────────────────────────────────────────
def build_features_single_context(
    samples: list[HealthContradictSample],
    model_key: str,
    condition: str,
) -> np.ndarray:
    """Build feature matrix for a single-context condition (NC, CC, IC).

    Features (per instance):
      - 3 uncertainty signals: logit_margin, entropy, confidence
      - 384-dim embedding of the context document (IC→incorrect_doc,
        CC→correct_doc, NC→question only)
      - 384-dim embedding of the question

    Returns: shape (N, 3 + 384 + 384) = (N, 771)
    """
    template_id = config.CONDITION_TO_TEMPLATE[condition]
    unc = load_uncertainty_signals(model_key, template_id)

    # Uncertainty features
    unc_feats = np.column_stack([
        unc["logit_margin"],
        unc["entropy"],
        unc["confidence"],
    ])

    # Document embeddings
    if condition == "IC":
        docs = [s.incorrect_doc for s in samples]
    elif condition == "CC":
        docs = [s.correct_doc for s in samples]
    else:  # NC — no document, use empty string
        docs = [""] * len(samples)

    questions = [s.question for s in samples]

    doc_embs = embed_texts(docs)
    q_embs = embed_texts(questions)

    return np.hstack([unc_feats, doc_embs, q_embs])


def build_features_conflict_context(
    samples: list[HealthContradictSample],
    model_key: str,
    condition: str,
) -> np.ndarray:
    """Build feature matrix for a conflict condition (CIC or ICC).

    Features (per instance):
      - 3 uncertainty signals from this condition
      - 384-dim embedding of correct doc
      - 384-dim embedding of incorrect doc
      - 384-dim embedding of question

    Returns: shape (N, 3 + 384*3) = (N, 1155)
    """
    template_id = config.CONDITION_TO_TEMPLATE[condition]
    unc = load_uncertainty_signals(model_key, template_id)

    unc_feats = np.column_stack([
        unc["logit_margin"],
        unc["entropy"],
        unc["confidence"],
    ])

    correct_docs = [s.correct_doc for s in samples]
    incorrect_docs = [s.incorrect_doc for s in samples]
    questions = [s.question for s in samples]

    correct_embs = embed_texts(correct_docs)
    incorrect_embs = embed_texts(incorrect_docs)
    q_embs = embed_texts(questions)

    return np.hstack([unc_feats, correct_embs, incorrect_embs, q_embs])


def build_features(
    samples: list[HealthContradictSample],
    model_key: str,
    condition: str,
) -> np.ndarray:
    """Build features for any condition (dispatches to appropriate builder)."""
    if condition in ("CIC", "ICC"):
        return build_features_conflict_context(samples, model_key, condition)
    else:
        return build_features_single_context(samples, model_key, condition)


# ── Build supervision labels ──────────────────────────────────────────
def build_labels(
    samples: list[HealthContradictSample],
    model_key: str,
    condition: str,
    confidently_wrong_threshold: float = config.HC_CONFIDENTLY_WRONG_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Build supervision labels for the conflict detector.

    Returns dict with:
      - 'confidently_wrong': binary, 1 if model is wrong AND confidence exceeds
        the configured threshold
      - 'is_wrong': binary, 1 if model's prediction != ground truth
      - 'confidence': model confidence under this condition
      - 'correct': binary, 1 if correct
    """
    template_id = config.CONDITION_TO_TEMPLATE[condition]
    unc = load_uncertainty_signals(model_key, template_id)

    ground_truths = np.array([s.answer for s in samples])
    is_correct = (unc["predicted_label"] == ground_truths).astype(int)
    is_wrong = 1 - is_correct
    confidently_wrong = (
        (is_wrong == 1) & (unc["confidence"] > confidently_wrong_threshold)
    ).astype(int)

    return {
        "confidently_wrong": confidently_wrong,
        "is_wrong": is_wrong,
        "correct": is_correct,
        "confidence": unc["confidence"],
    }


# ── Cross-condition supervision labels ───────────────────────────────
def build_cross_condition_labels(
    samples: list[HealthContradictSample],
    model_key: str,
    target_condition: str,
    reference_condition: str = "CC",
    delta_threshold: float = config.HC_CROSS_CONDITION_DELTA_THRESHOLD,
    confidently_wrong_threshold: float = config.HC_CONFIDENTLY_WRONG_THRESHOLD,
) -> dict[str, np.ndarray]:
    """Build supervision labels using cross-condition uncertainty shift.

    The core idea from the original proposal: for each instance, compare the
    model's confidence under the *target* condition (e.g. IC) vs. a
    *reference* condition (e.g. CC or CIC).  A large drop in confidence
    indicates that the target condition's context is misleading the model —
    i.e., conflicting evidence *would* change the model's mind.

    Supervision signal:
      delta = confidence_reference - confidence_target
      conflict_susceptible = 1 if delta > threshold  (model would become
                             less confident with better evidence)

    For IC as target:  reference=CC makes sense (correct context baseline).
    For CIC/ICC as target: reference=CC works (no-conflict baseline).
    For NC as target: reference=CC captures "model lacks context".

    Parameters
    ----------
    target_condition : the condition being evaluated (IC, CIC, ICC, NC)
    reference_condition : the condition to compare against (default: CC)
    delta_threshold : how large the confidence shift must be to count

    Returns dict with:
      - 'conflict_susceptible': binary, the cross-condition label
      - 'confidence_delta': continuous, reference_conf - target_conf
      - 'is_wrong': binary, wrong under target condition
      - 'confidence': model confidence under target condition
      - 'confidence_reference': model confidence under reference condition
      - 'prediction_flips': binary, model changes answer between conditions
    """
    target_tid = config.CONDITION_TO_TEMPLATE[target_condition]
    ref_tid = config.CONDITION_TO_TEMPLATE[reference_condition]

    unc_target = load_uncertainty_signals(model_key, target_tid)
    unc_ref = load_uncertainty_signals(model_key, ref_tid)

    ground_truths = np.array([s.answer for s in samples])
    is_correct_target = (unc_target["predicted_label"] == ground_truths).astype(int)
    is_wrong = 1 - is_correct_target

    # Cross-condition signals
    confidence_delta = unc_ref["confidence"] - unc_target["confidence"]
    prediction_flips = (unc_target["predicted_label"] != unc_ref["predicted_label"]).astype(int)

    # The supervision label: would the model benefit from seeing the reference
    # evidence?  Operationalised as: confidence drops substantially under the
    # target condition compared to the reference.
    conflict_susceptible = (confidence_delta > delta_threshold).astype(int)

    # Also provide a stricter label: prediction actually flips AND confidence
    # was high under target (confidently susceptible to conflict)
    confident_and_flips = (
        (prediction_flips == 1)
        & (unc_target["confidence"] > confidently_wrong_threshold)
    ).astype(int)

    return {
        "conflict_susceptible": conflict_susceptible,
        "confident_and_flips": confident_and_flips,
        "confidence_delta": confidence_delta,
        "prediction_flips": prediction_flips,
        "is_wrong": is_wrong,
        "correct": is_correct_target,
        "confidence": unc_target["confidence"],
        "confidence_reference": unc_ref["confidence"],
        "logit_margin_delta": unc_ref["logit_margin"] - unc_target["logit_margin"],
    }
