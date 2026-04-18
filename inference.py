"""
Model loading and logit-based inference for YES/NO questions.

For instruct models (e.g. Llama-3.1-8B-Instruct):
  1. Wrap the raw prompt with the model's chat template.
  2. Run a single forward pass.
  3. Extract logits at the next-token position for YES/NO tokens.
  4. Softmax over the constrained {YES, NO} answer set.

Supports special model classes (e.g. Llama4ForConditionalGeneration)
and automatic bfloat16 override for models that require it.
"""

from __future__ import annotations

import importlib
import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

import config

logger = logging.getLogger(__name__)


# ── Result container ───────────────────────────────────────────────────
@dataclass
class PredictionResult:
    """Stores raw prediction outputs for one sample."""
    predicted_label: str   # "yes" or "no" (lowercase normalised)
    p_yes: float
    p_no: float
    logit_yes: float
    logit_no: float


# ── Model wrapper ──────────────────────────────────────────────────────
class LogitExtractor:
    """Loads a causal-LM and extracts YES/NO logits.

    Automatically applies the chat template for instruct models.
    """

    def __init__(self, model_key: str, max_input_tokens: int | None = None):
        model_id = config.MODELS[model_key]
        self.model_key = model_key
        self.max_input_tokens = max_input_tokens
        self.chat_template_kwargs = getattr(config, "CHAT_TEMPLATE_KWARGS", {}).get(
            model_key, {}
        )
        logger.info("Loading model %s  (%s) …", model_key, model_id)

        # ── Dtype selection ──────────────────────────────────────────
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        torch_dtype = dtype_map.get(config.DTYPE, torch.float16)

        # Some models (Gemma-3, Llama-4) overflow in float16;
        # force bfloat16 for them.
        if model_key in getattr(config, "BFLOAT16_MODELS", set()):
            torch_dtype = torch.bfloat16
            logger.info("  Overriding dtype to bfloat16 for %s", model_key)

        # ── Tokenizer ────────────────────────────────────────────────
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id, trust_remote_code=True,
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # ── Model loading ────────────────────────────────────────────
        # Some architectures (Llama-4 MoE) register under a dedicated
        # class instead of AutoModelForCausalLM.
        special = getattr(config, "SPECIAL_MODEL_CLASS", {}).get(model_key)
        if special is not None:
            mod_path, cls_name = special
            mod = importlib.import_module(mod_path)
            model_cls = getattr(mod, cls_name)
            logger.info("  Using special model class: %s.%s", mod_path, cls_name)
        else:
            model_cls = AutoModelForCausalLM

        self.model = model_cls.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            device_map=config.DEVICE,
            trust_remote_code=True,
        )
        self.model.eval()

        # Detect whether the tokenizer has a chat template
        self.has_chat_template = (
            hasattr(self.tokenizer, "chat_template")
            and self.tokenizer.chat_template is not None
        )
        logger.info("Chat template available: %s", self.has_chat_template)

        # Resolve token IDs for YES / NO (all casing variants).
        self.yes_ids = self._resolve_token_ids(config.YES_TOKENS)
        self.no_ids = self._resolve_token_ids(config.NO_TOKENS)
        logger.info(
            "Yes token IDs: %s  |  No token IDs: %s",
            self.yes_ids, self.no_ids,
        )

    # ── Token-ID resolution ────────────────────────────────────────────
    def _resolve_token_ids(self, surface_forms: list[str]) -> list[int]:
        """Map surface strings to their token IDs (deduplicated)."""
        ids: set[int] = set()
        for tok_str in surface_forms:
            encoded = self.tokenizer.encode(tok_str, add_special_tokens=False)
            if len(encoded) == 1:
                ids.add(encoded[0])
            else:
                # Some tokenisers split tokens; take the last sub-token.
                ids.add(encoded[-1])
        return sorted(ids)

    # ── Chat-template formatting ─────────────────────────────────────
    def _format_prompt(self, raw_prompt: str) -> str:
        """Wrap the raw prompt with the instruct chat template if available."""
        if self.has_chat_template:
            messages = [{"role": "user", "content": raw_prompt}]
            try:
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    **self.chat_template_kwargs,
                )
            except TypeError:
                if self.chat_template_kwargs:
                    logger.warning(
                        "Tokenizer for %s ignored chat template kwargs %s; "
                        "falling back to default template behavior.",
                        self.model_key,
                        sorted(self.chat_template_kwargs.keys()),
                    )
                return self.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
        return raw_prompt

    # ── Single-sample inference ────────────────────────────────────────
    @torch.no_grad()
    def predict(self, prompt: str) -> PredictionResult:
        """Run one forward pass and return YES/NO probabilities.

        Parameters
        ----------
        prompt : str
            The raw text prompt (matching HealthContradict format).
            Will be wrapped with chat template for instruct models.

        Returns
        -------
        PredictionResult
        """
        formatted = self._format_prompt(prompt)
        inputs = self.tokenizer(
            formatted,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_input_tokens,
        ).to(self.model.device)

        outputs = self.model(**inputs)
        # logits shape: (1, seq_len, vocab_size); take last position
        next_logits = outputs.logits[0, -1, :]  # (vocab_size,)

        # Aggregate logits across casing variants via logsumexp
        logit_yes = torch.logsumexp(next_logits[self.yes_ids], dim=0)
        logit_no = torch.logsumexp(next_logits[self.no_ids], dim=0)

        # Normalise to binary distribution
        logits_pair = torch.stack([logit_yes, logit_no])          # (2,)
        probs = F.softmax(logits_pair, dim=0)                     # (2,)
        p_yes, p_no = probs[0].item(), probs[1].item()

        predicted_label = "yes" if p_yes >= p_no else "no"

        return PredictionResult(
            predicted_label=predicted_label,
            p_yes=p_yes,
            p_no=p_no,
            logit_yes=logit_yes.item(),
            logit_no=logit_no.item(),
        )

    # ── Batch convenience ──────────────────────────────────────────────
    def predict_batch(self, prompts: list[str]) -> list[PredictionResult]:
        """Process a list of raw prompts sequentially."""
        return [self.predict(p) for p in prompts]
