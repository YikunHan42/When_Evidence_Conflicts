"""
Configuration for Uncertainty-Aware Evaluation of Biomedical LLMs.
"""

from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
HC_DIR = PROJECT_ROOT / "HealthContradict-main"
DATASET_JSONL = HC_DIR / "dataset" / "dataset_ready.jsonl"
PROMPTS_DIR = HC_DIR / "dataset"           # prompts_1.jsonl … prompts_5.jsonl
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

# ── Models ─────────────────────────────────────────────────────────────
MODELS = {
    # --- existing ---
    "llama3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",
    "meditron3-8b": "OpenMeditron/Meditron3-8B",
    "qwen3-4b": "Qwen/Qwen3-4B-Instruct-2507",
    # --- newer models ---
    "qwen3-8b": "Qwen/Qwen3-8B",
    "qwen3.5-9b": "Qwen/Qwen3.5-9B",
    # Use the instruction-tuned Gemma 4 checkpoint for chat-style evaluation.
    "gemma4-31b": "google/gemma-4-31B-it",
    "phi4-14b": "microsoft/phi-4",
    "llama4-scout": "meta-llama/Llama-4-Scout-17B-16E-Instruct",
}

# Models that need special loader classes instead of AutoModelForCausalLM.
# Maps model_key -> (module_path, class_name).
SPECIAL_MODEL_CLASS = {
    "llama4-scout": ("transformers", "Llama4ForConditionalGeneration"),
}

# Optional keyword overrides passed into tokenizer.apply_chat_template().
# Qwen3/Qwen3.5 default to reasoning mode; disable thinking so the evaluator
# can score the first direct YES/NO answer token instead of a reasoning prefix.
CHAT_TEMPLATE_KWARGS = {
    "qwen3-8b": {"enable_thinking": False},
    "qwen3.5-9b": {"enable_thinking": False},
}

# Models that require bfloat16 (float16 causes overflow/NaN).
BFLOAT16_MODELS = {"gemma4-31b", "llama4-scout"}

# ── Context conditions ──────────────────────────────────────────────────
# Mapping: condition name → template_id in HealthContradict prompts
CONDITIONS = ["NC", "CC", "IC", "CIC", "ICC"]
CONDITION_TO_TEMPLATE = {
    "NC": 1,    # parametric only
    "CC": 2,    # correct context only
    "IC": 3,    # incorrect context only
    "CIC": 4,   # correct-first, then incorrect
    "ICC": 5,   # incorrect-first, then correct
}
TEMPLATE_TO_CONDITION = {v: k for k, v in CONDITION_TO_TEMPLATE.items()}

# ── Answer tokens ───────────────────────────────────────────────────────
# The HealthContradict repo uses uppercase "YES" / "NO" for token lookup.
# We also include lowercase/title-case variants for robustness.
YES_TOKENS = ["YES", "Yes", "yes"]
NO_TOKENS = ["NO", "No", "no"]

# ── ConflictBank paths ─────────────────────────────────────────────────
CB_DIR = PROJECT_ROOT / "ConflictBank"
CB_DATASET_JSONL = CB_DIR / "dataset" / "cb_subset.jsonl"
CB_RESULTS_DIR = PROJECT_ROOT / "results_cb"
CB_FIGURES_DIR = PROJECT_ROOT / "figures_cb"

# ManConCorpus paths
MANCON_DIR = PROJECT_ROOT / "ManConCorpus"
MANCON_PAIRS_TSV = MANCON_DIR / "selected_pairs_openai_merged_complete.tsv"
MANCON_RESULTS_DIR = PROJECT_ROOT / "results_mancon"
MANCON_FIGURES_DIR = PROJECT_ROOT / "figures_mancon"

# ── ConflictBank answer tokens (4-option MC) ─────────────────────────
MC_OPTIONS = ["A", "B", "C", "D"]
MC_TOKENS = {
    "A": ["A", "a"],
    "B": ["B", "b"],
    "C": ["C", "c"],
    "D": ["D", "d"],
}

# ── ConflictBank thresholds ───────────────────────────────────────────
HC_CONFIDENTLY_WRONG_THRESHOLD = 0.7
HC_CROSS_CONDITION_DELTA_THRESHOLD = 0.3

CB_SUBSAMPLE_SIZE = 1500
CB_CONFIDENTLY_WRONG_THRESHOLD = 0.5   # lower than HC's 0.7 (4-class baseline = 25%)
CB_CROSS_CONDITION_DELTA_THRESHOLD = 0.3

# ManConCorpus thresholds (same binary setup as HealthContradict)
MANCON_CONFIDENTLY_WRONG_THRESHOLD = 0.7
MANCON_CROSS_CONDITION_DELTA_THRESHOLD = 0.3

# ── Calibration ─────────────────────────────────────────────────────────
ECE_NUM_BINS = 10

# ── Selective prediction ────────────────────────────────────────────────
TAU_GRID = [i / 100 for i in range(50, 100)]  # confidence thresholds 0.50 … 0.99

# ── Inference ───────────────────────────────────────────────────────────
BATCH_SIZE = 8
MAX_NEW_TOKENS = 1  # constrained to single Yes/No token
DEVICE = "cuda"     # set to "cpu" if no GPU available
DTYPE = "float16"   # HealthContradict repo uses float16

# ── Reproducibility ─────────────────────────────────────────────────────
SEED = 42
