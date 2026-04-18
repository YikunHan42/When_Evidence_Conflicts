# When Evidence Conflicts: Uncertainty and Order Effects in Retrieval-Augmented Biomedical Question Answering

This repo integrates the code and cached outputs needed for the paper's `HealthContradict` results and visualizations.

## Coverage

- Core evaluation code for the five `HealthContradict` conditions: `NC`, `CC`, `IC`, `CIC`, `ICC`
- Conflict-aware selective prediction pipeline
- Alpha-sensitivity and confidence-threshold-sensitivity sweeps
- Plotting scripts for the main and appendix selective-prediction figures
- `HealthContradict-main/dataset/` with `dataset_ready.jsonl` and prompt files
- Cached outputs under `results/`
- Generated figures under `figures/`

## Main Files

- `main.py`: main `HealthContradict` evaluation pipeline
- `run_conflict_prediction.py`: conflict-aware selective prediction from saved predictions
- `sweep_healthcontradict_alpha.py`: alpha sensitivity sweep
- `sweep_healthcontradict_conf_threshold.py`: confidently-wrong threshold sensitivity sweep
- `plot_selective_prediction_summary.py`: selective-prediction summary figures
- `plot_confidence_only_gain_summary.py`: confidence-only gain figure
- `plot_alpha_sensitivity_summary.py`: alpha-sensitivity figure
- `plot_conf_threshold_sensitivity_summary.py`: threshold-sensitivity figure

## Supporting Modules

- `config.py`
- `data_loader.py`
- `prompts.py`
- `inference.py`
- `uncertainty.py`
- `calibration.py`
- `analysis.py`
- `selective_prediction.py`
- `conflict_features.py`
- `conflict_detector.py`
- `visualization.py`
- `visualization_conflict.py`

## Expected Layout

```text
paper_repo_hc_v1/
|-- HealthContradict-main/
|   `-- dataset/
|-- results/
|-- figures/
|-- main.py
|-- run_conflict_prediction.py
|-- sweep_healthcontradict_alpha.py
|-- sweep_healthcontradict_conf_threshold.py
|-- plot_*.py
`-- supporting modules
```

## Quick Start

Install dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Run the main evaluation:

```bash
python main.py --models llama3.1-8b --use-prebuilt-prompts
```

Run conflict-aware selective prediction:

```bash
python run_conflict_prediction.py --models llama3.1-8b
```

Run the appendix sweeps:

```bash
python sweep_healthcontradict_alpha.py
python sweep_healthcontradict_conf_threshold.py
```

Regenerate summary figures:

```bash
python plot_confidence_only_gain_summary.py
python plot_selective_prediction_summary.py
python plot_alpha_sensitivity_summary.py
python plot_conf_threshold_sensitivity_summary.py
```

## Notes

- This snapshot assumes the same relative folder structure as copied here.
- Model checkpoints are not included.
- `config.py` still defaults to `DEVICE = "cuda"`. Change it if you want CPU execution.
- The `results/` and `figures/` folders already contain cached outputs from the current workspace.
