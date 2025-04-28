# WTA SAE Training and Evaluation with dictionary_learning & SAEBench

Train Winner-Take-All (WTA) Sparse Autoencoders (SAEs) using `dictionary_learning` and evaluate them with `SAEBench`.

## 1. Training WTA Autoencoders

Uses `dictionary_learning/train_wta.py` to train WTA SAEs, ablating over specified sparsity rates.

### Configuration (`dictionary_learning/train_wta.py`)

Adjust key parameters like:
*   `MODEL_NAME`, `LAYER`, `DATASET_NAME`
*   `sparsity_rates`: List of rates for ablation.
*   `SAVE_DIR`: Base output directory (creates subdirs per rate).
*   Buffer/batch sizes, total tokens, expansion factor.
*   `trainer_cfg` for `WTATrainer` settings.

### Running Training (`dictionary_learning/train_sae.sh`)

1.  Configure SLURM directives and conda environment in the `.sh` script.
2.  Submit:
    ```bash
    sbatch dictionary_learning/train_sae.sh
    ```
    This runs `train_wta.py`, saving checkpoints (`ae.pt`) and configs (`config.json`) into subdirectories of `SAVE_DIR`.

## 2. Running Evaluations

Uses `SAEBench/sae_bench/custom_saes/run_all_evals_dictionary_learning_saes.py` to evaluate locally saved SAEs.

### Configuration (`run_all_evals_dictionary_learning_saes.py`)

*   Update `MODEL_CONFIGS` for the base LLM used.
*   Set `model_name_to_eval`.
*   **Set `local_sae_parent_dir` to the main directory containing your trained SAE folders (e.g., `~/scratch/SAE/dictionary_learning`).**
*   Choose `eval_types` (e.g., `"core"`, `"scr"`, `"tpp"`, `"sparse_probing"`).
*   Optionally use `exclude_keywords`, `include_keywords` to filter SAE folders.

### Running Evaluations (`SAEBench/sae_bench/custom_saes/run_evals.sh`)

1.  Configure SLURM directives and conda environment in the `.sh` script.
2.  Submit:
    ```bash
    sbatch SAEBench/sae_bench/custom_saes/run_evals.sh
    ```
    This runs `run_all_evals...py`, which finds SAEs in `local_sae_parent_dir`, loads them (handles `WTATrainer` state dicts), runs the selected evals, and saves results to `SAEBench/sae_bench/custom_saes/eval_results/`.