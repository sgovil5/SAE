import os

import numpy as np
import torch
from sae_lens import SAE
from transformer_lens import HookedTransformer

from sae_bench.evals.unlearning.eval_config import UnlearningEvalConfig
from sae_bench.evals.unlearning.utils.feature_activation import (
    get_top_features,
    load_sparsity_data,
    save_feature_sparsity,
)
from sae_bench.evals.unlearning.utils.metrics import calculate_metrics_list


def run_metrics_calculation(
    model: HookedTransformer,
    sae: SAE,
    activation_store,
    forget_sparsity: np.ndarray,
    retain_sparsity: np.ndarray,
    artifacts_folder: str,
    sae_name: str,
    config: UnlearningEvalConfig,
    force_rerun: bool,
):
    dataset_names = config.dataset_names

    for retain_threshold in config.retain_thresholds:
        top_features_custom = get_top_features(
            forget_sparsity, retain_sparsity, retain_threshold=retain_threshold
        )

        main_ablate_params = {
            "intervention_method": config.intervention_method,
        }

        n_features_lst = config.n_features_list
        multipliers = config.multipliers

        sweep = {
            "features_to_ablate": [
                np.array(top_features_custom[:n]) for n in n_features_lst
            ],
            "multiplier": multipliers,
        }

        save_metrics_dir = os.path.join(artifacts_folder, sae_name, "results/metrics")

        metrics_lst = calculate_metrics_list(
            model,
            (
                config.llm_batch_size * 2
            ),  # multiple choice questions are shorter, so we can afford a larger batch size
            sae,
            main_ablate_params,
            sweep,
            artifacts_folder,
            force_rerun,
            dataset_names,
            n_batch_loss_added=config.n_batch_loss_added,
            activation_store=activation_store,
            target_metric=config.target_metric,
            save_metrics=config.save_metrics,
            save_metrics_dir=save_metrics_dir,
            retain_threshold=retain_threshold,
        )

    return metrics_lst  # type: ignore


def run_eval_single_sae(
    model: HookedTransformer,
    sae: SAE,
    config: UnlearningEvalConfig,
    artifacts_folder: str,
    sae_release_and_id: str,
    force_rerun: bool,
):
    """sae_release_and_id: str is the name used when saving data for this SAE. This data will be reused at various points in the evaluation."""

    os.makedirs(artifacts_folder, exist_ok=True)

    torch.set_grad_enabled(False)

    # calculate feature sparsity
    save_feature_sparsity(
        model,
        sae,
        artifacts_folder,
        sae_release_and_id,
        config.dataset_size,
        config.seq_len,
        config.llm_batch_size,
    )
    forget_sparsity, retain_sparsity = load_sparsity_data(
        artifacts_folder, sae_release_and_id
    )

    # do intervention and calculate eval metrics
    # activation_store = setup_activation_store(sae, model)
    activation_store = None
    results = run_metrics_calculation(
        model,
        sae,
        activation_store,
        forget_sparsity,
        retain_sparsity,
        artifacts_folder,
        sae_release_and_id,
        config,
        force_rerun,
    )

    return results
