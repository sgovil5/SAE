import json

import torch

import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.testing_utils as testing_utils
from sae_bench.evals.unlearning.eval_config import UnlearningEvalConfig
from sae_bench.sae_bench_utils.sae_selection_utils import select_saes_multiple_patterns

results_filename = (
    "tests/acceptance/test_data/unlearning/unlearning_expected_results.json"
)


def test_end_to_end_different_seed():
    """Estimated runtime: 5 minutes
    NOTE: Will require bio-forget-corpus.jsonl to be present in the data directory (see unlearning/README.md)
    """
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {device}")

    test_config = UnlearningEvalConfig()

    test_config.retain_thresholds = [0.01]
    test_config.n_features_list = [10]
    test_config.multipliers = [25, 50]

    test_config.dataset_size = 256

    test_config.random_seed = 48
    test_config.model_name = "gemma-2-2b-it"
    tolerance = 0.04
    test_config.llm_dtype = "bfloat16"
    test_config.llm_batch_size = 4

    sae_regex_patterns = [
        r"sae_bench_gemma-2-2b_topk_width-2pow14_date-1109",
    ]
    sae_block_pattern = [
        r"blocks.5.hook_resid_post__trainer_2",
    ]

    selected_saes = select_saes_multiple_patterns(sae_regex_patterns, sae_block_pattern)

    run_results = unlearning.run_eval(
        test_config,
        selected_saes,
        device,
        output_path="evals/unlearning/test_results/",
        force_rerun=True,
        clean_up_artifacts=True,
    )

    with open("test_data.json", "w") as f:
        json.dump(run_results, f, indent=4)

    with open(results_filename) as f:
        expected_results = json.load(f)

    sae_name = "sae_bench_gemma-2-2b_topk_width-2pow14_date-1109_blocks.5.hook_resid_post__trainer_2"

    run_result_metrics = run_results[sae_name]["eval_result_metrics"]

    testing_utils.compare_dicts_within_tolerance(
        run_result_metrics,
        expected_results[sae_name]["eval_result_metrics"],
        tolerance,
        keys_to_compare=["unlearning_score"],
    )
