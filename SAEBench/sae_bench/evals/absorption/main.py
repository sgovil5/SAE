import argparse
import gc
import os
import statistics
import time
from dataclasses import asdict
from datetime import datetime

import pandas as pd
import torch
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer

from sae_bench.evals.absorption.eval_config import AbsorptionEvalConfig
from sae_bench.evals.absorption.eval_output import (
    EVAL_TYPE_ID_ABSORPTION,
    AbsorptionEvalOutput,
    AbsorptionMeanMetrics,
    AbsorptionMetricCategories,
    AbsorptionResultDetail,
)
from sae_bench.evals.absorption.feature_absorption import (
    run_feature_absortion_experiment,
)
from sae_bench.evals.absorption.k_sparse_probing import run_k_sparse_probing_experiment
from sae_bench.sae_bench_utils import (
    activation_collection,
    general_utils,
    get_eval_uuid,
    get_sae_bench_version,
    get_sae_lens_version,
)
from sae_bench.sae_bench_utils.sae_selection_utils import get_saes_from_regex


def run_eval(
    config: AbsorptionEvalConfig,
    selected_saes: list[tuple[str, SAE]] | list[tuple[str, str]],
    device: str,
    output_path: str,
    force_rerun: bool = False,
):
    torch.set_grad_enabled(True)
    """
    selected_saes is a list of either tuples of (sae_lens release, sae_lens id) or (sae_name, SAE object)
    """

    if "gemma" not in config.model_name:
        print(
            "\n\n\nWARNING: We recommend running this eval on LLMS >= 2B parameters\n\n\n"
        )

    eval_instance_id = get_eval_uuid()
    sae_lens_version = get_sae_lens_version()
    sae_bench_commit_hash = get_sae_bench_version()

    results_dict = {}

    llm_dtype = general_utils.str_to_dtype(config.llm_dtype)

    model = HookedTransformer.from_pretrained_no_processing(
        config.model_name, device=device, dtype=llm_dtype
    )

    for sae_release, sae_object_or_id in tqdm(
        selected_saes, desc="Running SAE evaluation on all selected SAEs"
    ):
        sae_id, sae, sparsity = general_utils.load_and_format_sae(
            sae_release, sae_object_or_id, device
        )  # type: ignore
        sae = sae.to(device=device, dtype=llm_dtype)

        sae_result_path = general_utils.get_results_filepath(
            output_path, sae_release, sae_id
        )

        if os.path.exists(sae_result_path) and not force_rerun:
            print(f"Skipping {sae_release}_{sae_id} as results already exist")
            continue

        k_sparse_probing_results = run_k_sparse_probing_experiment(
            model=model,
            sae=sae,
            layer=sae.cfg.hook_layer,
            sae_name=f"{sae_release}_{sae_id}",
            force=force_rerun,
            max_k_value=config.max_k_value,
            f1_jump_threshold=config.f1_jump_threshold,
            prompt_template=config.prompt_template,
            prompt_token_pos=config.prompt_token_pos,
            device=device,
            k_sparse_probe_l1_decay=config.k_sparse_probe_l1_decay,
            k_sparse_probe_batch_size=config.k_sparse_probe_batch_size,
            k_sparse_probe_num_epochs=config.k_sparse_probe_num_epochs,
        )

        # Save k_sparse_probing_results as a separate JSON
        artifacts_folder = os.path.join("artifacts", "absorption")
        os.makedirs(artifacts_folder, exist_ok=True)
        k_sparse_probing_file = f"{sae_release}_{sae_id}_k_sparse_probing.json"
        k_sparse_probing_file = k_sparse_probing_file.replace("/", "_")
        k_sparse_probing_path = os.path.join(artifacts_folder, k_sparse_probing_file)
        os.makedirs(os.path.dirname(k_sparse_probing_path), exist_ok=True)
        k_sparse_probing_results.to_json(
            k_sparse_probing_path, orient="records", indent=4
        )

        raw_df = run_feature_absortion_experiment(
            model=model,
            sae=sae,
            layer=sae.cfg.hook_layer,
            sae_name=f"{sae_release}_{sae_id}",
            force=force_rerun,
            max_k_value=config.max_k_value,
            feature_split_f1_jump_threshold=config.f1_jump_threshold,
            prompt_template=config.prompt_template,
            prompt_token_pos=config.prompt_token_pos,
            batch_size=config.llm_batch_size,
            device=device,
        )
        agg_df = _aggregate_results_df(raw_df)

        # aggregate results and produce the output
        mean_absorption_fractions = []
        full_absorption_rates = []
        num_split_features = []
        eval_result_details = []
        for _, row in agg_df.iterrows():
            letter = row["letter"]
            mean_absorption_fractions.append(row["mean_absorption_fraction"])
            full_absorption_rates.append(row["full_absorption_rate"])
            num_split_features.append(row["num_split_feats"])
            eval_result_details.append(
                AbsorptionResultDetail(
                    first_letter=letter,  # type: ignore
                    mean_absorption_fraction=row["mean_absorption_fraction"],  # type: ignore
                    full_absorption_rate=row["full_absorption_rate"],  # type: ignore
                    num_full_absorption=row["num_full_absorption"],  # type: ignore
                    num_probe_true_positives=row["num_probe_true_positives"],  # type: ignore
                    num_split_features=row["num_split_feats"],  # type: ignore
                )
            )

        eval_output = AbsorptionEvalOutput(
            eval_type_id=EVAL_TYPE_ID_ABSORPTION,
            eval_config=config,
            eval_id=eval_instance_id,
            datetime_epoch_millis=int(datetime.now().timestamp() * 1000),
            eval_result_metrics=AbsorptionMetricCategories(
                mean=AbsorptionMeanMetrics(
                    mean_absorption_fraction_score=statistics.mean(
                        mean_absorption_fractions
                    ),
                    mean_full_absorption_score=statistics.mean(full_absorption_rates),
                    mean_num_split_features=statistics.mean(num_split_features),
                    std_dev_absorption_fraction_score=statistics.stdev(
                        mean_absorption_fractions
                    ),
                    std_dev_full_absorption_score=statistics.stdev(
                        full_absorption_rates
                    ),
                    std_dev_num_split_features=statistics.stdev(num_split_features),
                )
            ),
            eval_result_details=eval_result_details,
            sae_bench_commit_hash=sae_bench_commit_hash,
            sae_lens_id=sae_id,
            sae_lens_release_id=sae_release,
            sae_lens_version=sae_lens_version,
            sae_cfg_dict=asdict(sae.cfg),
        )

        results_dict[f"{sae_release}_{sae_id}"] = asdict(eval_output)

        eval_output.to_json_file(sae_result_path, indent=2)

        gc.collect()
        torch.cuda.empty_cache()

    return results_dict


def _aggregate_results_df(
    df: pd.DataFrame,
) -> pd.DataFrame:
    agg_df = (
        df[["letter", "absorption_fraction", "is_full_absorption"]]
        .groupby(["letter"])
        .sum()
        .reset_index()
        .merge(
            df[["letter", "num_probe_true_positives", "split_feats"]]
            .groupby(["letter"])
            .agg(
                {
                    "num_probe_true_positives": "mean",
                    "split_feats": lambda x: x.iloc[
                        0
                    ],  # Take the first split_feats list for each letter
                }
            )
            .reset_index()
        )
    )
    agg_df["num_split_feats"] = agg_df["split_feats"].apply(len)
    agg_df["mean_absorption_fraction"] = (
        agg_df["absorption_fraction"] / agg_df["num_probe_true_positives"]
    )
    agg_df["num_full_absorption"] = agg_df["is_full_absorption"]
    agg_df["full_absorption_rate"] = (
        agg_df["num_full_absorption"] / agg_df["num_probe_true_positives"]
    )
    return agg_df


def arg_parser():
    default_config = AbsorptionEvalConfig()

    parser = argparse.ArgumentParser(description="Run absorption evaluation")
    parser.add_argument(
        "--random_seed",
        type=int,
        default=default_config.random_seed,
        help="Random seed",
    )
    parser.add_argument("--model_name", type=str, required=True, help="Model name")
    parser.add_argument(
        "--f1_jump_threshold",
        type=float,
        default=default_config.f1_jump_threshold,
        help="F1 jump threshold",
    )
    parser.add_argument(
        "--max_k_value",
        type=int,
        default=default_config.max_k_value,
        help="Maximum k value",
    )
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=default_config.prompt_template,
        help="Prompt template",
    )
    parser.add_argument(
        "--prompt_token_pos",
        type=int,
        default=default_config.prompt_token_pos,
        help="Prompt token position",
    )
    parser.add_argument(
        "--sae_regex_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE selection",
    )
    parser.add_argument(
        "--sae_block_pattern",
        type=str,
        required=True,
        help="Regex pattern for SAE block selection",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="eval_results/absorption",
        help="Output folder",
    )
    parser.add_argument(
        "--llm_batch_size",
        type=int,
        default=None,
        help="Batch size for LLM. If None, will be populated using LLM_NAME_TO_BATCH_SIZE",
    )
    parser.add_argument(
        "--llm_dtype",
        type=str,
        default=None,
        choices=[None, "float32", "float64", "float16", "bfloat16"],
        help="Data type for LLM. If None, will be populated using LLM_NAME_TO_DTYPE",
    )
    parser.add_argument(
        "--k_sparse_probe_l1_decay",
        type=float,
        default=default_config.k_sparse_probe_l1_decay,
        help="L1 decay for k-sparse probes.",
    )
    parser.add_argument(
        "--k_sparse_probe_batch_size",
        type=float,
        default=default_config.k_sparse_probe_batch_size,
        help="L1 decay for k-sparse probes.",
    )

    parser.add_argument(
        "--force_rerun", action="store_true", help="Force rerun of experiments"
    )

    return parser


def create_config_and_selected_saes(
    args,
) -> tuple[AbsorptionEvalConfig, list[tuple[str, str]]]:
    config = AbsorptionEvalConfig(
        random_seed=args.random_seed,
        f1_jump_threshold=args.f1_jump_threshold,
        max_k_value=args.max_k_value,
        prompt_template=args.prompt_template,
        prompt_token_pos=args.prompt_token_pos,
        model_name=args.model_name,
        k_sparse_probe_l1_decay=args.k_sparse_probe_l1_decay,
        k_sparse_probe_batch_size=args.k_sparse_probe_batch_size,
    )

    if args.llm_batch_size is not None:
        config.llm_batch_size = args.llm_batch_size
    else:
        config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[
            config.model_name
        ]

    if args.llm_dtype is not None:
        config.llm_dtype = args.llm_dtype
    else:
        config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

    if args.random_seed is not None:
        config.random_seed = args.random_seed

    selected_saes = get_saes_from_regex(args.sae_regex_pattern, args.sae_block_pattern)
    assert len(selected_saes) > 0, "No SAEs selected"

    releases = set([release for release, _ in selected_saes])

    print(f"Selected SAEs from releases: {releases}")

    for release, sae in selected_saes:
        print(f"Sample SAEs: {release}, {sae}")

    return config, selected_saes


if __name__ == "__main__":
    """
    python evals/absorption/main.py \
    --sae_regex_pattern "sae_bench_pythia70m_sweep_standard_ctx128_0712" \
    --sae_block_pattern "blocks.4.hook_resid_post__trainer_10" \
    --model_name pythia-70m-deduped
    """
    args = arg_parser().parse_args()

    device = general_utils.setup_environment()

    start_time = time.time()

    config, selected_saes = create_config_and_selected_saes(args)
    # create output folder
    os.makedirs(args.output_folder, exist_ok=True)

    # run the evaluation on all selected SAEs
    results_dict = run_eval(
        config, selected_saes, device, args.output_folder, args.force_rerun
    )

    end_time = time.time()

    print(f"Finished evaluation in {end_time - start_time:.2f} seconds")


# Use this code snippet to use custom SAE objects
# if __name__ == "__main__":
#     import sae_bench.custom_saes.identity_sae as identity_sae
#     import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae

#     """
#     python evals/absorption/main.py
#     """
#     device = general_utils.setup_environment()

#     start_time = time.time()

#     random_seed = 42
#     output_folder = "eval_results/absorption"

#     model_name = "gemma-2-2b"
#     hook_layer = 20

#     repo_id = "google/gemma-scope-2b-pt-res"
#     filename = f"layer_{hook_layer}/width_16k/average_l0_71/params.npz"
#     sae = jumprelu_sae.load_jumprelu_sae(repo_id, filename, hook_layer)
#     selected_saes = [(f"{repo_id}_{filename}_gemmascope_sae", sae)]

#     config = AbsorptionEvalConfig(
#         random_seed=random_seed,
#         model_name=model_name,
#     )

#     config.llm_batch_size = activation_collection.LLM_NAME_TO_BATCH_SIZE[config.model_name]
#     config.llm_dtype = activation_collection.LLM_NAME_TO_DTYPE[config.model_name]

#     # create output folder
#     os.makedirs(output_folder, exist_ok=True)

#     # run the evaluation on all selected SAEs
#     results_dict = run_eval(
#         config,
#         selected_saes,
#         device,
#         output_folder,
#         force_rerun=True,
#     )

#     end_time = time.time()

#     print(f"Finished evaluation in {end_time - start_time} seconds")
