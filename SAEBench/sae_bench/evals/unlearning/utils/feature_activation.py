import json
import os
import random

import numpy as np
import torch
from datasets import load_dataset
from sae_lens import SAE
from transformer_lens import HookedTransformer

import sae_bench.sae_bench_utils.dataset_utils as dataset_utils
from sae_bench.sae_bench_utils.activation_collection import (
    get_feature_activation_sparsity,
)

FORGET_FILENAME = "feature_sparsity_forget.txt"
RETAIN_FILENAME = "feature_sparsity_retain.txt"

SPARSITIES_DIR = "results/sparsities"


def get_forget_retain_data(
    forget_corpora: str = "bio-forget-corpus",
    retain_corpora: str = "wikitext",
    min_len: int = 50,
    max_len: int = 2000,
    batch_size: int = 4,
) -> tuple[list[str], list[str]]:
    retain_dataset = []
    if retain_corpora == "wikitext":
        raw_retain = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        for x in raw_retain:
            if len(x["text"]) > min_len:  # type: ignore
                retain_dataset.append(str(x["text"]))  # type: ignore
    else:
        raise Exception("Unknown retain corpora")

    forget_dataset = []
    for line in open(f"./sae_bench/evals/unlearning/data/{forget_corpora}.jsonl"):
        if "bio-forget-corpus" in forget_corpora:
            raw_text = json.loads(line)["text"]
        else:
            raw_text = line
        if len(raw_text) > min_len:
            forget_dataset.append(str(raw_text))

    return forget_dataset, retain_dataset


def get_shuffled_forget_retain_tokens(
    model: HookedTransformer,
    forget_corpora: str = "bio-forget-corpus",
    retain_corpora: str = "wikitext",
    batch_size: int = 2048,
    seq_len: int = 1024,
):
    """
    get shuffled forget tokens and retain tokens, with given batch size and sequence length
    note: wikitext has less than 2048 batches with seq_len=1024
    """
    forget_dataset, retain_dataset = get_forget_retain_data(
        forget_corpora, retain_corpora
    )

    print(len(forget_dataset), len(forget_dataset[0]))
    print(len(retain_dataset), len(retain_dataset[0]))

    shuffled_forget_dataset = random.sample(
        forget_dataset, min(batch_size, len(forget_dataset))
    )

    forget_tokens = dataset_utils.tokenize_and_concat_dataset(
        model.tokenizer,  # type: ignore
        shuffled_forget_dataset,
        seq_len=seq_len,
    ).to(model.cfg.device)
    retain_tokens = dataset_utils.tokenize_and_concat_dataset(
        model.tokenizer,  # type: ignore
        retain_dataset,
        seq_len=seq_len,
    ).to(model.cfg.device)

    print(forget_tokens.shape, retain_tokens.shape)
    shuffled_forget_tokens = forget_tokens[torch.randperm(forget_tokens.shape[0])]
    shuffled_retain_tokens = retain_tokens[torch.randperm(retain_tokens.shape[0])]

    return shuffled_forget_tokens[:batch_size], shuffled_retain_tokens[:batch_size]


def gather_residual_activations(model: HookedTransformer, target_layer: int, inputs):
    target_act = None

    def gather_target_act_hook(mod, inputs, outputs):
        nonlocal target_act  # make sure we can modify the target_act from the outer scope
        target_act = outputs[0]
        return outputs

    handle = model.model.layers[target_layer].register_forward_hook(  # type: ignore
        gather_target_act_hook
    )
    _ = model.forward(inputs)  # type: ignore
    handle.remove()
    return target_act


def get_top_features(forget_score, retain_score, retain_threshold=0.01):
    # criteria for selecting features: retain score < 0.01 and then sort by forget score
    high_retain_score_features = np.where(retain_score >= retain_threshold)[0]
    modified_forget_score = forget_score.copy()
    modified_forget_score[high_retain_score_features] = 0
    top_features = modified_forget_score.argsort()[::-1]
    # print(top_features[:20])

    n_non_zero_features = np.count_nonzero(modified_forget_score)
    top_features_non_zero = top_features[:n_non_zero_features]

    return top_features_non_zero


def check_existing_results(artifacts_folder: str, sae_name) -> bool:
    forget_path = os.path.join(
        artifacts_folder, sae_name, SPARSITIES_DIR, FORGET_FILENAME
    )
    retain_path = os.path.join(
        artifacts_folder, sae_name, SPARSITIES_DIR, RETAIN_FILENAME
    )
    return os.path.exists(forget_path) and os.path.exists(retain_path)


def calculate_sparsity(
    model: HookedTransformer, sae: SAE, forget_tokens, retain_tokens, batch_size: int
):
    feature_sparsity_forget = (
        get_feature_activation_sparsity(
            forget_tokens,
            model,
            sae,
            batch_size=batch_size,
            layer=sae.cfg.hook_layer,
            hook_name=sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )
        .cpu()
        .numpy()
    )
    feature_sparsity_retain = (
        get_feature_activation_sparsity(
            retain_tokens,
            model,
            sae,
            batch_size=batch_size,
            layer=sae.cfg.hook_layer,
            hook_name=sae.cfg.hook_name,
            mask_bos_pad_eos_tokens=True,
        )
        .cpu()
        .numpy()
    )
    return feature_sparsity_forget, feature_sparsity_retain


def save_results(
    artifacts_folder: str,
    sae_name: str,
    feature_sparsity_forget,
    feature_sparsity_retain,
):
    output_dir = os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR)
    os.makedirs(output_dir, exist_ok=True)
    np.savetxt(
        os.path.join(output_dir, FORGET_FILENAME), feature_sparsity_forget, fmt="%f"
    )
    np.savetxt(
        os.path.join(output_dir, RETAIN_FILENAME), feature_sparsity_retain, fmt="%f"
    )


def load_sparsity_data(
    artifacts_folder: str, sae_name: str
) -> tuple[np.ndarray, np.ndarray]:
    forget_sparsity = np.loadtxt(
        os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR, FORGET_FILENAME),
        dtype=float,
    )
    retain_sparsity = np.loadtxt(
        os.path.join(artifacts_folder, sae_name, SPARSITIES_DIR, RETAIN_FILENAME),
        dtype=float,
    )
    return forget_sparsity, retain_sparsity


def save_feature_sparsity(
    model: HookedTransformer,
    sae: SAE,
    artifacts_folder: str,
    sae_name: str,
    dataset_size: int,
    seq_len: int,
    batch_size: int,
):
    if check_existing_results(artifacts_folder, sae_name):
        print(f"Sparsity calculation for {sae_name} is already done")
        return

    forget_tokens, retain_tokens = get_shuffled_forget_retain_tokens(
        model, batch_size=dataset_size, seq_len=seq_len
    )

    feature_sparsity_forget, feature_sparsity_retain = calculate_sparsity(
        model, sae, forget_tokens, retain_tokens, batch_size
    )

    save_results(
        artifacts_folder, sae_name, feature_sparsity_forget, feature_sparsity_retain
    )
