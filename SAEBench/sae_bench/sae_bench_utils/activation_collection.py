import os
from typing import Any

import einops
import torch
from beartype import beartype
from jaxtyping import Bool, Float, Int, jaxtyped
from sae_lens import SAE
from tqdm import tqdm
from transformer_lens import HookedTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding

# Relevant at ctx len 128
LLM_NAME_TO_BATCH_SIZE = {
    "pythia-70m-deduped": 512,
    "pythia-160m-deduped": 256,
    "gemma-2-2b": 32,
    "gemma-2-9b": 32,
    "gemma-2-2b-it": 32,
    "gemma-2-9b-it": 32,
}

LLM_NAME_TO_DTYPE = {
    "pythia-70m-deduped": "float32",
    "pythia-160m-deduped": "float32",
    "gemma-2-2b": "bfloat16",
    "gemma-2-2b-it": "bfloat16",
    "gemma-2-9b": "bfloat16",
    "gemma-2-9b-it": "bfloat16",
}


def get_module(model: AutoModelForCausalLM, layer_num: int) -> torch.nn.Module:
    if model.config.architectures[0] == "Gemma2ForCausalLM":
        return model.model.layers[layer_num]
    else:
        raise ValueError(
            f"Model {model.config.architectures[0]} not supported, please add the appropriate module"
        )


@torch.no_grad()
def get_layer_activations(
    model: AutoModelForCausalLM,
    target_layer: int,
    inputs: BatchEncoding,
    source_pos_B: torch.Tensor,
) -> torch.Tensor:
    acts_BLD = None

    def gather_target_act_hook(module, inputs, outputs):
        nonlocal acts_BLD
        acts_BLD = outputs[0]
        return outputs

    handle = get_module(model, target_layer).register_forward_hook(
        gather_target_act_hook
    )

    _ = model(
        input_ids=inputs["input_ids"].to(model.device),
        attention_mask=inputs.get("attention_mask", None),
    )

    handle.remove()

    assert acts_BLD is not None

    acts_BD = acts_BLD[list(range(acts_BLD.shape[0])), source_pos_B, :].clone()

    return acts_BD


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_bos_pad_eos_mask(
    tokens: Int[torch.Tensor, "dataset_size seq_len"], tokenizer: AutoTokenizer | Any
) -> Bool[torch.Tensor, "dataset_size seq_len"]:
    mask = (
        (tokens == tokenizer.pad_token_id)  # type: ignore
        | (tokens == tokenizer.eos_token_id)  # type: ignore
        | (tokens == tokenizer.bos_token_id)  # type: ignore
    ).to(dtype=torch.bool)
    return ~mask


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_llm_activations(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
    show_progress: bool = True,
) -> Float[torch.Tensor, "dataset_size seq_len d_model"]:
    """Collects activations for an LLM model from a given layer for a given set of tokens.
    VERY IMPORTANT NOTE: If mask_bos_pad_eos_tokens is True, we zero out activations for BOS, PAD, and EOS tokens.
    Later, we ignore zeroed activations."""

    all_acts_BLD = []

    for i in tqdm(
        range(0, len(tokens), batch_size),
        desc="Collecting activations",
        disable=not show_progress,
    ):
        tokens_BL = tokens[i : i + batch_size]

        acts_BLD = None

        def activation_hook(resid_BLD: torch.Tensor, hook):
            nonlocal acts_BLD
            acts_BLD = resid_BLD

        model.run_with_hooks(
            tokens_BL, stop_at_layer=layer + 1, fwd_hooks=[(hook_name, activation_hook)]
        )

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
            acts_BLD = acts_BLD * attn_mask_BL[:, :, None]  # type: ignore

        all_acts_BLD.append(acts_BLD)

    return torch.cat(all_acts_BLD, dim=0)


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_all_llm_activations(
    tokenized_inputs_dict: dict[
        str, dict[str, Int[torch.Tensor, "dataset_size seq_len"]]
    ],
    model: HookedTransformer,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
) -> dict[str, Float[torch.Tensor, "dataset_size seq_len d_model"]]:
    """If we have a dictionary of tokenized inputs for different classes, this function collects activations for all classes.
    We assume that the tokenized inputs have both the input_ids and attention_mask keys.
    VERY IMPORTANT NOTE: We zero out masked token activations in this function. Later, we ignore zeroed activations."""
    all_classes_acts_BLD = {}

    for class_name in tokenized_inputs_dict:
        tokens = tokenized_inputs_dict[class_name]["input_ids"]

        acts_BLD = get_llm_activations(
            tokens, model, batch_size, layer, hook_name, mask_bos_pad_eos_tokens
        )

        all_classes_acts_BLD[class_name] = acts_BLD

    return all_classes_acts_BLD


@jaxtyped(typechecker=beartype)
@torch.no_grad
def collect_sae_activations(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    sae: SAE | Any,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
    selected_latents: list[int] | None = None,
    activation_dtype: torch.dtype | None = None,
) -> Float[torch.Tensor, "dataset_size seq_len indexed_d_sae"]:
    """Collects SAE activations for a given set of tokens.
    Note: If evaluating many SAEs, it is more efficient to use save_activations() and encode_precomputed_activations()."""
    sae_acts = []

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        tokens_BL = tokens[i : i + batch_size]
        _, cache = model.run_with_cache(
            tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
        )
        resid_BLD: Float[torch.Tensor, "batch seq_len d_model"] = cache[hook_name]

        sae_act_BLF: Float[torch.Tensor, "batch seq_len d_sae"] = sae.encode(resid_BLD)

        if selected_latents is not None:
            sae_act_BLF = sae_act_BLF[:, :, selected_latents]

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
        else:
            attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)

        attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)

        sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]

        if activation_dtype is not None:
            sae_act_BLF = sae_act_BLF.to(dtype=activation_dtype)

        sae_acts.append(sae_act_BLF)

    all_sae_acts_BLF = torch.cat(sae_acts, dim=0)
    return all_sae_acts_BLF


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_feature_activation_sparsity(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    sae: SAE | Any,
    batch_size: int,
    layer: int,
    hook_name: str,
    mask_bos_pad_eos_tokens: bool = False,
) -> Float[torch.Tensor, "d_sae"]:
    """Get the activation sparsity for each SAE feature.
    Note: If evaluating many SAEs, it is more efficient to use save_activations() and get the sparsity from the saved activations."""
    device = sae.device
    running_sum_F = torch.zeros(sae.W_dec.shape[0], dtype=torch.float32, device=device)
    total_tokens = 0

    for i in tqdm(range(0, tokens.shape[0], batch_size)):
        tokens_BL = tokens[i : i + batch_size]
        _, cache = model.run_with_cache(
            tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
        )
        resid_BLD: Float[torch.Tensor, "batch seq_len d_model"] = cache[hook_name]

        sae_act_BLF: Float[torch.Tensor, "batch seq_len d_sae"] = sae.encode(resid_BLD)
        # make act to zero or one
        sae_act_BLF = (sae_act_BLF > 0).to(dtype=torch.float32)

        if mask_bos_pad_eos_tokens:
            attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, model.tokenizer)
        else:
            attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)

        attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)

        sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]
        total_tokens += attn_mask_BL.sum().item()

        running_sum_F += einops.reduce(sae_act_BLF, "B L F -> F", "sum")

    return running_sum_F / total_tokens


@jaxtyped(typechecker=beartype)
@torch.no_grad
def create_meaned_model_activations(
    all_llm_activations_BLD: dict[
        str, Float[torch.Tensor, "batch_size seq_len d_model"]
    ],
) -> dict[str, Float[torch.Tensor, "batch_size d_model"]]:
    """Mean activations across the sequence length dimension for each class while ignoring padding tokens.
    VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""

    all_llm_activations_BD = {}
    for class_name in all_llm_activations_BLD:
        acts_BLD = all_llm_activations_BLD[class_name]
        dtype = acts_BLD.dtype

        activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
        nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
        nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

        meaned_acts_BD = (
            einops.reduce(acts_BLD, "B L D -> B D", "sum") / nonzero_acts_B[:, None]
        )
        all_llm_activations_BD[class_name] = meaned_acts_BD

    return all_llm_activations_BD


@jaxtyped(typechecker=beartype)
@torch.no_grad
def get_sae_meaned_activations(
    all_llm_activations_BLD: dict[
        str, Float[torch.Tensor, "batch_size seq_len d_model"]
    ],
    sae: SAE | Any,
    sae_batch_size: int,
) -> dict[str, Float[torch.Tensor, "batch_size d_sae"]]:
    """Encode LLM activations with an SAE and mean across the sequence length dimension for each class while ignoring padding tokens.
    VERY IMPORTANT NOTE: We assume that the activations have been zeroed out for masked tokens."""

    dtype = sae.dtype

    all_sae_activations_BF = {}
    for class_name in all_llm_activations_BLD:
        all_acts_BLD = all_llm_activations_BLD[class_name]

        all_acts_BF = []

        for i in range(0, len(all_acts_BLD), sae_batch_size):
            acts_BLD = all_acts_BLD[i : i + sae_batch_size]
            acts_BLF = sae.encode(acts_BLD)

            activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
            nonzero_acts_BL = (activations_BL != 0.0).to(dtype=dtype)
            nonzero_acts_B = einops.reduce(nonzero_acts_BL, "B L -> B", "sum")

            acts_BLF = acts_BLF * nonzero_acts_BL[:, :, None]
            acts_BF = (
                einops.reduce(acts_BLF, "B L F -> B F", "sum") / nonzero_acts_B[:, None]
            )
            acts_BF = acts_BF.to(dtype=dtype)

            all_acts_BF.append(acts_BF)

        all_acts_BF = torch.cat(all_acts_BF, dim=0)
        all_sae_activations_BF[class_name] = all_acts_BF

    return all_sae_activations_BF


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def save_activations(
    tokens: Int[torch.Tensor, "dataset_size seq_len"],
    model: HookedTransformer,
    batch_size: int,
    layer: int,
    hook_name: str,
    num_chunks: int,
    save_size: int,
    artifacts_dir: str,
):
    """Save transformer activations to disk in chunks for later processing.

    Saves files named 'activations_XX_of_YY.pt' where XX is the chunk number (1-based)
    and YY is num_chunks. Each file contains a dict with 'activations' and 'tokens' keys."""
    dataset_size = tokens.shape[0]

    for save_idx in range(num_chunks):
        start_idx = save_idx * save_size
        end_idx = min((save_idx + 1) * save_size, dataset_size)
        tokens_SL = tokens[start_idx:end_idx]
        activations_list = []

        for i in tqdm(
            range(0, tokens_SL.shape[0], batch_size),
            desc=f"Saving chunk {save_idx + 1}/{num_chunks}",
        ):
            tokens_BL = tokens_SL[i : i + batch_size]
            _, cache = model.run_with_cache(
                tokens_BL, stop_at_layer=layer + 1, names_filter=hook_name
            )
            resid_BLD = cache[hook_name]

            activations_list.append(resid_BLD.cpu())

        activations_SLD = torch.cat(activations_list, dim=0)
        save_path = os.path.join(
            artifacts_dir, f"activations_{save_idx + 1}_of_{num_chunks}.pt"
        )

        file_contents = {"activations": activations_SLD, "tokens": tokens_SL.cpu()}

        torch.save(file_contents, save_path)
        print(f"Saved activations and tokens to {save_path}")


@jaxtyped(typechecker=beartype)
@torch.no_grad()
def encode_precomputed_activations(
    sae: SAE | Any,
    sae_batch_size: int,
    num_chunks: int,
    activation_dir: str,
    mask_bos_pad_eos_tokens: bool = False,
    selected_latents: list[int] | None = None,
    activation_dtype: torch.dtype | None = None,
) -> Float[torch.Tensor, "dataset_size seq_len d_sae"]:
    """Process saved activations through an SAE model, handling memory constraints through batching.

    This is the second stage of activation processing, meant to be run after save_activations().
    It loads the saved activation chunks, processes them through the SAE, and optionally:
    - Applies masking for special tokens
    - Selects specific SAE features
    - Converts to a specified dtype

    The batched processing allows handling large datasets that don't fit in memory.

    Returns:
        Tensor of encoded activations [dataset_size, seq_len, d_sae]
        If selected_latents is provided, d_sae will be len(selected_latents)
        Otherwise, d_sae will be the full SAE feature dimension"""

    all_sae_acts = []

    for save_idx in range(num_chunks):
        activation_file = os.path.join(
            activation_dir, f"activations_{save_idx + 1}_of_{num_chunks}.pt"
        )
        data = torch.load(activation_file)
        resid_SLD = data["activations"].to(device=sae.device)
        tokens_SL = data["tokens"]

        sae_act_batches = []
        num_samples = resid_SLD.shape[0]

        for batch_start in tqdm(
            range(0, num_samples, sae_batch_size),
            desc=f"Encoding chunk {save_idx + 1}/{num_chunks}",
        ):
            batch_end = min(batch_start + sae_batch_size, num_samples)
            resid_BLD = resid_SLD[batch_start:batch_end]
            tokens_BL = tokens_SL[batch_start:batch_end]

            sae_act_BLF = sae.encode(resid_BLD)

            if selected_latents is not None:
                sae_act_BLF = sae_act_BLF[:, :, selected_latents]

            if mask_bos_pad_eos_tokens:
                attn_mask_BL = get_bos_pad_eos_mask(tokens_BL, sae.model.tokenizer)  # type: ignore
            else:
                attn_mask_BL = torch.ones_like(tokens_BL, dtype=torch.bool)

            attn_mask_BL = attn_mask_BL.to(device=sae_act_BLF.device)
            sae_act_BLF = sae_act_BLF * attn_mask_BL[:, :, None]

            if activation_dtype is not None:
                sae_act_BLF = sae_act_BLF.to(dtype=activation_dtype)

            sae_act_batches.append(sae_act_BLF)

        sae_act_SLF = torch.cat(sae_act_batches, dim=0)
        all_sae_acts.append(sae_act_SLF)

    all_sae_acts = torch.cat(all_sae_acts, dim=0)
    return all_sae_acts
