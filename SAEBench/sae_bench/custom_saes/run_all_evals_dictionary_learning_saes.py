import json
import os

import torch
from huggingface_hub import snapshot_download
from tqdm import tqdm

import sae_bench.custom_saes.base_sae as base_sae
import sae_bench.custom_saes.batch_topk_sae as batch_topk_sae
import sae_bench.custom_saes.gated_sae as gated_sae
import sae_bench.custom_saes.jumprelu_sae as jumprelu_sae
import sae_bench.custom_saes.relu_sae as relu_sae
import sae_bench.custom_saes.topk_sae as topk_sae
import sae_bench.evals.absorption.main as absorption
import sae_bench.evals.autointerp.main as autointerp
import sae_bench.evals.core.main as core
import sae_bench.evals.ravel.main as ravel
import sae_bench.evals.scr_and_tpp.main as scr_and_tpp
import sae_bench.evals.sparse_probing.main as sparse_probing
import sae_bench.evals.unlearning.main as unlearning
import sae_bench.sae_bench_utils.general_utils as general_utils
import sae_bench.custom_saes.wta_sae as wta_sae

MODEL_CONFIGS = {
    "pythia-70m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3],
        "d_model": 512,
    },
    "pythia-410m-deduped": {
        "batch_size": 512,
        "dtype": "float32",
        "layers": [3],
        "d_model": 1024,
    },
    # "pythia-160m-deduped": {
    #     "batch_size": 256,
    #     "dtype": "float32",
    #     "layers": [8],
    #     "d_model": 768,
    # },
    # "gemma-2-2b": {
    #     "batch_size": 32,
    #     "dtype": "bfloat16",
    #     "layers": [5, 12, 19],
    #     "d_model": 2304,
    # },
}

output_folders = {
    "absorption": "eval_results/absorption",
    "autointerp": "eval_results/autointerp",
    "core": "eval_results/core",
    "scr": "eval_results/scr",
    "tpp": "eval_results/tpp",
    "sparse_probing": "eval_results/sparse_probing",
    "unlearning": "eval_results/unlearning",
    "ravel": "eval_results/ravel",
}


TRAINER_LOADERS = {
    # "MatryoshkaBatchTopKTrainer": batch_topk_sae.load_dictionary_learning_matryoshka_batch_topk_sae,
    # "BatchTopKTrainer": batch_topk_sae.load_dictionary_learning_batch_topk_sae,
    # "TopKTrainer": topk_sae.load_dictionary_learning_topk_sae,
    # "StandardTrainerAprilUpdate": relu_sae.load_dictionary_learning_relu_sae,
    # "StandardTrainer": relu_sae.load_dictionary_learning_relu_sae,
    # "PAnnealTrainer": relu_sae.load_dictionary_learning_relu_sae,
    # "JumpReluTrainer": jumprelu_sae.load_dictionary_learning_jump_relu_sae,
    # "GatedSAETrainer": gated_sae.load_dictionary_learning_gated_sae,
    "WTATrainer": wta_sae.load_dictionary_learning_wta_sae,
}


def get_all_local_autoencoders(local_sae_parent_dir: str) -> list[str]:
    """
    Finds all subdirectories within the local_sae_parent_dir that contain a config.json file.
    Returns a list of relative paths to these subdirectories.
    """
    sae_locations = []
    if not os.path.isdir(local_sae_parent_dir):
        print(f"Warning: Local SAE directory not found: {local_sae_parent_dir}")
        return []

    # Walk through the directory
    for root, dirs, files in os.walk(local_sae_parent_dir):
        # Check if config.json exists in the current directory
        if "config.json" in files and "ae.pt" in files:
            # Calculate the relative path from the parent directory
            relative_path = os.path.relpath(root, local_sae_parent_dir)
            # Handle the case where the SAE is directly in the parent directory
            if relative_path == ".":
                 # Check if the parent dir itself contains config.json and ae.pt
                 # This case is less common for multiple SAEs but possible.
                 # We usually expect SAEs in subdirectories.
                 # Let's assume SAEs are always in subdirectories for now.
                 # If config.json is in the root, it might be a single SAE run?
                 # For consistency, let's focus on subdirs.
                 pass
            else:
                 # Check if it's a direct subdirectory containing the config
                 # Avoids adding intermediate directories if structure is nested deeper
                 if os.path.exists(os.path.join(local_sae_parent_dir, relative_path, "config.json")):
                     sae_locations.append(relative_path)
            # Prevent os.walk from going deeper into already found SAE directories
            dirs[:] = [d for d in dirs if os.path.relpath(os.path.join(root, d), local_sae_parent_dir) not in sae_locations]


    # Deduplicate in case of unusual structures, though the logic above tries to avoid it
    sae_locations = sorted(list(set(sae_locations)))
    print(f"Found local SAE locations relative to {local_sae_parent_dir}: {sae_locations}")
    return sae_locations


def load_dictionary_learning_sae(
    local_sae_parent_dir: str,
    location: str,
    model_name: str,
    device: str,
    dtype: torch.dtype,
    layer: int | None = None,
) -> base_sae.BaseSAE:
    sae_dir = os.path.join(local_sae_parent_dir, location)
    config_file = os.path.join(sae_dir, "config.json")
    ae_file = os.path.join(sae_dir, "ae.pt")
    
    if not os.path.exists(config_file):
        raise FileNotFoundError(f"Config file not found at: {config_file}")
    if not os.path.exists(ae_file):
        raise FileNotFoundError(f"SAE checkpoint file not found at: {ae_file}")

    # Print for debugging
    print(f"Loading config from: {config_file}")
    print(f"Loading ae.pt from: {ae_file}")
    
    with open(config_file) as f:
        config = json.load(f)

    trainer_class = config["trainer"]["trainer_class"]
    
    # Print for debugging
    print(f"Using trainer class: {trainer_class}")
    print(f"Loading ae.pt from: {ae_file}")

    # --- Revised Parameter Extraction ---
    # Function to safely get nested keys, checking 'model', 'trainer', then top-level
    def get_param(keys_to_try, default=None):
        # Ensure keys_to_try is a list
        if isinstance(keys_to_try, str):
            keys_to_try = [keys_to_try]
        
        # Check under 'model' sub-dictionary
        model_cfg = config.get("model", {})
        for key_name in keys_to_try:
            value = model_cfg.get(key_name)
            if value is not None:
                print(f"Found parameter '{key_name}' in 'model': {value}")
                return value
        
        # Check under 'trainer' sub-dictionary
        trainer_cfg = config.get("trainer", {})
        for key_name in keys_to_try:
            value = trainer_cfg.get(key_name)
            if value is not None:
                print(f"Found parameter '{key_name}' in 'trainer': {value}")
                return value

        # Check under 'buffer' sub-dictionary (for d_in specific keys)
        buffer_cfg = config.get("buffer", {})
        for key_name in keys_to_try:
             if key_name in ["activation_dim", "d_submodule"]: # Only check buffer for these specific keys
                value = buffer_cfg.get(key_name)
                if value is not None:
                    print(f"Found parameter '{key_name}' in 'buffer': {value}")
                    return value

        # Check at the top level
        for key_name in keys_to_try:
            value = config.get(key_name)
            if value is not None:
                print(f"Found parameter '{key_name}' at top level: {value}")
                return value
                
        print(f"Parameter not found using keys: {keys_to_try}. Returning default: {default}")
        return default

    # Define primary and alternative key names
    d_in_keys = ["d_model", "activation_size", "activation_dim", "d_submodule"]
    d_sae_keys = ["dict_size"]
    k_keys = ["k"]
    l1_coeff_keys = ["l1_coeff", "l1_coefficient"]
    use_bias_keys = ["use_bias"]

    # Extract parameters using the flexible get_param function
    d_in = get_param(d_in_keys)
    if d_in is None:
        if model_name and model_name in MODEL_CONFIGS:
             d_in = MODEL_CONFIGS[model_name]['d_model']
             print(f"Parameter 'd_in' not found in config, using MODEL_CONFIGS fallback: {d_in}")
        else:
             raise ValueError(f"Could not determine 'd_in' from config {config_file} or MODEL_CONFIGS fallback.")

    d_sae = get_param(d_sae_keys)
    if d_sae is None:
        raise ValueError(f"Could not determine 'd_sae' using keys {d_sae_keys} from config: {config_file}")

    l1_coefficient = get_param(l1_coeff_keys, default=0.0)
    use_bias = get_param(use_bias_keys, default=True)
    # --- End Revised Parameter Extraction ---

    print(f"Instantiating SAE for trainer class: {trainer_class}")
    print(f"Parameters: d_in={d_in}, d_sae={d_sae}, l1_coeff={l1_coefficient}, use_bias={use_bias}, device={device}, dtype={dtype}")

    # Instantiate based on trainer class
    if trainer_class == "StandardTrainerAprilUpdate" or trainer_class == "StandardTrainer" or trainer_class == "PAnnealTrainer":
        # Assuming these map to ReluSAE
        sae = relu_sae.ReluSAE(
            d_in=d_in,
            d_sae=d_sae,
            l1_coefficient=l1_coefficient,
            dtype=dtype,
            device=device,
            # Add other required ReluSAE params if any, e.g., use_bias
            # use_bias=cfg_dict.get("use_bias", True) 
            use_bias=use_bias
        )
    elif trainer_class == "TopKTrainer":
        # k = cfg_dict.get("k")
        k = get_param(k_keys)
        if k is None:
             raise ValueError(f"Could not determine 'k' for TopKTrainer from config: {config_file}")
        print(f"TopK Specific: k={k}")
        sae = topk_sae.TopKSAE(
             d_in=d_in,
             d_sae=d_sae,
             k=k,
             l1_coefficient=l1_coefficient, # Check if TopKSAE uses this
             dtype=dtype,
             device=device,
             # use_bias=cfg_dict.get("use_bias", True)
             use_bias=use_bias
        )
    elif trainer_class == "BatchTopKTrainer": # Add BatchTopK logic
         # k = cfg_dict.get("k")
         k = get_param(k_keys)
         if k is None:
              raise ValueError(f"Could not determine 'k' for BatchTopKTrainer from config: {config_file}")
         print(f"BatchTopK Specific: k={k}")
         sae = batch_topk_sae.BatchTopKSAE( # Assuming class name
              d_in=d_in,
              d_sae=d_sae,
              k=k,
              l1_coefficient=l1_coefficient, # Check if BatchTopKSAE uses this
              dtype=dtype,
              device=device,
              # use_bias=cfg_dict.get("use_bias", True)
              use_bias=use_bias
         )
    elif trainer_class == "JumpReluTrainer":
         # Add JumpRelu logic - check required params
         sae = jumprelu_sae.JumpReluSAE( # Assuming class name
             d_in=d_in,
             d_sae=d_sae,
             l1_coefficient=l1_coefficient,
             dtype=dtype,
             device=device,
             # Add other required JumpReluSAE params if any
             # use_bias=cfg_dict.get("use_bias", True)
             use_bias=use_bias
         )
    elif trainer_class == "GatedSAETrainer":
         sae = gated_sae.GatedSAE( # Assuming class name
             d_in=d_in,
             d_sae=d_sae,
             l1_coefficient=l1_coefficient,
             dtype=dtype,
             device=device,
             # Add other required GatedSAE params if any
             # use_bias=cfg_dict.get("use_bias", True)
             use_bias=use_bias
         )
    elif trainer_class == "WTATrainer":
        # Fetch sparsity_rate instead of k for WTA
        sparsity_rate_keys = ["sparsity_rate"]
        sparsity_rate = get_param(sparsity_rate_keys)
        if sparsity_rate is None:
            raise ValueError(f"Could not determine 'sparsity_rate' for WTATrainer from config: {config_file}")
        print(f"WTA Specific: sparsity_rate={sparsity_rate}")

        # Need hook_layer and model_name for WTASAE constructor
        # These are already passed to load_dictionary_learning_sae, but WTASAE needs them
        # Let's try to get layer from config first, fall back to MODEL_CONFIGS if needed
        layer_keys = ["layer"]
        hook_layer = get_param(layer_keys)
        if hook_layer is None:
             # Try to infer from model_name config if possible?
             # Or should we require it in the config?
             # For now, let's raise an error if not found in config.
            raise ValueError(f"Could not determine 'layer' for WTATrainer from config: {config_file}")
        
        # model_name is already an argument to the function
        # Make sure it's the full name expected by WTASAE (potentially from config)
        lm_name_keys = ["lm_name"]
        config_lm_name = get_param(lm_name_keys)
        if config_lm_name is None:
            print(f"Warning: Could not find 'lm_name' in config {config_file}. Using model_name argument: {model_name}")
            actual_model_name = model_name # Use the passed model_name
        else:
            actual_model_name = config_lm_name # Use the name from config

        sae = wta_sae.WTASAE(
            d_in=d_in,
            d_sae=d_sae,
            sparsity_rate=sparsity_rate, # Pass sparsity_rate
            model_name=actual_model_name, # Use model name from config or arg
            hook_layer=hook_layer, # Pass layer from config
            dtype=dtype,
            device=device,
            # WTASAE doesn't take use_bias in its constructor
            # use_bias=use_bias
        )
    # Add elif blocks for MatryoshkaBatchTopKTrainer if needed
    else:
        # Fallback or error if trainer_class is unknown
        # Check if the loader map has a corresponding class we can import?
        # This requires inspecting the TRAINER_LOADERS functions or the modules they point to.
        # For now, raise an error.
        raise NotImplementedError(f"SAE loading not implemented for trainer class: {trainer_class}. Config: {config_file}")

    # Load state dictionary
    print(f"Loading state dict from: {ae_file}")
    state_dict = torch.load(ae_file, map_location=device)
    
    # Check if the state dict is nested (e.g., under 'model_state_dict')
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    elif 'state_dict' in state_dict: # another common pattern
        state_dict = state_dict['state_dict']

    # Adjust keys if necessary (e.g., remove "module." prefix if saved with DataParallel)
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

    # --- Specific Key Mapping/Transposition for WTATrainer ---
    if trainer_class == "WTATrainer":
        print("Applying WTASAE specific key mapping and transposition...")
        key_mapping = {
            "encoder.weight": "W_enc",
            "decoder.weight": "W_dec",
            "encoder.bias": "b_enc",
            "b_dec": "b_dec", # Already matches
            "threshold": "threshold", # Already matches
        }
        # Create a new dictionary with renamed keys, keeping others as is
        renamed_params = {key_mapping.get(k, k): v for k, v in state_dict.items()}
        
        # Transpose weights if they exist in the renamed dict
        if "W_enc" in renamed_params:
            renamed_params["W_enc"] = renamed_params["W_enc"].T
        if "W_dec" in renamed_params:
             renamed_params["W_dec"] = renamed_params["W_dec"].T
        
        # Use the renamed parameters for loading
        final_state_dict = renamed_params
        print(f"Mapped Checkpoint keys: {final_state_dict.keys()}")
    else:
        # For other trainer types, assume keys match or handle specific mappings if needed
        final_state_dict = state_dict
        print(f"Using original Checkpoint keys: {final_state_dict.keys()}")
    # --- End Specific Key Mapping ---

    try:
        # Load the potentially modified state dict
        print(f"Attempting to load state_dict into SAE. SAE keys: {sae.state_dict().keys()}")
        sae.load_state_dict(final_state_dict)
        print("State dict loaded successfully.")
    except RuntimeError as e:
        print(f"Error loading state dict for {ae_file}. Keys might mismatch.")

    sae = sae.to(dtype)
    sae.eval()

    return sae


def verify_saes_load(
    local_sae_parent_dir: str,
    sae_locations: list[str],
    model_name: str,
    device: str,
    dtype: torch.dtype,
):
    """Verify that all SAEs load correctly from local paths."""
    print(f"Verifying SAE loading from: {local_sae_parent_dir}")
    for sae_location in sae_locations:
        print(f"Attempting to load SAE at relative location: {sae_location}")
        try:
            sae = load_dictionary_learning_sae(
                local_sae_parent_dir=local_sae_parent_dir,
                location=sae_location,
                layer=None,
                model_name=model_name,
                device=device,
                dtype=dtype,
            )
            print(f"Successfully loaded SAE: {sae_location}")
            del sae
        except Exception as e:
             print(f"Failed to load SAE at {sae_location}: {e}")
             raise e


def run_evals(
    local_sae_parent_dir: str,
    model_name: str,
    sae_locations: list[str],
    llm_batch_size: int,
    llm_dtype: str,
    device: str,
    eval_types: list[str],
    random_seed: int,
    api_key: str | None = None,
    force_rerun: bool = True,
    cache_dir: str | None = None,
):
    """Run selected evaluations for the given model and SAEs."""

    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model_name}")

    # Mapping of eval types to their functions and output paths
    eval_runners = {
        "absorption": (
            lambda selected_saes, is_final: absorption.run_eval(
                absorption.AbsorptionEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    cache_dir=cache_dir,
                ),
                selected_saes,
                device,
                "eval_results/absorption",
                force_rerun,
            )
        ),
        "autointerp": (
            lambda selected_saes, is_final: autointerp.run_eval(
                autointerp.AutoInterpEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                api_key,  # type: ignore
                "eval_results/autointerp",
                force_rerun,
            )
        ),
        "core": (
            lambda selected_saes, is_final: core.multiple_evals(
                selected_saes=selected_saes,
                n_eval_reconstruction_batches=200,
                n_eval_sparsity_variance_batches=2000,
                eval_batch_size_prompts=16,
                compute_featurewise_density_statistics=True,
                compute_featurewise_weight_based_metrics=True,
                exclude_special_tokens_from_reconstruction=True,
                dataset="Skylion007/openwebtext",
                context_size=128,
                output_folder="eval_results/core",
                verbose=True,
                dtype=llm_dtype,
                device=device,
            )
        ),
        "ravel": (
            lambda selected_saes, is_final: ravel.run_eval(
                ravel.RAVELEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size // 4,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/ravel",
                force_rerun,
            )
        ),
        "scr": (
            lambda selected_saes, is_final: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    perform_scr=True,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    cache_dir=cache_dir,
                ),
                selected_saes,
                device,
                "eval_results",
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "tpp": (
            lambda selected_saes, is_final: scr_and_tpp.run_eval(
                scr_and_tpp.ScrAndTppEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    perform_scr=False,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                    cache_dir=cache_dir,
                ),
                selected_saes,
                device,
                "eval_results",
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "sparse_probing": (
            lambda selected_saes, is_final: sparse_probing.run_eval(
                sparse_probing.SparseProbingEvalConfig(
                    model_name=model_name,
                    random_seed=random_seed,
                    llm_batch_size=llm_batch_size,
                    llm_dtype=llm_dtype,
                ),
                selected_saes,
                device,
                "eval_results/sparse_probing",
                force_rerun,
                clean_up_activations=is_final,
                save_activations=True,
            )
        ),
        "unlearning": (
            lambda selected_saes, is_final: unlearning.run_eval(
                unlearning.UnlearningEvalConfig(
                    model_name="gemma-2-2b-it",
                    random_seed=random_seed,
                    llm_dtype=llm_dtype,
                    llm_batch_size=llm_batch_size
                    // 8,
                ),
                selected_saes,
                device,
                "eval_results/unlearning",
                force_rerun,
            )
        ),
    }

    for eval_type in eval_types:
        if eval_type not in eval_runners:
            raise ValueError(f"Unsupported eval type: {eval_type}")

    verify_saes_load(
        local_sae_parent_dir,
        sae_locations,
        model_name,
        device,
        general_utils.str_to_dtype(llm_dtype),
    )

    # Run selected evaluations
    for eval_type in tqdm(eval_types, desc="Evaluations"):
        if eval_type == "autointerp" and api_key is None:
            print("Skipping autointerp evaluation due to missing API key")
            continue
        if eval_type == "unlearning":
            if not os.path.exists(
                "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
            ):
                print(
                    "Skipping unlearning evaluation due to missing bio-forget-corpus.jsonl"
                )
                continue

        print(f"\n\n\nRunning {eval_type} evaluation\n\n\n")

        try:
            for i, sae_location in enumerate(sae_locations):
                is_final = False
                if i == len(sae_locations) - 1:
                    is_final = True

                sae = load_dictionary_learning_sae(
                    local_sae_parent_dir=local_sae_parent_dir,
                    location=sae_location,
                    layer=None,
                    model_name=model_name,
                    device=device,
                    dtype=general_utils.str_to_dtype(llm_dtype),
                )
                safe_location = sae_location.replace(os.path.sep, "_")
                unique_sae_id = f"{model_name}_{safe_location}"
                print(f"Generated unique_sae_id: {unique_sae_id}")
                selected_saes = [(unique_sae_id, sae)]

                os.makedirs(output_folders[eval_type], exist_ok=True)
                eval_runners[eval_type](selected_saes, is_final)

                del sae

        except Exception as e:
            print(f"Error running {eval_type} evaluation: {e}")
            continue


if __name__ == "__main__":
    """
    This will run all evaluations on all selected dictionary_learning SAEs found locally.
    Specify the parent directory containing the SAE output folders in `local_sae_parent_dir`.
    Set the model_name corresponding to the SAEs being evaluated in `model_name_to_eval`.
    Ensure the configuration for `model_name_to_eval` exists in `MODEL_CONFIGS`.
    Also specify the eval types you want to run in `eval_types`.
    You can also specify any keywords to exclude/include in the SAE folder names using `exclude_keywords` and `include_keywords`.
    NOTE: This relies on each SAE being located in a folder which contains an ae.pt file and a config.json file.
    """
    RANDOM_SEED = 42
    
    model_name_to_eval = "pythia-410m-deduped"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    sae_bench_dir = os.path.dirname(os.path.dirname(script_dir))
    workspace_root = os.path.dirname(sae_bench_dir)
    local_sae_parent_dir = os.path.expanduser("~/scratch/SAE/dictionary_learning")

    print(f"Looking for local SAEs in: {local_sae_parent_dir}")
    
    scratch_dir = os.path.expanduser("~/scratch")
    os.makedirs(scratch_dir, exist_ok=True)
    
    hf_cache_dir = os.path.join(scratch_dir, "hf_cache")
    os.makedirs(hf_cache_dir, exist_ok=True)
    os.environ["HF_HOME"] = hf_cache_dir
    os.environ["TRANSFORMERS_CACHE"] = os.path.join(hf_cache_dir, "transformers")
    os.environ["HF_DATASETS_CACHE"] = os.path.join(hf_cache_dir, "datasets")
    
    wandb_cache_dir = os.path.join(scratch_dir, "wandb")
    os.makedirs(wandb_cache_dir, exist_ok=True)
    os.environ["WANDB_CACHE_DIR"] = wandb_cache_dir
    
    download_location = os.path.join(scratch_dir, "downloaded_saes")
    os.makedirs(download_location, exist_ok=True)

    device = general_utils.setup_environment()

    eval_types = [
        "core",
        "scr",
        "tpp",
        "sparse_probing",
        "ravel",
    ]

    if "autointerp" in eval_types:
        try:
            with open("openai_api_key.txt") as f:
                api_key = f.read().strip()
        except FileNotFoundError:
            raise Exception("Please create openai_api_key.txt with your API key")
    else:
        api_key = None

    if "unlearning" in eval_types:
        if not os.path.exists(
            "./sae_bench/evals/unlearning/data/bio-forget-corpus.jsonl"
        ):
            raise Exception(
                "Please download bio-forget-corpus.jsonl for unlearning evaluation"
            )

    exclude_keywords = ["checkpoints", "old"]
    include_keywords = []

    print(f"\n\n\nEvaluating local SAEs for model: {model_name_to_eval}\n\n\n")

    if model_name_to_eval not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_name_to_eval}' not found in MODEL_CONFIGS. Please add its configuration.")

    llm_batch_size = MODEL_CONFIGS[model_name_to_eval]["batch_size"]
    str_dtype = MODEL_CONFIGS[model_name_to_eval]["dtype"]
    torch_dtype = general_utils.str_to_dtype(str_dtype)

    sae_locations = get_all_local_autoencoders(local_sae_parent_dir)

    sae_locations = general_utils.filter_keywords(
        sae_locations,
        exclude_keywords=exclude_keywords,
        include_keywords=include_keywords,
    )

    if not sae_locations:
        print(f"No SAE locations found or remaining after filtering in {local_sae_parent_dir}.")
    else:
        print(f"Filtered SAE locations to evaluate: {sae_locations}")
        run_evals(
            local_sae_parent_dir=local_sae_parent_dir,
            model_name=model_name_to_eval,
            sae_locations=sae_locations,
            llm_batch_size=llm_batch_size,
            llm_dtype=str_dtype,
            device=device,
            eval_types=eval_types,
            api_key=api_key,
            random_seed=RANDOM_SEED,
            force_rerun=True,
        )
