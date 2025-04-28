import torch as t
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import argparse
from tqdm import tqdm

# Necessary imports from the project (adjust paths if needed)
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.utils import hf_dataset_to_generator
from dictionary_learning.trainers.wta import WTASAE # Assuming ae.pt is a WTASAE instance
from nnsight import LanguageModel

# Default configuration (can be overridden by config.json)
DEFAULT_BUFFER_SIZE_GB = 2
DEFAULT_NUM_BATCHES_TO_FETCH = 50 # How many batches of activations to collect for analysis
DEFAULT_TOP_K_FEATURES_HIST = 20 # How many features to plot histograms for

def load_config_and_model(trainer_dir):
    """Loads config and SAE model."""
    config_path = os.path.join(trainer_dir, "config.json")
    model_path = os.path.join(trainer_dir, "ae.pt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading config from {config_path}")
    with open(config_path, 'r') as f:
        config = json.load(f)

    print(f"Loading model from {model_path}")
    # Ensure WTASAE class is available for torch.load
    sae_model = t.load(model_path, map_location=config.get('device', 'cpu'))
    sae_model.eval() # Set model to evaluation mode
    # If it's a state dict, you might need to instantiate the model first:
    # sae_model = WTASAE(**config_relevant_params)
    # sae_model.load_state_dict(t.load(model_path))
    # sae_model.to(config.get('device', 'cpu'))
    # sae_model.eval()

    print(f"Loaded SAE Type: {type(sae_model)}")
    return config, sae_model

def get_feature_activations(config, sae_model, num_batches_to_fetch, buffer_size_gb):
    """Gets feature activations using ActivationBuffer."""
    device = config.get('device', 'cpu')

    # --- Initialize Base Model ---
    print(f"Initializing base model: {config['lm_name']}")
    model = LanguageModel(config['lm_name'], dispatch=True, device_map=device)
    submodule = model
    # Navigate to the target submodule using the path stored in config
    submodule_path_parts = config['submodule_name'].split('.')
    # Assumes path like "gpt_neox.layers[3]" - needs adjustment if format differs
    for part in submodule_path_parts:
        if '[' in part and ']' in part:
            name, index = part.split('[')
            index = int(index.replace(']', ''))
            submodule = getattr(submodule, name)[index]
        else:
             submodule = getattr(submodule, part)
    print(f"Target submodule: {type(submodule)}")

    # --- Initialize Dataset ---
    print(f"Initializing dataset: {config['dataset_name']}")
    generator = hf_dataset_to_generator(config['dataset_name'])

    # --- Initialize Buffer ---
    # Estimate buffer parameters based on desired memory usage
    activation_dim = config['activation_dim']
    sae_batch_size = config.get('sae_batch_size', 8192) # Get from config or use default
    context_length = config.get('context_length', 128) # Get from config or use default
    llm_batch_size = config.get('llm_batch_size', 512) # Get from config or use default

    # Calculate n_ctxs based on buffer_size_gb
    bytes_per_activation_batch = activation_dim * sae_batch_size * 4 # float32
    est_bytes_per_context = bytes_per_activation_batch / (sae_batch_size // context_length)
    target_buffer_bytes = buffer_size_gb * (1024**3)
    n_ctxs = int(target_buffer_bytes / est_bytes_per_context)
    print(f"Buffer Target Size: {buffer_size_gb} GB")
    print(f"Estimated n_ctxs for buffer: {n_ctxs} (adjust buffer_size_gb if needed)")

    buffer = ActivationBuffer(
        generator,
        model,
        submodule,
        n_ctxs=n_ctxs,
        ctx_len=context_length,
        refresh_batch_size=llm_batch_size,
        out_batch_size=sae_batch_size,
        io="out", # Assuming 'out' based on train_wta.py
        d_submodule=activation_dim,
        device=device,
    )

    # --- Collect Activations ---
    all_feature_acts = []
    print(f"Fetching {num_batches_to_fetch} batches of activations...")
    for _ in tqdm(range(num_batches_to_fetch), desc="Fetching activations"):
        base_model_acts = buffer.next()
        with t.no_grad():
            # Assuming WTA SAE encode directly outputs sparse features
            feature_acts = sae_model.encode(base_model_acts)
        all_feature_acts.append(feature_acts.cpu())

    buffer.close() # Clean up buffer resources if necessary
    print("Activation fetching complete.")
    return t.cat(all_feature_acts, dim=0)

def plot_activation_frequency(feature_acts, save_dir):
    """Calculates and plots the log frequency of activation for each feature."""
    num_samples = feature_acts.shape[0]
    num_features = feature_acts.shape[1]

    # Calculate frequency
    is_active = feature_acts != 0
    num_active_per_feature = is_active.sum(dim=0).float()
    freq_per_feature = num_active_per_feature / num_samples

    # Filter out features that never activate
    active_mask = freq_per_feature > 0
    active_freqs = freq_per_feature[active_mask]
    active_indices = t.where(active_mask)[0]

    if active_freqs.numel() == 0:
        print("Warning: No features activated.")
        return None, None

    # Sort by frequency
    sorted_freqs, sorted_indices_of_active = t.sort(active_freqs, descending=True)
    original_indices_sorted = active_indices[sorted_indices_of_active]

    # Plot Log Frequency
    plt.figure(figsize=(12, 6))
    plt.plot(np.log10(sorted_freqs.numpy()))
    plt.xlabel("Feature Index (Sorted by Frequency)")
    plt.ylabel("Log10(Activation Frequency)")
    plt.title(f"Log Activation Frequency ({active_freqs.numel()}/{num_features} features active)")
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plot_path = os.path.join(save_dir, "log_activation_frequency.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved activation frequency plot to {plot_path}")

    return freq_per_feature, original_indices_sorted # Return all freqs and sorted original indices

def plot_activation_histograms(feature_acts, sorted_feature_indices, num_features_to_plot, save_dir):
    """Plots histograms of non-zero activation values for the most frequent features."""
    if sorted_feature_indices is None or num_features_to_plot == 0:
        return

    num_features_to_plot = min(num_features_to_plot, len(sorted_feature_indices))
    top_k_indices = sorted_feature_indices[:num_features_to_plot]

    print(f"Plotting activation histograms for top {num_features_to_plot} features...")

    n_cols = 5
    n_rows = (num_features_to_plot + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)
    axes = axes.flatten()

    for i, feature_idx in enumerate(tqdm(top_k_indices, desc="Plotting Histograms")):
        feature_values = feature_acts[:, feature_idx]
        non_zero_values = feature_values[feature_values != 0].numpy()

        if len(non_zero_values) > 0:
            axes[i].hist(non_zero_values, bins=50, log=True) # Log scale often helpful
            axes[i].set_title(f"Feature {feature_idx.item()}")
            axes[i].set_xlabel("Activation Value")
            axes[i].set_ylabel("Frequency (log scale)")
        else:
            axes[i].set_title(f"Feature {feature_idx.item()} (No activations)")
        axes[i].grid(True, linestyle='--', linewidth=0.5)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plot_path = os.path.join(save_dir, "top_activation_histograms.png")
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved activation histogram plot to {plot_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize SAE Feature Activations")
    parser.add_argument("trainer_dir", type=str, help="Path to the trainer directory (e.g., ./wta_sae_model_0.01/trainer_0)")
    parser.add_argument("--num_batches", type=int, default=DEFAULT_NUM_BATCHES_TO_FETCH, help="Number of SAE batches to fetch for analysis")
    parser.add_argument("--buffer_gb", type=float, default=DEFAULT_BUFFER_SIZE_GB, help="Approximate size of activation buffer in GB")
    parser.add_argument("--top_k_hist", type=int, default=DEFAULT_TOP_K_FEATURES_HIST, help="Number of top features to plot activation histograms for")
    args = parser.parse_args()

    if not os.path.isdir(args.trainer_dir):
        print(f"Error: Trainer directory not found: {args.trainer_dir}")
        exit(1)

    # Create a sub-directory for visualizations
    vis_save_dir = os.path.join(args.trainer_dir, "visualizations")
    os.makedirs(vis_save_dir, exist_ok=True)

    # 1. Load Config and Model
    config, sae_model = load_config_and_model(args.trainer_dir)

    # 2. Get Activations
    feature_acts = get_feature_activations(config, sae_model, args.num_batches, args.buffer_gb)

    # 3. Plot Activation Frequency
    _, sorted_indices = plot_activation_frequency(feature_acts, vis_save_dir)

    # 4. Plot Activation Histograms
    plot_activation_histograms(feature_acts, sorted_indices, args.top_k_hist, vis_save_dir)

    print("--- Visualization complete ---")
