import torch as t
from dictionary_learning import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.trainers.wta import WTASAE, WTATrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator
from nnsight import LanguageModel
import random
import os  # Import os for path manipulation

# Set device and random seed for reproducibility
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
RANDOM_SEED = 42
MODEL_NAME = "EleutherAI/pythia-410m-deduped"
LAYER = 3
DATASET_NAME = "monology/pile-uncopyrighted"


# Set random seeds for reproducibility
random.seed(RANDOM_SEED)
t.manual_seed(RANDOM_SEED)

# Initialize model
model = LanguageModel(MODEL_NAME, dispatch=True, device_map=DEVICE)

# Buffer parameters
context_length = 128
llm_batch_size = 512  # Fits on a 24GB GPU
sae_batch_size = 8192
num_contexts_per_sae_batch = sae_batch_size // context_length
num_inputs_in_buffer = num_contexts_per_sae_batch * 20
num_tokens = 100_000_000

# SAE parameters
expansion_factor = 8
# List of sparsity rates to ablate over
sparsity_rates = [5.0e-3, 1.0e-2, 5.0e-2]
steps = int(num_tokens / sae_batch_size)  # Total number of batches to train
warmup_steps = 1000
decay_start = None
auxk_alpha = 1 / 32

# Get model dimensions
submodule = model.gpt_neox.layers[LAYER]
submodule_name = f"resid_post_layer_{LAYER}"
io = "out"
activation_dim = model.config.hidden_size
dict_size = expansion_factor * activation_dim

# Create data generator
generator = hf_dataset_to_generator(DATASET_NAME)

# Create activation buffer
print("Creating activation buffer...")
buffer = ActivationBuffer(
    generator,
    model,
    submodule,
    n_ctxs=num_inputs_in_buffer,
    ctx_len=context_length,
    refresh_batch_size=llm_batch_size,
    out_batch_size=sae_batch_size,
    io=io,
    d_submodule=activation_dim,
    device=DEVICE,
)

# Loop over sparsity rates
for sparsity_rate in sparsity_rates:
    print(f"--- Training with sparsity_rate = {sparsity_rate} ---")

    SAVE_DIR = f"./wta_sae_model_{sparsity_rate}"
    os.makedirs(SAVE_DIR, exist_ok=True)

    # Configure WTA trainer
    trainer_cfg = {
        "trainer": WTATrainer,
        "dict_class": WTASAE,
        "steps": steps,
        "activation_dim": activation_dim,
        "dict_size": dict_size,
        "sparsity_rate": sparsity_rate,
        "layer": LAYER,
        "lm_name": MODEL_NAME,
        "lr": None,  # Let the trainer calculate the optimal learning rate
        "auxk_alpha": auxk_alpha,
        "warmup_steps": warmup_steps,
        "decay_start": decay_start,
        "threshold_beta": 0.999,
        "threshold_start_step": 1000,
        "seed": RANDOM_SEED,
        "device": DEVICE,
        "wandb_name": f"WTATrainer-{MODEL_NAME}-{submodule_name}-sparsity_{sparsity_rate}",
        "submodule_name": submodule_name,
    }

    save_interval = 4000
    save_steps = list(range(save_interval, steps, save_interval))

    # Train the model
    print("Starting training...")
    wta_sae = trainSAE(
        data=buffer,
        trainer_configs=[trainer_cfg],
        steps=steps,
        save_steps=save_steps,
        save_dir=SAVE_DIR,
        normalize_activations=True,
    )

    print(f"Training complete for sparsity_rate = {sparsity_rate}!")

print("--- Ablation study finished! ---")
