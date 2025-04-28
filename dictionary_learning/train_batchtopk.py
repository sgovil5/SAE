import torch as t
from dictionary_learning import AutoEncoder
from dictionary_learning.buffer import ActivationBuffer
from dictionary_learning.trainers.batch_top_k import BatchTopKSAE, BatchTopKTrainer
from dictionary_learning.training import trainSAE
from dictionary_learning.utils import hf_dataset_to_generator
from nnsight import LanguageModel
import random

# Set device and random seed for reproducibility
DEVICE = "cuda" if t.cuda.is_available() else "cpu"
RANDOM_SEED = 42
MODEL_NAME = "EleutherAI/pythia-410m-deduped"
LAYER = 3
DATASET_NAME = "monology/pile-uncopyrighted"
SAVE_DIR = "./batchtopk_sae_model"

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
k = 64  # Number of active features per sample
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

# Configure BatchTopK trainer
trainer_cfg = {
    "trainer": BatchTopKTrainer,
    "dict_class": BatchTopKSAE,
    "steps": steps,
    "activation_dim": activation_dim,
    "dict_size": dict_size,
    "k": k,
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
    "wandb_name": f"BatchTopKTrainer-{MODEL_NAME}-{submodule_name}",
    "submodule_name": submodule_name,
}

save_interval = 1000
save_steps = list(range(save_interval, steps, save_interval))

# Train the model
print("Starting training...")
batchtopk_sae = trainSAE(
    data=buffer,
    trainer_configs=[trainer_cfg],
    steps=steps,
    save_steps=save_steps,
    save_dir=SAVE_DIR,
    normalize_activations=True,
)

print("Training complete!")