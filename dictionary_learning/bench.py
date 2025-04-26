import torch

# Load your SAE
hf_repo_id = "sgovil5/wtasae"
sae_checkpoint_path = "ae.pt"
model_name = "pythia-70m-deduped"  # Adjust as needed
layer = 3  # Adjust as needed
hook_name = f"blocks.{layer}.hook_resid_post"  # Adjust as needed

custom_sae = load_dictionary_learning_sae_from_hf_hub(
    hf_repo_id=hf_repo_id,
    sae_checkpoint_path=sae_checkpoint_path,
    model_name=model_name,
    layer=layer,
    hook_name=hook_name,
    device="cuda"  # or "cpu"
)

# Run evaluations (example for core eval)
from sae_bench.evals.core.main import run_core_eval

results = run_core_eval(
    saes=[custom_sae],
    model_name=model_name,
    device="cuda",  # or "cpu"
    save_results=True
)

# Print results
print(results)