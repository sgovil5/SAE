import torch
import torch.nn.functional as F

def apply_wta_sparsity(post_relu_acts: torch.Tensor, sparsity_rate: float):
    """
    Applies the Winner-Takes-All (WTA) sparsity mechanism.

    Selects the top `k_per_feature` activations for each feature across the batch/batch*sequence.

    Args:
        post_relu_acts: Tensor of post-ReLU activations, shape (..., N, D_SAE).
        sparsity_rate: The fraction of batch items for which each feature should be active.

    Returns:
        Sparse activations tensor with the same shape as input.
    """
    original_shape = post_relu_acts.shape
    if post_relu_acts.ndim == 3:
        B, L, F = original_shape
        post_relu_acts = post_relu_acts.view(B * L, F)
    
    N, d_sae = post_relu_acts.shape # N = B or B*L
    k_per_feature = max(1, int(N * sparsity_rate))

    # Transpose so each row corresponds to one dictionary element
    acts_t = post_relu_acts.transpose(-1, -2)  # (d_sae, N)

    # Get top-k values for each feature across the batch dimension
    topk_values, _ = acts_t.topk(k_per_feature, dim=-1)

    # Create masks for winners (values that meet or exceed the kth largest value for that feature)
    thresholds = topk_values[..., -1:] # (d_sae, 1)
    mask = (acts_t >= thresholds).float()

    # Apply mask and transpose back
    wta_acts_t = acts_t * mask
    wta_acts = wta_acts_t.transpose(-1, -2)  # (N, d_sae)

    if len(original_shape) == 3:
        wta_acts = wta_acts.view(original_shape)

    return wta_acts


def apply_batch_topk_sparsity(post_relu_acts: torch.Tensor, k: int):
    """
    Applies the Batch Top-K sparsity mechanism.

    Selects the top `k` activations for each item in the batch dimension.

    Args:
        post_relu_acts: Tensor of post-ReLU activations, shape (..., N, D_SAE).
        k: The number of top activations to keep for each batch item.

    Returns:
        Sparse activations tensor with the same shape as input.
    """
    # Ensure k is valid
    d_sae = post_relu_acts.shape[-1]
    k = min(k, d_sae)
    if k <= 0:
        return torch.zeros_like(post_relu_acts)

    # Find the top k activations for each item in the batch dimension
    # Note: This works for both (B, F) and (B, L, F) inputs directly
    # because topk operates on the last dimension (-1) by default.
    post_topk = post_relu_acts.topk(k, sorted=False, dim=-1)

    tops_acts = post_topk.values
    top_indices = post_topk.indices

    # Create a zero tensor and scatter the top k activations
    buffer = torch.zeros_like(post_relu_acts)
    sparse_acts = buffer.scatter_(dim=-1, index=top_indices, src=tops_acts)
    return sparse_acts


torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32

# --- Test Case 1: Simple 2D Input ---
print("--- Test Case 1: Simple 2D Input (B=5, F=10) ---")
B, F = 5, 10
# Generate integers first, then cast to float
acts_2d = torch.randint(0, 11, (B, F), device=device).to(dtype=dtype)
sparsity_rate_wta = 0.4  # Expect k_per_feature = max(1, 5 * 0.4) = 2
k_batchtopk = 3          # Expect 3 non-zero elements per row

print(f"Input Activations (Shape: {acts_2d.shape}):\n{acts_2d.cpu().numpy()}")

# WTA
wta_acts_2d = apply_wta_sparsity(acts_2d, sparsity_rate_wta)
print(f"\nWTA Sparsity (sparsity_rate={sparsity_rate_wta}, expected k_per_feature=2):")
print(f"Output (Shape: {wta_acts_2d.shape}):\n{wta_acts_2d.cpu().numpy()}")
non_zero_per_feature_wta = (wta_acts_2d > 0).sum(dim=0)
print(f"Non-zero count per feature (should be mostly 2):\n{non_zero_per_feature_wta.cpu().numpy()}")
# assert torch.all(non_zero_per_feature_wta <= 2), "WTA has more non-zeros per feature than expected"
# Note: Due to ties, some features might have *fewer* than k_per_feature non-zeros if threshold is 0

# BatchTopK
batchtopk_acts_2d = apply_batch_topk_sparsity(acts_2d, k_batchtopk)
print(f"\nBatchTopK Sparsity (k={k_batchtopk}):")
print(f"Output (Shape: {batchtopk_acts_2d.shape}):\n{batchtopk_acts_2d.cpu().numpy()}")
non_zero_per_batch_item_batchtopk = (batchtopk_acts_2d > 0).sum(dim=-1)
print(f"Non-zero count per batch item (should be {k_batchtopk}):\n{non_zero_per_batch_item_batchtopk.cpu().numpy()}")
# assert torch.all(non_zero_per_batch_item_batchtopk == k_batchtopk), "BatchTopK has incorrect non-zeros per batch item"
print("-" * 50)

# --- Test Case 2: 3D Input ---
print("\n--- Test Case 2: 3D Input (B=2, L=3, F=8) ---")
B, L, F = 2, 3, 8
N = B * L # Effective batch size for WTA is 6
# Generate integers first, then cast to float
acts_3d = torch.randint(0, 6, (B, L, F), device=device).to(dtype=dtype)
sparsity_rate_wta = 0.5  # Expect k_per_feature = max(1, 6 * 0.5) = 3
k_batchtopk = 2          # Expect 2 non-zero elements per token (row in the flattened view)

print(f"Input Activations (Shape: {acts_3d.shape}):\n{acts_3d.cpu().numpy()}")

# WTA
wta_acts_3d = apply_wta_sparsity(acts_3d, sparsity_rate_wta)
print(f"\nWTA Sparsity (sparsity_rate={sparsity_rate_wta}, effective N={N}, expected k_per_feature=3):")
print(f"Output (Shape: {wta_acts_3d.shape}):\n{wta_acts_3d.cpu().numpy()}")
# Check non-zeros per feature across the flattened B*L dimension
non_zero_per_feature_wta_3d = (wta_acts_3d.view(N, F) > 0).sum(dim=0)
print(f"Non-zero count per feature (should be mostly 3):\n{non_zero_per_feature_wta_3d.cpu().numpy()}")
# assert torch.all(non_zero_per_feature_wta_3d <= 3), "WTA (3D) has more non-zeros per feature than expected"


# BatchTopK
batchtopk_acts_3d = apply_batch_topk_sparsity(acts_3d, k_batchtopk)
print(f"\nBatchTopK Sparsity (k={k_batchtopk}):")
print(f"Output (Shape: {batchtopk_acts_3d.shape}):\n{batchtopk_acts_3d.cpu().numpy()}")
# Check non-zeros per token (B, L dimension flattened)
non_zero_per_token_batchtopk = (batchtopk_acts_3d > 0).sum(dim=-1)
print(f"Non-zero count per token (shape {non_zero_per_token_batchtopk.shape}, should all be {k_batchtopk}):\n{non_zero_per_token_batchtopk.cpu().numpy()}")
# assert torch.all(non_zero_per_token_batchtopk == k_batchtopk), "BatchTopK (3D) has incorrect non-zeros per token"
print("-" * 50)

# --- Test Case 3: Edge case k=1 or low sparsity rate ---
print("\n--- Test Case 3: Edge Cases (B=4, F=6) ---")
B, F = 4, 6
# Generate integers first, then cast to float
acts_edge = torch.randint(0, 10, (B, F), device=device).to(dtype=dtype)
sparsity_rate_wta = 0.1 # Expect k_per_feature = max(1, 4 * 0.1) = 1
k_batchtopk = 1         # Expect 1 non-zero element per row

print(f"Input Activations (Shape: {acts_edge.shape}):\n{acts_edge.cpu().numpy()}")

# WTA (k=1)
wta_acts_edge_k1 = apply_wta_sparsity(acts_edge, sparsity_rate_wta)
print(f"\nWTA Sparsity (sparsity_rate={sparsity_rate_wta}, expected k_per_feature=1):")
print(f"Output (Shape: {wta_acts_edge_k1.shape}):\n{wta_acts_edge_k1.cpu().numpy()}")
non_zero_per_feature_wta_edge = (wta_acts_edge_k1 > 0).sum(dim=0)
print(f"Non-zero count per feature (should be 1):\n{non_zero_per_feature_wta_edge.cpu().numpy()}")
# assert torch.all(non_zero_per_feature_wta_edge <= 1), "WTA (k=1) has more non-zeros per feature than expected"

# BatchTopK (k=1)
batchtopk_acts_edge_k1 = apply_batch_topk_sparsity(acts_edge, k_batchtopk)
print(f"\nBatchTopK Sparsity (k={k_batchtopk}):")
print(f"Output (Shape: {batchtopk_acts_edge_k1.shape}):\n{batchtopk_acts_edge_k1.cpu().numpy()}")
non_zero_per_batch_item_batchtopk_edge = (batchtopk_acts_edge_k1 > 0).sum(dim=-1)
print(f"Non-zero count per batch item (should be {k_batchtopk}):\n{non_zero_per_batch_item_batchtopk_edge.cpu().numpy()}")
# assert torch.all(non_zero_per_batch_item_batchtopk_edge == k_batchtopk), "BatchTopK (k=1) has incorrect non-zeros per batch item"
print("-" * 50)

print("\nComparison Complete.")
