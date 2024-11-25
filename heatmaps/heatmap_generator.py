import torch
import matplotlib.pyplot as plt

# Initialize parameters
seq_len = 128
bsz = 1
n_heads = 1
device = "cpu"

# Simulate input boundaries and leading tokens
leading_tokens = 4

# boundaries = torch.tensor([256, 512, 768], device=device)  # Example chunk boundaries
# uniform chunk boundaries with gap of 8, starrting with 0 upto 1023
boundaries = torch.arange(0, seq_len, 4, device=device)
# Generate the is_chunk_boundary tensor
is_chunk_boundary = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
is_chunk_boundary[:, boundaries.int()] = True

# Compute t_abs and k_abs
t_abs = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1).expand(bsz, seq_len, 1)  # [B, L_q, 1]
k_abs = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(1).expand(bsz, 1, seq_len)   # [B, 1, L_k]

# Compute conditions
is_chunk_boundary_k = is_chunk_boundary.unsqueeze(1)  # [B, 1, L_k]
lower_bound = (t_abs - leading_tokens).clamp(min=0)

condition_1 = (k_abs < t_abs) & is_chunk_boundary_k  # First condition
condition_2 = (k_abs >= lower_bound) & (k_abs <= t_abs)  # Second condition
mask_condition = condition_1 | condition_2  # Combined condition
# condition 4 is just k_abs < t_abs
condition_4 = k_abs <= t_abs

# Compute the inverted condition
not_is_chunk_boundary_k = (~is_chunk_boundary).unsqueeze(1)  # Invert the is_chunk_boundary_k
condition_3 = (k_abs < t_abs) & not_is_chunk_boundary_k  # Third condition (inverted)

# Expand masks to [B, H, L_q, L_k]
condition_1 = condition_1.unsqueeze(1).expand(-1, n_heads, -1, -1)
condition_2 = condition_2.unsqueeze(1).expand(-1, n_heads, -1, -1)
mask_condition = mask_condition.unsqueeze(1).expand(-1, n_heads, -1, -1)
condition_3 = condition_3.unsqueeze(1).expand(-1, n_heads, -1, -1)
condition_4 = condition_4.unsqueeze(1).expand(-1, n_heads, -1, -1)

# Generate random attention weights and apply softmax
attn_weights = torch.randn(bsz, n_heads, seq_len, seq_len, device=device)  # Example random attention weights
attn_weights = attn_weights.masked_fill(mask_condition.logical_not(), float("-inf"))  # Apply combined mask
attn_probs = torch.softmax(attn_weights, dim=-1)

# Modify the heatmap with random values where the mask is active
def generate_heatmap(mask, filename):
    heatmap_tensor = attn_probs[0, 0, :, :].cpu().detach().float()  # Extract 1024x1024 tensor
    random_values = torch.empty_like(heatmap_tensor).uniform_(0.2, 1.0)
    heatmap_tensor[mask[0, 0, :, :].cpu()] = random_values[mask[0, 0, :, :].cpu()]
    heatmap_tensor[mask[0, 0, :, :].logical_not().cpu()] = 0  # Ensure mask is respected

    # Plot and save the heatmap
    plt.figure(figsize=(8, 8))
    plt.imshow(heatmap_tensor, cmap="viridis", aspect="auto")
    # Remove all ticks on x and y axes
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # Save the heatmap
    plt.savefig(f"{filename}.png", dpi=500)
    plt.close()

# Generate the rectangular heatmap based on condition_1
def generate_rectangular_heatmap(mask, filename):
    heatmap_tensor = attn_probs[0, 0, :, :].cpu().detach().float()  # Extract 1024x1024 tensor
    random_values = torch.empty_like(heatmap_tensor).uniform_(0.2, 1.0)
    heatmap_tensor[mask[0, 0, :, :].cpu()] = random_values[mask[0, 0, :, :].cpu()]
    heatmap_tensor[mask[0, 0, :, :].logical_not().cpu()] = 0  # Ensure mask is respected

    # Identify columns (x indices) that are entirely empty
    non_empty_columns = heatmap_tensor.sum(dim=0) > 0  # Sum along rows (y-axis)
    rectangular_heatmap = heatmap_tensor[:, non_empty_columns]  # Keep only non-empty columns

    # Plot and save the heatmap
    plt.figure(figsize=(2, 8))
    plt.imshow(rectangular_heatmap, cmap="viridis", aspect="auto")
    # Remove all ticks on x and y axes
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # Save the heatmap
    plt.savefig(f"{filename}.png", dpi=500)
    plt.close()

# Generate the shifted rectangular heatmap based on mask_condition
def generate_shifted_rectangular_heatmap(mask_condition, causality_mask, filename):
    heatmap_tensor = attn_probs[0, 0, :, :].cpu().detach().float()  # Extract tensor
    random_values = torch.empty_like(heatmap_tensor).uniform_(0.2, 1.0)
    heatmap_tensor[mask_condition[0, 0, :, :].cpu()] = random_values[mask_condition[0, 0, :, :].cpu()]
    heatmap_tensor[mask_condition[0, 0, :, :].logical_not().cpu()] = 0  # Ensure mask is respected

    # For each row, shift zeros to the right within causally valid positions
    for row in range(heatmap_tensor.shape[0]):
        row_values = heatmap_tensor[row, :].clone()
        valid_positions = causality_mask[0, 0, row, :].cpu()  # Causally valid positions
        non_zero_values = row_values[valid_positions].nonzero(as_tuple=False).squeeze()
        if len(non_zero_values.shape) > 0:  # If there are non-zero values
            shifted_values = row_values[non_zero_values]
            # Create a new row with shifted values
            new_row = torch.zeros_like(row_values)
            new_row[:shifted_values.shape[0]] = shifted_values
            heatmap_tensor[row, :] = new_row

    # Identify columns (x indices) that are entirely empty
    non_empty_columns = heatmap_tensor.sum(dim=0) > 0  # Sum along rows
    rectangular_heatmap = heatmap_tensor[:, non_empty_columns]  # Keep only non-empty columns

    # Plot and save the heatmap
    plt.figure(figsize=(2, 8))
    plt.imshow(rectangular_heatmap, cmap="viridis", aspect="auto")
    # Remove all ticks on x and y axes
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    # Save the heatmap
    plt.savefig(f"{filename}.png", dpi=500)
    plt.close()

# Generate the shifted rectangular heatmap for mask_condition
generate_shifted_rectangular_heatmap(mask_condition, condition_4, "heatmap_testtime_true")


# Generate the rectangular heatmap for condition_1
generate_rectangular_heatmap(condition_1, "heatmap_testtime")

# Generate the heatmaps
generate_heatmap(condition_1, "heatmap_chunks")
generate_heatmap(condition_2, "heatmap_swa")
generate_heatmap(mask_condition, "heatmap_swa_chunks")
generate_heatmap(condition_3, "heatmap_ssm")
generate_heatmap(condition_4, "heatmap_fullattn")