import json
import os
from functools import partial
from pathlib import Path
from torch.nn.attention.flex_attention import BlockMask

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from einops._torch_specific import allow_ops_in_compiled_graph
from ml_dtypes import bfloat16
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

allow_ops_in_compiled_graph()
flex_attention = torch.compile(flex_attention)


class MaskedAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, block_mask):
        q = self.wq(x)
        k = self.wk(x)
        v = self.wv(x)

        q = rearrange(q, "b s (h d) -> b h s d", h=self.num_heads)
        k = rearrange(k, "b s (h d) -> b h s d", h=self.num_heads)
        v = rearrange(v, "b s (h d) -> b h s d", h=self.num_heads)

        if block_mask is None:
            with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
                x = F.scaled_dot_product_attention(q, k, v)
        else:
            x = flex_attention(q, k, v, block_mask=block_mask)

        x = rearrange(x, "b h s d -> b s (h d)")
        x = self.wo(x)
        return x


class FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RelationalBlock(nn.Module):
    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
    ):
        super().__init__()

        self.norms = nn.ModuleDict(
            {l: nn.RMSNorm(d_model) for l in ["feat", "nbr", "col", "full", "ffn"]}
        )
        self.attns = nn.ModuleDict(
            {
                l: MaskedAttention(d_model, num_heads)
                for l in ["feat", "nbr", "col", "full"]
            }
        )
        self.ffn = FFN(d_model, d_ff)

    def forward(self, x, block_masks):
        for l in ["col", "feat", "nbr", "full"]:
            x = x + self.attns[l](self.norms[l](x), block_mask=block_masks[l])
        x = x + self.ffn(self.norms["ffn"](x))
        return x


def _make_block_mask(mask, batch_size, seq_len, device):
    def _mod(b, h, q_idx, kv_idx):
        return mask[b, q_idx, kv_idx]

    return create_block_mask(
        mask_mod=_mod,
        B=batch_size,
        H=None,
        Q_LEN=seq_len,
        KV_LEN=seq_len,
        device=device,
        _compile=True,
    )


class RelationalTransformer(nn.Module):
    def __init__(
        self,
        num_blocks,
        d_model,
        d_text,
        num_heads,
        d_ff,
        use_fk_contrastive=False,
        use_row_contrastive=False,
        contrastive_weight=0.1,
        contrastive_temperature=0.07,
    ):
        super().__init__()

        self.enc_dict = nn.ModuleDict(
            {
                "number": nn.Linear(1, d_model, bias=True),
                "text": nn.Linear(d_text, d_model, bias=True),
                "datetime": nn.Linear(1, d_model, bias=True),
                "col_name": nn.Linear(d_text, d_model, bias=True),
                "boolean": nn.Linear(1, d_model, bias=True),
            }
        )
        self.dec_dict = nn.ModuleDict(
            {
                "number": nn.Linear(d_model, 1, bias=True),
                "text": nn.Linear(d_model, d_text, bias=True),
                "datetime": nn.Linear(d_model, 1, bias=True),
                "boolean": nn.Linear(d_model, 1, bias=True),
            }
        )
        self.norm_dict = nn.ModuleDict(
            {
                "number": nn.RMSNorm(d_model),
                "text": nn.RMSNorm(d_model),
                "datetime": nn.RMSNorm(d_model),
                "col_name": nn.RMSNorm(d_model),
                "boolean": nn.RMSNorm(d_model),
            }
        )
        self.mask_embs = nn.ParameterDict(
            {
                t: nn.Parameter(torch.randn(d_model))
                for t in ["number", "text", "datetime", "boolean"]
            }
        )
        self.blocks = nn.ModuleList(
            [RelationalBlock(d_model, num_heads, d_ff) for i in range(num_blocks)]
        )
        self.norm_out = nn.RMSNorm(d_model)
        self.d_model = d_model
        
        # Contrastive learning parameters (shared weight and temperature)
        self.use_fk_contrastive = use_fk_contrastive
        self.use_row_contrastive = use_row_contrastive
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
    
    def fk_contrastive_loss(self, batch, x):
        """
        FK relationship based contrastive loss (optimized version).
        
        Positive pairs: Nodes connected by FK
        Negative pairs: Unconnected nodes
        
        Args:
            batch: Input batch dict
            x: Transformer output (B, S, d_model)
        
        Returns:
            contrast_loss: Scalar loss
        """
        node_idxs = batch["node_idxs"]  # (B, S)
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]  # (B, S, 5)
        is_padding = batch["is_padding"]  # (B, S)
        masks = batch["masks"]  # (B, S) - masked cells to predict
        
        B, S, D = x.shape
        device = x.device
        
        # Create valid mask: exclude padding AND masked cells
        valid_mask = ~is_padding & ~masks  # (B, S)
        
        # Flatten batch and sequence dimensions
        x_flat = x.view(B * S, D)  # (B*S, d_model)
        node_idxs_flat = node_idxs.view(-1)  # (B*S,)
        valid_mask_flat = valid_mask.view(-1)  # (B*S,)
        
        # Filter out padding
        valid_x = x_flat[valid_mask_flat]  # (V, d_model)
        valid_node_idxs = node_idxs_flat[valid_mask_flat]  # (V,)
        
        # Get unique node IDs and filter out padding (-1)
        unique_nodes, inverse_indices = torch.unique(valid_node_idxs, return_inverse=True)
        valid_node_mask = unique_nodes != -1
        unique_nodes = unique_nodes[valid_node_mask]
        
        num_nodes = unique_nodes.shape[0]
        if num_nodes < 2:
            return x.new_zeros(())
        
        # Map node IDs to indices using searchsorted (fully vectorized)
        # Sort unique_nodes for searchsorted
        sorted_nodes, sort_idx = torch.sort(unique_nodes)
        
        # Map each valid cell to its node index (in sorted order)
        node_mapping = torch.searchsorted(sorted_nodes, valid_node_idxs)
        
        # Handle out of bounds (nodes not in unique_nodes, though shouldn't happen)
        valid_cells_mask = (node_mapping < num_nodes) & (sorted_nodes[node_mapping] == valid_node_idxs)
        valid_x = valid_x[valid_cells_mask]
        node_mapping = sort_idx[node_mapping[valid_cells_mask]]  # Map back to original order
        
        # Compute node representations using scatter_add
        node_reprs = torch.zeros(num_nodes, D, device=device, dtype=x.dtype)
        node_counts = torch.zeros(num_nodes, device=device, dtype=x.dtype)
        
        node_reprs.scatter_add_(0, node_mapping.unsqueeze(1).expand(-1, D), valid_x)
        node_counts.scatter_add_(0, node_mapping, torch.ones_like(node_mapping, dtype=x.dtype))
        
        # Average pooling
        node_reprs = node_reprs / node_counts.unsqueeze(1).clamp(min=1)
        
        # Build neighbor adjacency matrix efficiently
        # Collect all neighbor relationships at once
        f2p_nbr_idxs_flat = f2p_nbr_idxs.view(B * S, -1)  # (B*S, 5)
        valid_nbr_idxs = f2p_nbr_idxs_flat[valid_mask_flat]  # (V, 5)
        valid_nbr_idxs = valid_nbr_idxs[valid_cells_mask]  # (V', 5)
        
        # Build positive mask - fully vectorized approach
        pos_mask = torch.zeros(num_nodes, num_nodes, dtype=torch.bool, device=device)
        
        # Get first cell index for each node using scatter
        # Create inverse mapping: for each cell, mark if it's the first cell of its node
        sorted_mapping, sort_indices = torch.sort(node_mapping)
        
        # Find first occurrence of each node_id
        unique_mapping = torch.unique_consecutive(sorted_mapping, return_inverse=False)
        first_cell_mask = torch.cat([
            torch.tensor([True], device=device),
            sorted_mapping[1:] != sorted_mapping[:-1]
        ])
        first_cell_per_node_unsorted = sort_indices[first_cell_mask]
        
        # Map back to node indices (first_cell_per_node_unsorted is in sorted order of node_mapping)
        # We need to create a mapping from node_idx to first_cell
        first_cell_per_node = torch.zeros(num_nodes, dtype=torch.long, device=device)
        node_ids_with_cells = sorted_mapping[first_cell_mask]
        first_cell_per_node[node_ids_with_cells] = first_cell_per_node_unsorted
        
        # Get neighbors for all nodes at once (N, 5)
        all_nbrs = valid_nbr_idxs[first_cell_per_node]  # (N, 5)
        
        # Vectorized neighbor matching using broadcasting
        # all_nbrs: (N, 5), unique_nodes: (N,)
        # Create comparison: (N, 5, N) where [i, j, k] = (all_nbrs[i, j] == unique_nodes[k])
        all_nbrs_expanded = all_nbrs.unsqueeze(2)  # (N, 5, 1)
        unique_nodes_expanded = unique_nodes.unsqueeze(0).unsqueeze(0)  # (1, 1, N)
        
        # matches[i, j, k] = True if node i's j-th neighbor equals unique_nodes[k]
        matches = (all_nbrs_expanded == unique_nodes_expanded)  # (N, 5, N)
        
        # Filter out -1 (padding) neighbors
        valid_nbr_mask = (all_nbrs != -1).unsqueeze(2)  # (N, 5, 1)
        matches = matches & valid_nbr_mask  # (N, 5, N)
        
        # Aggregate over neighbor dimension: pos_mask[i, k] = True if any neighbor of i equals unique_nodes[k]
        pos_mask = matches.any(dim=1)  # (N, N)
        
        # Make bidirectional
        pos_mask = pos_mask | pos_mask.t()
        
        # Remove self-connections
        pos_mask.fill_diagonal_(False)
        
        if not pos_mask.any():
            return x.new_zeros(())
        
        # Normalize representations
        node_reprs_norm = F.normalize(node_reprs, dim=-1)  # (N, d_model)
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(node_reprs_norm, node_reprs_norm.t())  # (N, N)
        sim_matrix = sim_matrix / self.contrastive_temperature
        
        # Fully vectorized InfoNCE loss computation
        has_positive = pos_mask.any(dim=1)  # (N,)
        if not has_positive.any():
            return x.new_zeros(())
        
        # Create mask for all samples except self
        eye_mask = torch.eye(num_nodes, dtype=torch.bool, device=device)  # (N, N)
        
        # For numerical stability, subtract max before exp
        sim_matrix_stable = sim_matrix - sim_matrix.max(dim=1, keepdim=True)[0]
        
        # Mask out self-similarities for denominator
        exp_sim = torch.exp(sim_matrix_stable)
        exp_sim_masked = exp_sim.masked_fill(eye_mask, 0.0)  # (N, N)
        
        # Denominator: sum over all non-self samples
        denominator = exp_sim_masked.sum(dim=1)  # (N,)
        
        # Numerator: sum over positive samples only
        exp_pos = exp_sim.masked_fill(~pos_mask, 0.0)  # (N, N)
        numerator = exp_pos.sum(dim=1)  # (N,)
        
        # InfoNCE loss: -log(numerator / denominator)
        # Only compute for nodes with positive pairs
        loss_per_node = -torch.log(numerator[has_positive] / denominator[has_positive] + 1e-8)
        
        return loss_per_node.mean()

    def row_contrastive_loss(self, batch, x):
        """
        Intra-Node Consistency Contrastive Loss (Row-level)
        
        Positive pairs: Cell token -> its node pooled embedding
        Negative pairs: Cell token -> other nodes' pooled embeddings
        
        Args:
            batch: Input batch dict
            x: Transformer output (B, S, d_model)
        
        Returns:
            contrast_loss: Scalar loss
        """
        node_idxs = batch["node_idxs"]        # (B, S)
        is_padding = batch["is_padding"]      # (B, S)
        masks = batch["masks"]                # (B, S) - masked cells to predict
        B, S, D = x.shape
        device = x.device

        # Extract valid tokens only: exclude padding AND masked cells
        valid = (~is_padding & ~masks).view(-1)        # (B*S,)
        h = x.view(B*S, D)[valid]                      # (V, D)
        nids = node_idxs.view(-1)[valid].long()        # (V,)

        # Remove padding nodes (-1)
        keep = nids != -1
        h, nids = h[keep], nids[keep]
        if h.numel() == 0:
            return x.new_zeros(())

        # Get unique nodes and node pooling (average)
        uniq, inv = torch.unique(nids, return_inverse=True)   # inv: (V,) in [0..N-1]
        N = uniq.size(0)
        if N < 2:
            return x.new_zeros(())

        # Average pooling for node representations
        node_sum = torch.zeros(N, D, device=device, dtype=h.dtype)
        node_cnt = torch.zeros(N, device=device, dtype=h.dtype)
        node_sum.scatter_add_(0, inv.unsqueeze(1).expand(-1, D), h)
        node_cnt.scatter_add_(0, inv, torch.ones_like(inv, dtype=h.dtype))
        z = node_sum / node_cnt.clamp(min=1).unsqueeze(1)     # (N, D)

        # Normalize cell and node representations
        q = F.normalize(h, dim=-1)           # (V, D) - cell representations
        k = F.normalize(z, dim=-1)           # (N, D) - node representations

        # Similarity matrix: each cell vs all nodes
        # sim[i, m] = q_i · k_m / tau
        sim = (q @ k.t()) / self.contrastive_temperature      # (V, N)

        # Labels: each cell's ground truth node index is inv
        # InfoNCE loss (cell-to-node classification)
        loss = F.cross_entropy(sim, inv, reduction='mean')
        return loss

    def forward(self, batch):
        node_idxs = batch["node_idxs"]
        f2p_nbr_idxs = batch["f2p_nbr_idxs"]
        col_name_idxs = batch["col_name_idxs"]
        table_name_idxs = batch["table_name_idxs"]
        is_padding = batch["is_padding"]
        batch_size, seq_len = node_idxs.shape

        batch_size, seq_len = node_idxs.shape
        device = node_idxs.device

        # Padding mask for attention pairs (allow only non-pad -> non-pad)
        pad = (~is_padding[:, :, None]) & (~is_padding[:, None, :])  # (B, S, S)

        # cells in the same node
        same_node = node_idxs[:, :, None] == node_idxs[:, None, :]  # (B, S, S)

        # kv index is among q's foreign -> primary neighbors
        kv_in_f2p = (node_idxs[:, None, :, None] == f2p_nbr_idxs[:, :, None, :]).any(
            -1
        )  # (B, S, S)

        # q index is among kv's primary -> foreign neighbors (reverse relation)
        q_in_f2p = (node_idxs[:, :, None, None] == f2p_nbr_idxs[:, None, :, :]).any(
            -1
        )  # (B, S, S)

        # Same column AND same table
        same_col_table = (col_name_idxs[:, :, None] == col_name_idxs[:, None, :]) & (
            table_name_idxs[:, :, None] == table_name_idxs[:, None, :]
        )  # (B, S, S)

        # Final boolean masks (apply padding once here)
        attn_masks = {
            "feat": (same_node | kv_in_f2p) & pad,
            "nbr": q_in_f2p & pad,
            "col": same_col_table & pad,
            "full": pad,
        }

        # Make them contiguous for better kernel performance
        for l in attn_masks:
            attn_masks[l] = attn_masks[l].contiguous()

        # Convert to block masks
        make_block_mask = partial(
            _make_block_mask,
            batch_size=batch_size,
            seq_len=seq_len,
            device=device,
        )
        block_masks = {
            l: make_block_mask(attn_mask) for l, attn_mask in attn_masks.items()
        }

        x = 0
        x = x + (
            self.norm_dict["col_name"](
                self.enc_dict["col_name"](batch["col_name_values"])
            )
            * (~is_padding)[..., None]
        )

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            x = x + (
                self.norm_dict[t](self.enc_dict[t](batch[t + "_values"]))
                * ((batch["sem_types"] == i) & ~batch["masks"] & ~is_padding)[..., None]
            )
            x = x + (
                self.mask_embs[t]
                * ((batch["sem_types"] == i) & batch["masks"] & ~is_padding)[..., None]
            )

        for i, block in enumerate(self.blocks):
            x = block(x, block_masks)

        x = self.norm_out(x)
        
        # Contrastive losses (can be used independently or together)
        contrast_loss = x.new_zeros(())
        if self.training:
            if self.use_fk_contrastive:
                contrast_loss = contrast_loss + self.fk_contrastive_loss(batch, x)
            if self.use_row_contrastive:
                contrast_loss = contrast_loss + self.row_contrastive_loss(batch, x)

        loss_out = x.new_zeros(())
        yhat_out = {"number": None, "text": None, "datetime": None, "boolean": None}

        B, S, _ = x.shape
        sem_types = batch["sem_types"]  # (B,S) ints 0..3
        masks = batch["masks"].bool()  # (B,S) where to train

        for i, t in enumerate(["number", "text", "datetime", "boolean"]):
            yhat = self.dec_dict[t](x)  # (B,S, D_t)
            y = batch[f"{t}_values"]  # (B,S, D_y)
            sem_type_mask = (sem_types == i) & masks  # (B,S) mask for this type

            if not sem_type_mask.any():
                if t in yhat_out:
                    # still touch the param to avoid unused param error
                    loss_out = loss_out + (yhat.sum() * 0.0)
                    yhat_out[t] = yhat
                continue

            if t in ("number", "datetime"):
                loss_t = F.huber_loss(yhat, y, reduction="none").mean(-1)
            elif t == "boolean":
                loss_t = F.binary_cross_entropy_with_logits(
                    yhat, (y > 0).float(), reduction="none"
                ).mean(-1)
            elif t == "text":
                raise ValueError("masking text not supported")

            # masked sum for this type
            loss_out = loss_out + (loss_t * sem_type_mask).sum()

            if t in yhat_out:
                yhat_out[t] = yhat

        loss_out = loss_out / masks.sum()
        
        # Add contrastive loss with weight
        total_loss = loss_out + self.contrastive_weight * contrast_loss

        return total_loss, yhat_out
