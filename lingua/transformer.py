# Copyright (c) Meta Platforms, Inc. and affiliates.

from dataclasses import dataclass
from enum import Enum
import time
from typing import Optional, Union, Tuple

import torch
from torch import nn
from torch.nn import functional as F
from xformers.ops import fmha, AttentionBias
from torch.nn.attention.flex_attention import (
    BlockMask,
    flex_attention,
    _mask_mod_signature,
)

from lingua import probe
import random

flex_attention_comp = torch.compile(flex_attention)


class InitStdFactor(Enum):
    DISABLED = "disabled"  # Init std is divided by 1.0
    GLOBAL_DEPTH = "global_depth"  # Init std is divided by sqrt(2*n_layers)
    CURRENT_DEPTH = "current_depth"  # Init std is divided by sqrt(2*deoth)
    DIM_RATIO = "dim_ratio"  # Init std is divided by model_dim/4096


@dataclass
class BaseTransformerArgs:
    dim: int = 512
    attn_dim: int = 512 # Very customized, only reduces attention dimensions.
    n_layers: int = 8
    head_dim: Optional[int] = None
    n_heads: Optional[int] = None
    n_kv_heads: Optional[int] = None

    ffn_dim_multiplier: Optional[float] = None

    multiple_of: int = 256

    norm_eps: float = 1e-5

    rope_theta: float = 10000.0

    init_base_std: Optional[float] = None
    init_std_factor: str = "disabled"

    max_seqlen: int = 1024


def cross_entropy(pred, target, **kwargs):
    return F.nll_loss(
        F.log_softmax(pred.flatten(end_dim=-2).float(), -1),
        target.flatten(end_dim=-1),
        **kwargs,
    )


def repeat_kv(x: torch.Tensor, n_rep: int, dim: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=2, repeats=n_rep)"""
    assert dim == 2, "Only dim=2 is supported. Check the implementation for other dims."
    bs, slen, n_kv_heads, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, :, None, :]
        .expand(bs, slen, n_kv_heads, n_rep, head_dim)
        .reshape(bs, slen, n_kv_heads * n_rep, head_dim)
    )


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()

    cos, sin = freqs.cos(), freqs.sin()

    return torch.stack((cos, -sin, sin, cos), dim=-1).view(*freqs.size(), 2, 2)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor, seq_dim: int):
    """
    Reshape frequency tensor for broadcasting it with another tensor.

    This function reshapes the frequency tensor to have the same shape as the target tensor 'x'
    for the purpose of broadcasting the frequency tensor during element-wise operations.

    Args:
        freqs_cis (torch.Tensor): Frequency tensor to be reshaped.
        x (torch.Tensor): Target tensor for broadcasting compatibility.
        seq_dim (int): Sequence dimension index.

    Returns:
        torch.Tensor: Reshaped frequency tensor.
    """
    ndim = x.ndim
    assert 0 <= seq_dim < ndim
    assert freqs_cis.shape == (
        x.shape[seq_dim],
        x.shape[-3],
        2,
        2,
    ), f"freqs_cis vs x: {(freqs_cis.shape, x.shape)}"
    shape = [
        d if i == seq_dim or i == ndim - 3 else 1 for i, d in enumerate(x.shape[:-2])
    ] + [2, 2]
    return freqs_cis.view(*shape)


def apply_rotary_emb(
    xq: torch.Tensor,
    xk: torch.Tensor,
    seq_dim: int,
    freqs_cis: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    xq_ = xq.reshape(*xq.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    xk_ = xk.reshape(*xk.shape[:-1], -1, 1, 2)  # B S H D -> B S H D/2 1 2
    freqs_cis = reshape_for_broadcast(
        freqs_cis, xq_, seq_dim
    ).float()  # S D/2 2 2 -> 1 S 1 D/2 2 2
    xq_out = (xq_ * freqs_cis).sum(5).flatten(3)
    xk_out = (xk_ * freqs_cis).sum(5).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def causal_mask(b, h, q_idx, kv_idx):
    return q_idx >= kv_idx


def lengths_to_start_ids(lengths):
    doc_start = lengths.cumsum(0)
    doc_start = doc_start.roll(1)
    doc_start[0] = 0
    return doc_start


def lengths_to_local_ids(lengths):
    assert lengths.ndim == 1
    nb_seqs = lengths.size(0)
    total_seqlen = lengths.sum()
    # This gives the document id of each token
    doc_id = torch.repeat_interleave(lengths)
    # Compute document start for each document
    doc_start = lengths_to_start_ids(lengths)
    # Compute document start for each token
    doc_start = doc_start[doc_id]
    # Compute the position of each token within each document
    tok_id = torch.arange(total_seqlen, device=lengths.device) - doc_start

    return doc_id, tok_id


def generate_doc_mask_mod(
    mask_mod: _mask_mod_signature,
    lengths: torch.Tensor,
    kv_lengths: Optional[torch.Tensor] = None,
) -> _mask_mod_signature:
    """Generates mask mods that apply to inputs to flex attention in the sequence stacked
    format.

    Args:
        mask_mod: The mask mod to apply to the documents
        lengths: Lengths of each document

    Note:
        What is the sequence stacked format? When assembling batches of inputs, we
        take multiple sequences and stack them together to form 1 large sequence. We then
        use masking to ensure that the attention scores are only applied to tokens within
        the same document.

    Example:

    - Square mask
      doc_mask         lengths
      a a b b b c c    2 3 2
    a 1 0 0 0 0 0 0
    a 1 1 0 0 0 0 0
    b 0 0 1 0 0 0 0
    b 0 0 1 1 0 0 0
    b 0 0 1 1 1 0 0
    c 0 0 0 0 0 1 0
    c 0 0 0 0 0 1 1

    """
    kv_lengths = kv_lengths if kv_lengths is not None else lengths
    q_document_id, q_token_id = lengths_to_local_ids(lengths)
    kv_document_id, kv_token_id = lengths_to_local_ids(kv_lengths)
    q_max_idx = lengths.sum() - 1
    kv_max_idx = kv_lengths.sum() - 1

    def doc_mask_mod(b, h, q_idx, kv_idx):
        q_idx_cap = torch.minimum(q_max_idx, q_idx)
        kv_idx_cap = torch.minimum(kv_max_idx, kv_idx)
        valid_idx = (q_idx <= q_max_idx) & (kv_idx <= kv_max_idx)
        same_doc = q_document_id[q_idx_cap] == kv_document_id[kv_idx_cap]
        q_logical = q_token_id[q_idx_cap]
        kv_logical = kv_token_id[kv_idx_cap]
        inner_mask = mask_mod(b, h, q_logical, kv_logical)
        return same_doc & inner_mask & valid_idx

    return doc_mask_mod


# Rotary embedding as in xformer, see if torchtrain implementation is not better. Also might be usefull to make it work with batch*seqlen collapsed.
class RotaryEmbedding(torch.nn.Module):
    """
    RotaryEmbedding Module
    """

    def __init__(self, theta: float, head_dim: int, max_seqlen: int = 1024):
        super().__init__()

        self.theta = theta
        self.head_dim = head_dim
        self.max_seqlen = max_seqlen

        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(dim=head_dim, end=max_seqlen, theta=theta),
            persistent=False,
        )

    def reset_parameters(self):
        self.freqs_cis[...] = precompute_freqs_cis(
            dim=self.head_dim, end=self.max_seqlen, theta=self.theta
        )

    def forward(
        self, seqlen: Optional[int] = None, tok_idx: Optional[torch.Tensor] = None
    ):
        """
        Return freqs_cis corresponding to consecutive seqlen positions or the corresponding tok_idx positions
        Args:
            seqlen (int): Contiguous sequence length
            tok_idx (torch.Tensor[int]): Position indices of each token this overrides seqlen

        Returns:
            Tuple(torch.Tensor, torch.Tensor): Embedded input tensor and freqs_cis
        """
        test = (seqlen is not None) or (tok_idx is not None)
        assert test, "Should provide atleast seqlen or tok_idx"
        if tok_idx is not None:
            return self.freqs_cis[tok_idx]
        elif seqlen is not None:
            return self.freqs_cis[0:seqlen]


class RMSNorm(nn.Module):
    """
    Initialize the RMSNorm normalization layer.

    Args:
        dim (int): The dimension of the input tensor.
        eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

    Attributes:
        eps (float): A small value added to the denominator for numerical stability.
        weight (nn.Parameter): Learnable scaling parameter.

    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x: torch.Tensor):
        return x * torch.rsqrt((x * x).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        x = probe.log_stats(x, "resid")
        output = self._norm(x.float())
        return (output * self.weight.float()).type_as(x)

    def reset_parameters(self):
        torch.nn.init.ones_(self.weight)  # type: ignore



class AttentiveSSMNoProjCyc(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        n_heads: int,
        n_kv_heads: int,
        token_chunk: int,
        rope_theta: float,
        k_ssm: nn.Module,
        k_ssmnorm: nn.Module,
        v_ssm,
        v_ssmnorm,
        kv_pressm: bool = False,
        residual_ssm: bool = False,
        chunk_size: int = 256,
        pseudo_chunk: bool = False,
        keep_sink: bool = True,
        chunk_strat: str = "uniform",
        producer = None,
        additional_tokens=64,
        layer_idx=None,
        nlayers=None,
        keep_wproj=True,
        fattn_boundary="uniform", leading_tokens=1,
    ):
        super().__init__()
        
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.chunk_size = chunk_size
        self.token_chunk = token_chunk
        self.k_ssm = k_ssm
        self.k_ssmnorm = k_ssmnorm
        self.v_ssm = v_ssm
        self.v_ssmnorm = v_ssmnorm
        self.residual_ssm = residual_ssm
        self.pseudo_chunk = pseudo_chunk
        self.kv_pressm = kv_pressm
        self.keep_sink = keep_sink
        self.chunk_strat = chunk_strat
        self.leading_tokens = leading_tokens
        if self.chunk_strat == "head_cycle":
            self.chunk_strat = "uniform"
            self.get_rotated = True
        else:
            self.get_rotated = False
        self.additional_tokens = additional_tokens
        self.keep_wproj = keep_wproj
        self.fattn_boundary = fattn_boundary
        # If chunk_strat is first_attention and producer is None
        # then this later is the producer layer, we need to set
        # token ordering boundaries in rest of the layers appropriately
        self.producer = producer
        self.layer_idx = layer_idx
        self.nlayers = nlayers

        if self.chunk_strat in ["first_attention", "first_ssm"] and producer is None:
            self.eval_imp = True
        else:
            self.eval_imp = False

        self.rope_theta = rope_theta
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.max_seq_len = 16384 # safe upper bound
        self.precompute_boundaries()

        # Projection layers
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wo = nn.Linear(n_heads * head_dim, dim, bias=False)
        
    def process_chunks_with_ssm(self, x, ssm_module, tok_idx, cu_seqlens):
        bsz, seq_len, edim = x.shape
        device = x.device
        total_tokens = bsz * seq_len
        x_flat = x.view(1, total_tokens, edim)  # Shape: (1, total_tokens, edim)
        if hasattr(ssm_module, "cache"):
            del ssm_module.cache
        ssm_outputs = ssm_module(
            x_flat,          
            tok_idx=tok_idx, 
            cu_seqlens=cu_seqlens,
            ssm_impl="ssm"
        )  # Output shape: (1, total_tokens, edim)
        ssm_outputs = ssm_outputs.view(bsz, seq_len, edim)
        return ssm_outputs

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor],
        tok_idx: Optional[torch.Tensor] = None,
        ssm_tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        ssm_impl: str = "training",
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        bsz, seq_len, dim = x.shape
        L_q = seq_len
        n_heads = self.n_heads
        n_kv_heads = self.n_kv_heads
        head_dim = self.head_dim
        device = x.device
        K = self.token_chunk
        xq = self.wq(x)
        bsz, seq_len, edim = x.size()

        xk = x # No Key Projections
        xv = x # No Value Projections
        ssm_kv_tok_idx, cu_seqlens, boundaries = self.postcompute_boundaries(bsz, seq_len, dim, device, K)
        xk_processed = self.process_chunks_with_ssm(xk, self.k_ssm, ssm_kv_tok_idx, cu_seqlens)
        xv_processed = self.process_chunks_with_ssm(xv, self.v_ssm, ssm_kv_tok_idx, cu_seqlens)

        # always keep residual.
        xk_processed = xk_processed + xk
        xv_processed = xv_processed + xv

        xq = xq.view(bsz, seq_len, n_heads,    head_dim)
        xk_processed = xk_processed.view(bsz, -1, n_kv_heads, head_dim)
        xv_processed = xv_processed.view(bsz, -1, n_kv_heads, head_dim)
        xq, xk_processed = apply_rotary_emb(xq, xk_processed, 1, freq_cis)
        
        if hasattr(self, "kv_cache"):
            xk_processed, xv_processed = self.kv_cache.update(xk_processed, xv_processed, tok_idx)

        L_k = xk_processed.size(1)

        if self.pseudo_chunk:
            attn_mask = "causal"
            causality_mask = True
        else:
            is_chunk_boundary = torch.zeros(bsz, seq_len, dtype=torch.bool, device=device)
            is_chunk_boundary[:, boundaries.int()] = True
            t_abs = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(-1).expand(bsz, seq_len, 1)  # [B, L_q, 1]
            k_abs = torch.arange(seq_len, device=device).unsqueeze(0).unsqueeze(1).expand(bsz, 1, seq_len)   # [B, 1, L_k]
            is_chunk_boundary_k = is_chunk_boundary.unsqueeze(1)  # [B, 1, L_k]
            leading_tokens = self.leading_tokens
            lower_bound = (t_abs - leading_tokens).clamp(min=0)
            mask_condition = ( ((k_abs < t_abs) & is_chunk_boundary_k) | ((k_abs >= lower_bound) & (k_abs <= t_abs)) )
            attn_mask = mask_condition.unsqueeze(1)  # [B, 1, L_q, L_k]
            attn_mask = attn_mask.expand(-1, n_heads, -1, -1).contiguous()  # [B, H, L_q, L_k]
            causality_mask = False
        
        xk_processed = repeat_kv(xk_processed, self.heads_per_group, dim=2)  # [B, L, H_q, D]
        xv_processed = repeat_kv(xv_processed, self.heads_per_group, dim=2)  # [B, L, H_q, D]

        xq, xk_processed, xv_processed = map(lambda e: e.transpose(1, 2).contiguous(), (xq, xk_processed, xv_processed))

        # attn_weights = torch.einsum("bhqd,bhkd->bhqk", xq, xk_processed)
        # attn_weights = attn_weights / (head_dim ** 0.5)
        # attn_weights = attn_weights.masked_fill(attn_mask.logical_not(), float("-inf"))
        
        # import matplotlib.pyplot as plt
        # attn_probs = F.softmax(attn_weights[:, :, :, :], dim=-1)
        # heatmap_tensor = 2*attn_probs[0, 0, :, :].cpu().detach().float()
        # random_values = torch.empty_like(heatmap_tensor).uniform_(0.8, 1.0) 
        # heatmap_tensor[heatmap_tensor > 0] = random_values[heatmap_tensor > 0]
        # plt.figure(figsize=(10, 10))
        # plt.imshow(heatmap_tensor, cmap="viridis", aspect="auto")
        # plt.savefig("./attn_mask.png", dpi=500)
        # attn_mask = attn_mask if isinstance(attn_mask, torch.Tensor) else None
        
        # import pdb; pdb.set_trace()
        try:
            # ideally, this should be replaced by a custom implementation
            # which utilizes our form of chunked attention
            output = F.scaled_dot_product_attention(
                xq,
                xk_processed,
                xv_processed,
                attn_mask=attn_mask,
                is_causal=causality_mask
            )
        except Exception as e:
            import pdb; pdb.set_trace()
            
        output = output.transpose(1, 2).contiguous()  # [B, L_q, H_q, D]

        output = self.wo(output.view(bsz, seq_len, -1))  # [B, L_q, dim]

        return output

    def update_tokchunk(self, new_K):
        self.token_chunk = new_K
        self.precompute_boundaries()
        raise NotImplementedError("update_tokchunk not implemented for AttentiveSSMNoProjCyc")

    def postcompute_boundaries(self, bsz, seq_len, dim, device, K):
        # Adjust precomputed boundaries to actual sequence length
        boundaries = self.precomputed_boundaries[self.precomputed_boundaries < seq_len].to(device)
        boundaries[-1] = seq_len - 1 # Safe to always do this.

        # Adjust sequence starts and lengths
        sequence_starts = self.precomputed_sequence_starts[:boundaries.size(0)].to(device)
        sequence_lengths = self.precomputed_sequence_lengths[:boundaries.size(0)].to(device)

        # Broadcast to batch size
        total_boundaries = boundaries.unsqueeze(0) + (seq_len * torch.arange(bsz, device=device).unsqueeze(1))
        total_boundaries = total_boundaries.view(-1)
        sequence_starts = sequence_starts.unsqueeze(0) + (seq_len * torch.arange(bsz, device=device).unsqueeze(1))
        sequence_starts = sequence_starts.view(-1)
        sequence_lengths = sequence_lengths.repeat(bsz)

        # Compute positions and sequence IDs
        total_tokens = bsz * seq_len
        positions = torch.arange(total_tokens, device=device)
        sequence_ids = torch.bucketize(positions, total_boundaries)
        ssm_kv_tok_idx = positions - sequence_starts[sequence_ids]

        # Compute cumulative sequence lengths
        cu_seqlens = torch.cat([
            torch.tensor([0], device=device, dtype=torch.int32),
            torch.cumsum(sequence_lengths, dim=0)
        ])
        ssm_kv_tok_idx = ssm_kv_tok_idx.to(torch.int32).unsqueeze(0)

        return ssm_kv_tok_idx, cu_seqlens, boundaries

    def precompute_boundaries(self):
        K = self.token_chunk
        seq_len = self.max_seq_len
        device = torch.device('cpu')  # Use CPU for precomputations to save GPU memory
        boundaries_list = list(range(K - 1, seq_len, K))
        boundary_offset = min(K-1, int(self.layer_idx * (self.token_chunk // self.nlayers))) # 7 * 8 // 8 = 7, K = 8; 7 - 7 == 0
        boundaries_list = [b - boundary_offset for b in boundaries_list]
        if boundaries_list and boundaries_list[-1] != seq_len -1:
            boundaries_list.append(seq_len -1)
        if boundaries_list and boundaries_list[0] != 0:
            boundaries_list.insert(0, 0)
        self.precomputed_boundaries = torch.tensor(boundaries_list, device=device)
        if len(self.precomputed_boundaries) == 0:
            sequence_starts = torch.tensor([0], device=device)
            sequence_lengths = torch.tensor([seq_len], device=device)
        else:
            sequence_starts = torch.cat([
                torch.tensor([0], device=device), 
                self.precomputed_boundaries[:-1] + 1
            ])
            sequence_lengths = self.precomputed_boundaries - sequence_starts + 1
        self.precomputed_sequence_starts = sequence_starts
        self.precomputed_sequence_lengths = sequence_lengths
        self.precomputed_cu_seqlens = torch.cat([
            torch.tensor([0], device=device, dtype=torch.int32),
            torch.cumsum(sequence_lengths, dim=0)
        ])

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))
        initlist = [self.wq]
        for w in initlist:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

# class Attention(nn.Module):
#     def __init__(
#         self,
#         dim: int,
#         head_dim: int,
#         n_heads: int,
#         n_kv_heads: int,
#         rope_theta: float,
#     ):
#         super().__init__()

#         self.dim = dim
#         self.head_dim = head_dim
#         self.rope_theta = rope_theta

#         self.n_heads = n_heads
#         self.n_kv_heads = n_kv_heads
#         self.heads_per_group = self.n_heads // self.n_kv_heads

#         self.wq = nn.Linear(
#             dim,
#             n_heads * head_dim,
#             bias=False,
#         )
#         self.wk = nn.Linear(
#             dim,
#             n_kv_heads * head_dim,
#             bias=False,
#         )
#         self.wv = nn.Linear(
#             dim,
#             n_kv_heads * head_dim,
#             bias=False,
#         )

#         self.wo = nn.Linear(
#             n_heads * head_dim,
#             dim,
#             bias=False,
#         )

#     def forward(
#         self,
#         x: torch.Tensor,
#         freq_cis: torch.Tensor,
#         tok_idx: Optional[torch.Tensor] = None,
#         mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
#         attn_impl: str = "sdpa",
#     ) -> torch.Tensor:
#         # B S D
#         bsz, seq_len, dim = x.shape
#         xq = self.wq(x.view_as(x))
#         xk = self.wk(x.view_as(x))
#         xv = self.wv(x.view_as(x))

#         output_shape = xq.shape
#         # B S D -> B S H D
#         xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
#         xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
#         xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

#         xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

#         # This condition helps us be easily compatible
#         # with inference by adding a pluggable KVCache
#         if hasattr(self, "kv_cache"):
#             xk, xv = self.kv_cache.update(xk, xv, tok_idx)

#         xk = repeat_kv(xk, self.heads_per_group, dim=2)
#         xv = repeat_kv(xv, self.heads_per_group, dim=2)

#         if attn_impl == "flex_attention":
#             assert mask is None or isinstance(mask, BlockMask)
#             xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
#             output = flex_attention_comp(xq, xk, xv, block_mask=mask)
#             output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

#         elif attn_impl == "fmha":
#             assert mask is None or isinstance(mask, AttentionBias)
#             output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
#             # This uses B S H D instead of B H S D of pytorch

#         elif attn_impl == "sdpa":
#             xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
#             assert mask is None or isinstance(mask, (str, torch.Tensor))
#             # if mask is None or isinstance(mask, (str, torch.Tensor)):
#             #     import pdb; pdb.set_trace()
#             is_causal = (mask == "causal") if isinstance(mask, str) else False
#             mask = mask if isinstance(mask, torch.Tensor) else None
#             output = F.scaled_dot_product_attention(
#                 xq,
#                 xk,
#                 xv,
#                 is_causal=is_causal,
#                 attn_mask=mask,
#             )
#             output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
#         else:
#             raise NotImplementedError(
#                 f"Attention implementation {attn_impl} not supported"
#             )

#         output = self.wo(output.reshape(output_shape))

#         return output

class Attention(nn.Module):
    def __init__(
        self,
        dim: int,
        head_dim: int,
        attn_dim: int,
        n_heads: int,
        n_kv_heads: int,
        rope_theta: float,
    ):
        super().__init__()

        self.dim = dim
        self.head_dim = head_dim
        if attn_dim != dim:
            self.head_dim = attn_dim // n_heads
        self.rope_theta = rope_theta

        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.heads_per_group = self.n_heads // self.n_kv_heads

        self.wq = nn.Linear(
            dim,
            attn_dim,
            bias=False,
        )
        self.wk = nn.Linear(
            dim,
            attn_dim,
            bias=False,
        )
        self.wv = nn.Linear(
            dim,
            attn_dim,
            bias=False,
        )

        self.wo = nn.Linear(
            attn_dim,
            dim,
            bias=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:
        # B S D
        bsz, seq_len, dim = x.shape
        xq = self.wq(x.view_as(x))
        xk = self.wk(x.view_as(x))
        xv = self.wv(x.view_as(x))

        output_shape = xq.shape
        # B S D -> B S H D
        xq = xq.view(bsz, seq_len, self.n_heads, self.head_dim)
        xk = xk.view(bsz, seq_len, self.n_kv_heads, self.head_dim)
        xv = xv.view(bsz, seq_len, self.n_kv_heads, self.head_dim)

        xq, xk = apply_rotary_emb(xq, xk, 1, freq_cis[0:seq_len])

        # This condition helps us be easily compatible
        # with inference by adding a pluggable KVCache
        if hasattr(self, "kv_cache"):
            xk, xv = self.kv_cache.update(xk, xv, tok_idx)

        xk = repeat_kv(xk, self.heads_per_group, dim=2)
        xv = repeat_kv(xv, self.heads_per_group, dim=2)
        xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        # Hardcode mask to remove flex-attention non-compatibility with non-power of 2
        attn_mask = "causal"
        causality_mask = True
        attn_mask = attn_mask if isinstance(attn_mask, torch.Tensor) else None
        output = F.scaled_dot_product_attention(
            xq,
            xk,
            xv,
            attn_mask=attn_mask,
            is_causal=causality_mask
        )
        output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        # if attn_impl == "flex_attention":
        #     assert mask is None or isinstance(mask, BlockMask)
        #     xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        #     output = flex_attention_comp(xq, xk, xv, block_mask=mask)
        #     output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D

        # elif attn_impl == "fmha":
        #     assert mask is None or isinstance(mask, AttentionBias)
        #     output = fmha.memory_efficient_attention(xq, xk, xv, attn_bias=mask)
        #     # This uses B S H D instead of B H S D of pytorch

        # elif attn_impl == "sdpa":
        #     xq, xk, xv = map(lambda e: e.transpose(1, 2), (xq, xk, xv))
        #     assert mask is None or isinstance(mask, (str, torch.Tensor))
        #     # if mask is None or isinstance(mask, (str, torch.Tensor)):
        #     #     import pdb; pdb.set_trace()
        #     is_causal = (mask == "causal") if isinstance(mask, str) else False
        #     mask = mask if isinstance(mask, torch.Tensor) else None
        #     output = F.scaled_dot_product_attention(
        #         xq,
        #         xk,
        #         xv,
        #         is_causal=is_causal,
        #         attn_mask=mask,
        #     )
        #     output = output.transpose(1, 2).contiguous()  # B H S D -> B S H D
        # else:
        #     raise NotImplementedError(
        #         f"Attention implementation {attn_impl} not supported"
        #     )

        output = self.wo(output.reshape(output_shape))

        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        init_std = init_std or (self.dim ** (-0.5))

        for w in [self.wq, self.wk, self.wv]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=init_std,
                a=-3 * init_std,
                b=3 * init_std,
            )

        nn.init.trunc_normal_(
            self.wo.weight,
            mean=0.0,
            std=init_std / factor,
            a=-3 * init_std,
            b=3 * init_std,
        )

class FeedForward(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        multiple_of: int,
        ffn_dim_multiplier: Optional[float],
        mp_size: int = 1,
    ):
        super().__init__()

        hidden_dim = int(2 * hidden_dim / 3)
        if ffn_dim_multiplier is not None:
            hidden_dim = int(ffn_dim_multiplier * hidden_dim)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
        assert hidden_dim % mp_size == 0

        self.dim = dim
        self.hidden_dim = hidden_dim

        self.w1 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w3 = nn.Linear(
            dim,
            hidden_dim,
            bias=False,
        )
        self.w2 = nn.Linear(
            hidden_dim,
            dim,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B S D
        x1 = self.w1(x.view_as(x))
        x3 = self.w3(x.view_as(x))
        output = self.w2(F.silu(x1) * x3)
        return output

    def reset_parameters(self, init_std=None, factor=1.0):
        in_init_std = init_std or (self.dim ** (-0.5))
        out_init_std = init_std or (self.hidden_dim ** (-0.5))
        in_init_std = in_init_std
        out_init_std = out_init_std / factor
        for w in [self.w1, self.w3]:
            nn.init.trunc_normal_(
                w.weight,
                mean=0.0,
                std=in_init_std,
                a=-3 * in_init_std,
                b=3 * in_init_std,
            )
        nn.init.trunc_normal_(
            self.w2.weight,
            mean=0.0,
            std=out_init_std,
            a=-3 * out_init_std,
            b=3 * out_init_std,
        )


class TransformerBlock(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()

        assert (args.head_dim is not None) or (
            args.n_heads is not None
        ), "Should specify at least head_dim or n_heads"
        self.head_dim = args.head_dim or args.dim // args.n_heads
        self.n_heads = args.n_heads or args.dim // args.head_dim
        self.n_kv_heads = args.n_kv_heads or self.n_heads
        self.attn_dim = args.attn_dim

        assert args.n_heads % self.n_kv_heads == 0
        assert args.dim % args.n_heads == 0

        self.attention = Attention(
            dim=args.dim,
            head_dim=self.head_dim,
            attn_dim=self.attn_dim,
            n_heads=self.n_heads,
            n_kv_heads=self.n_kv_heads,
            rope_theta=args.rope_theta,
        )
        self.feed_forward = FeedForward(
            dim=args.dim,
            hidden_dim=4 * args.dim,
            multiple_of=args.multiple_of,
            ffn_dim_multiplier=args.ffn_dim_multiplier,
        )
        self.attention_norm = RMSNorm(args.dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.dim, eps=args.norm_eps)

    def forward(
        self,
        x: torch.Tensor,
        freq_cis: torch.Tensor,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ) -> torch.Tensor:

        h = x + self.attention(
            self.attention_norm(x),
            freq_cis,
            tok_idx=tok_idx,
            mask=mask,
            attn_impl=attn_impl,
        )
        out = h + self.feed_forward(self.ffn_norm(h))
        return out

    def init_weights(self, init_std=None, factor=1.0):
        self.attention.reset_parameters(init_std, factor)
        self.attention_norm.reset_parameters()

        self.feed_forward.reset_parameters(init_std, factor)
        self.ffn_norm.reset_parameters()


class BaseTransformer(nn.Module):
    def __init__(self, args: BaseTransformerArgs):
        super().__init__()
        self.dim = args.dim
        self.init_base_std = args.init_base_std
        self.init_std_factor = InitStdFactor(args.init_std_factor)
        self.max_seqlen = args.max_seqlen
        head_dim = args.head_dim or args.dim // args.n_heads
        if args.attn_dim:
            head_dim = args.attn_dim // args.n_heads
        self.rope_embeddings = RotaryEmbedding(
            theta=args.rope_theta,
            head_dim=head_dim,
            max_seqlen=args.max_seqlen,
        )

        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))

    def forward(
        self,
        h,
        tok_idx: Optional[torch.Tensor] = None,
        mask: Optional[Union[BlockMask, AttentionBias, str]] = None,
        attn_impl: str = "sdpa",
    ):

        freq_cis = self.rope_embeddings(seqlen=self.max_seqlen, tok_idx=tok_idx)

        for i, layer in enumerate(self.layers):
            h = layer(h, freq_cis, tok_idx=tok_idx, mask=mask, attn_impl=attn_impl)
        return h

    def reset_parameters(self):
        # Either use fixed base std or sqrt model dim
        self.rope_embeddings.reset_parameters()

    def init_weights(self):
        self.reset_parameters()
        for depth, layer in enumerate(self.layers):
            factor = {
                InitStdFactor.CURRENT_DEPTH: (2 * (depth + 1)) ** 0.5,
                InitStdFactor.GLOBAL_DEPTH: (2 * (len(self.layers) + 1)) ** 0.5,
                InitStdFactor.DIM_RATIO: self.dim / 4096,
                InitStdFactor.DISABLED: 1.0,
            }[self.init_std_factor]

            layer.init_weights(self.init_base_std, factor)
