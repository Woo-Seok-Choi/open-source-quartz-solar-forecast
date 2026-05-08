"""nwp_transformer: PatchTST encoder + variable-level attention + RevIN.

Maps a 48 h window of 14 NWP variables plus a per-site kWp capacity scalar
to a 48 h × 30-min PV power forecast in raw kW.

Pipeline:
    1. RevIN on the 14 NWP channels (per-instance time-axis normalization).
    2. Channel-independent patch embedding (PatchTST): each NWP variable
       becomes its own patch-token sequence.
    3. Shared Transformer encoder across variables.
    4. Per-variable patch aggregator: flatten patches and project to d_model.
    5. Variable-level attention pools the 14 variable tokens into a single
       context vector using a learnable query.
    6. kWp is fused into the context via a separate ``log1p`` MLP — added
       *after* attention so that LayerNorm inside the encoder cannot erase
       the capacity scale.
    7. Linear prediction head -> raw kW for pred_len steps.

References (excerpted, not imported):
    - Solar-VLM model.py:460-525, 786-847 (variable attention + learnable query)
    - Solar-VLM layers/Embed.py:173 (PatchEmbedding)
    - Kim et al. 2022, "Reversible Instance Normalization" (RevIN)
"""

from __future__ import annotations

import math

import torch
from torch import nn


class _PositionalEmbedding(nn.Module):
    """Standard sinusoidal positional embedding."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pe[:, : x.size(1)]


class _PatchEmbedding(nn.Module):
    """Channel-independent patch embedding (PatchTST style)."""

    def __init__(
        self,
        d_model: int,
        patch_len: int,
        stride: int,
        padding: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = nn.ReplicationPad1d((0, padding))
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = _PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed channel-independent patches.

        Args:
            x: Input tensor of shape ``[B, n_vars, L]``.

        Returns:
            Tensor of shape ``[B * n_vars, n_patches, d_model]``.
        """
        x = self.padding_patch(x)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        x = x.reshape(x.size(0) * x.size(1), x.size(2), x.size(3))
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x)


class _RevIN(nn.Module):
    """Reversible instance normalization on the last dimension.

    Computes per-sample mean/std over the time dimension (``dim=1``) and
    normalizes ``num_features`` channels. Affine parameters are learnable.
    Only the ``normalize`` direction is used here; the model predicts power
    in raw kW so denormalization is unnecessary on the output side.
    """

    def __init__(
        self, num_features: int, eps: float = 1e-5, affine: bool = True,
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(num_features))
            self.bias = nn.Parameter(torch.zeros(num_features))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize along the time dimension.

        Args:
            x: Tensor of shape ``[B, L, num_features]``.

        Returns:
            Tensor of the same shape with each channel zero-mean / unit-var.
        """
        mean = x.mean(dim=1, keepdim=True)
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps)
        x = (x - mean) / std
        if self.affine:
            x = x * self.weight + self.bias
        return x


class NWPTransformer(nn.Module):
    """PatchTST + variable-level attention for NWP-only PV forecasting.

    Args:
        n_nwp_vars: Number of NWP variables (default 14).
        seq_len: Input window length in 30-min steps.
        pred_len: Forecast horizon in 30-min steps.
        d_model, n_heads, e_layers, dropout, patch_len, stride, padding:
            standard PatchTST hyperparameters.

    Shape:
        - ``x``: ``[B, seq_len, n_nwp_vars]`` (NWP only)
        - ``kwp``: ``[B]`` (per-site capacity in kW)
        - output: ``[B, pred_len]`` (raw kW)
    """

    def __init__(
        self,
        n_nwp_vars: int = 14,
        seq_len: int = 96,
        pred_len: int = 96,
        d_model: int = 128,
        n_heads: int = 8,
        e_layers: int = 3,
        dropout: float = 0.2,
        patch_len: int = 16,
        stride: int = 8,
        padding: int = 8,
    ) -> None:
        super().__init__()
        self.n_nwp_vars = n_nwp_vars
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.d_model = d_model

        self.revin = _RevIN(n_nwp_vars)
        self.patch_embedding = _PatchEmbedding(
            d_model, patch_len, stride, padding, dropout
        )
        # PatchTST patch count: ReplicationPad1d adds `padding` to the right
        # before unfold(size=patch_len, step=stride).
        self.n_patches = (seq_len + padding - patch_len) // stride + 1

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=e_layers)

        self.flatten = nn.Flatten(start_dim=-2)
        self.var_proj = nn.Linear(self.n_patches * d_model, d_model)

        # Variable-level attention with a single learnable query token.
        self.var_query = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.var_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.var_norm = nn.LayerNorm(d_model)

        # Capacity (kWp) conditioning: log1p MLP -> d_model embedding fused
        # post-attention so it survives the Transformer's per-token LayerNorm.
        self.capacity_mlp = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.fuse_norm = nn.LayerNorm(d_model)

        self.head = nn.Linear(d_model, pred_len)

    def forward(self, x: torch.Tensor, kwp: torch.Tensor) -> torch.Tensor:
        """Forecast 48 h × 30-min PV power.

        Args:
            x: NWP tensor of shape ``[B, seq_len, n_nwp_vars]``.
                RevIN-normalized in-place.
            kwp: Per-site capacity tensor of shape ``[B]`` (in kW).

        Returns:
            Tensor of shape ``[B, pred_len]`` with predicted kW.
        """
        b = x.size(0)
        x = self.revin.normalize(x)  # [B, L, n_nwp_vars]

        # Channel-independent: route variable axis into batch.
        x = x.transpose(1, 2)  # [B, n_nwp_vars, L]
        x = self.patch_embedding(x)  # [B*n_nwp_vars, n_patches, d_model]
        x = self.encoder(x)  # same shape

        # Per-variable patch aggregator.
        x = self.flatten(x)  # [B*n_nwp_vars, n_patches*d_model]
        x = self.var_proj(x)  # [B*n_nwp_vars, d_model]
        x = x.view(b, self.n_nwp_vars, self.d_model)

        # Variable-level attention: learnable query pools 14 variable tokens.
        query = self.var_query.expand(b, -1, -1)
        attn_out, _ = self.var_attn(query, x, x)
        ctx = self.var_norm(attn_out + query).squeeze(1)  # [B, d_model]

        # Fuse capacity AFTER attention (post-norm). log1p stabilizes scale
        # for kWp ranging roughly 1-10.
        cap = torch.log1p(kwp).unsqueeze(-1)  # [B, 1]
        cap_emb = self.capacity_mlp(cap)  # [B, d_model]
        ctx = self.fuse_norm(ctx + cap_emb)

        return self.head(ctx)  # [B, pred_len]
