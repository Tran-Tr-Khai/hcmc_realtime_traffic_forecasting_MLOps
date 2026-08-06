"""Spatial-Temporal Graph Transformer Network (Dao, Zetsu & Hoang, 2024).

Implements the dual-attention STGT layer (Eq. 3-11) stacked into an
encoder-decoder that predicts future steps auto-regressively (Eq. 12-13),
using Laplacian eigenvector positional encoding for the graph.

Deviations from the paper, made for a tractable implementation:
- The encoder/decoder reuse a single STGT layer recurrently across time
  steps (like an RNN cell) instead of instantiating H distinct STGT layers,
  to keep the parameter count independent of the input window length.
- The feed-forward residual in Eq. 9-11 is added to the current-step
  attention output (not the raw 2*hidden concatenation) so the vertex
  representation stays at `hidden_dim` and can be fed back recurrently.
"""

import math

import numpy as np
import torch
from torch import nn


def build_adjacency_mask(adjacency: torch.Tensor) -> torch.Tensor:
    """Boolean neighbor mask (self-loops included) restricting attention to graph neighbors."""
    self_loops = torch.eye(adjacency.shape[0], dtype=torch.bool, device=adjacency.device)
    return (adjacency > 0) | self_loops


def compute_laplacian_positional_encoding(adjacency: np.ndarray, k: int) -> np.ndarray:
    """Laplacian eigenvector positional encoding used as `lambda` in Eq. 12."""
    adjacency = np.asarray(adjacency, dtype=np.float64)
    num_nodes = adjacency.shape[0]
    degree = adjacency.sum(axis=1)
    degree_inv_sqrt = np.power(np.clip(degree, 1e-6, None), -0.5)
    normalized_laplacian = np.eye(num_nodes) - degree_inv_sqrt[:, None] * adjacency * degree_inv_sqrt[None, :]

    _eigenvalues, eigenvectors = np.linalg.eigh(normalized_laplacian)
    k = max(min(k, num_nodes - 1), 0)
    # skip the trivial (near-zero) eigenvalue/eigenvector
    encoding = eigenvectors[:, 1 : k + 1]
    if encoding.shape[1] < k:
        encoding = np.pad(encoding, ((0, 0), (0, k - encoding.shape[1])))
    return encoding.astype(np.float32)


class DualGraphAttention(nn.Module):
    """Two parallel multi-head attention branches sharing keys/values (Eq. 3-8)."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

        self.key_proj = nn.Linear(hidden_dim, hidden_dim)
        self.value_proj = nn.Linear(hidden_dim, hidden_dim)
        self.query_current = nn.Linear(hidden_dim, hidden_dim)
        self.query_previous = nn.Linear(hidden_dim, hidden_dim)
        self.out_current = nn.Linear(hidden_dim, hidden_dim)
        self.out_previous = nn.Linear(hidden_dim, hidden_dim)
        self.norm_current = nn.LayerNorm(hidden_dim)
        self.norm_previous = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, num_nodes, _hidden_dim = x.shape
        return x.view(batch, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

    def _attend(
        self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        scores = scores.masked_fill(~mask, float("-inf"))
        weights = self.dropout(torch.softmax(scores, dim=-1))
        return torch.matmul(weights, value)

    def forward(
        self, h_current: torch.Tensor, h_previous: torch.Tensor, mask: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch, num_nodes, hidden_dim = h_current.shape

        key = self._split_heads(self.key_proj(h_current))
        value = self._split_heads(self.value_proj(h_current))

        attended_current = self._attend(self._split_heads(self.query_current(h_current)), key, value, mask)
        attended_current = attended_current.transpose(1, 2).reshape(batch, num_nodes, hidden_dim)
        current_out = self.norm_current(self.out_current(attended_current) + h_current)

        attended_previous = self._attend(self._split_heads(self.query_previous(h_previous)), key, value, mask)
        attended_previous = attended_previous.transpose(1, 2).reshape(batch, num_nodes, hidden_dim)
        previous_out = self.norm_previous(self.out_previous(attended_previous) + h_previous)

        return current_out, previous_out


class STGTLayer(nn.Module):
    """Spatial-temporal Graph Transformer layer with dual attention (Eq. 3-11)."""

    def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.attention = DualGraphAttention(hidden_dim, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, h_current: torch.Tensor, h_previous: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        current_out, previous_out = self.attention(h_current, h_previous, mask)
        concatenated = torch.cat([current_out, previous_out], dim=-1)
        return self.norm(self.feed_forward(concatenated) + current_out)


class InputEmbedding(nn.Module):
    """Linear embedding plus Laplacian positional encoding (Eq. 12)."""

    def __init__(self, input_dim: int, hidden_dim: int, pe_dim: int, dropout: float):
        super().__init__()
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.pe_proj = nn.Linear(pe_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, positional_encoding: torch.Tensor) -> torch.Tensor:
        embedded = self.value_proj(x) + self.pe_proj(positional_encoding).unsqueeze(0)
        return self.dropout(embedded)


class STGTN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_dim: int = 1,
        hidden_dim: int = 64,
        pe_dim: int = 8,
        num_heads: int = 4,
        output_len: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.output_len = output_len
        self.pe_dim = pe_dim

        self.encoder_embedding = InputEmbedding(input_dim, hidden_dim, pe_dim, dropout)
        self.encoder_layer = STGTLayer(hidden_dim, num_heads, dropout)
        self.encoder_refine = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))

        self.decoder_embedding = InputEmbedding(input_dim, hidden_dim, pe_dim, dropout)
        self.decoder_layer = STGTLayer(hidden_dim, num_heads, dropout)
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(
        self, x: torch.Tensor, adjacency_mask: torch.Tensor, positional_encoding: torch.Tensor
    ) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected input shape (batch, time, nodes, features), got {x.shape}")

        _batch_size, input_len, num_nodes, _features = x.shape
        if num_nodes != self.num_nodes:
            raise ValueError(f"Expected {self.num_nodes} nodes, got {num_nodes}")

        # first time step has no predecessor, so it skips the STGT layer (Eq. 12 base case)
        hidden = self.encoder_embedding(x[:, 0], positional_encoding)
        for step in range(1, input_len):
            step_input = self.encoder_embedding(x[:, step], positional_encoding)
            hidden = self.encoder_layer(step_input, hidden, adjacency_mask)
        hidden = self.encoder_refine(hidden)

        predictions = []
        decoder_input = x[:, -1]
        decoder_hidden = hidden
        for _ in range(self.output_len):
            step_input = self.decoder_embedding(decoder_input, positional_encoding)
            decoder_hidden = self.decoder_layer(step_input, decoder_hidden, adjacency_mask)
            prediction = self.output_head(decoder_hidden)
            predictions.append(prediction)
            decoder_input = prediction

        return torch.stack(predictions, dim=1)
