import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DualTemporalAttention(nn.Module):
    """
    Dual Temporal Attention from BigD777 Paper.
    Computes attention for both current (t) and previous (t-1) timesteps in PARALLEL.
    Formula: 
        w^t = softmax(Q^t * K^t / sqrt(d_k))
        w^{t-1} = softmax(Q^{t-1} * K^{t-1} / sqrt(d_k))
        h'_i = [h̃_i^t ++ h̃_i^{t-1}]  (concatenation)
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead
        
        # Separate projections for current and previous queries (shared K, V)
        self.q_current = nn.Linear(d_model, d_model)
        self.q_previous = nn.Linear(d_model, d_model)
        self.k_shared = nn.Linear(d_model, d_model)
        self.v_shared = nn.Linear(d_model, d_model)
        
        self.out_proj = nn.Linear(d_model * 2, d_model)  # 2x because of concatenation
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
        # Feed Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, x, adj_matrix):
        """
        x: (Batch, Time, Nodes, Features)
        adj_matrix: (Nodes, Nodes) - for spatial neighbor aggregation
        Returns: (Batch, Time, Nodes, Features)
        """
        B, T, N, D = x.shape
        
        if T < 2:
            # Not enough history, fallback to standard attention
            return self._standard_attention(x, adj_matrix)
        
        # Split into current and previous timesteps
        x_current = x[:, -1:, :, :]  # (B, 1, N, F) - last timestep
        x_previous = x[:, -2:-1, :, :]  # (B, 1, N, F) - second to last
        
        # Compute Q, K, V
        Q_curr = self.q_current(x_current).view(B, 1, N, self.nhead, self.d_k).transpose(2, 3)  # (B, 1, nhead, N, d_k)
        Q_prev = self.q_previous(x_previous).view(B, 1, N, self.nhead, self.d_k).transpose(2, 3)
        
        # Shared Keys and Values from spatial neighbors
        K = self.k_shared(x).view(B, T, N, self.nhead, self.d_k).transpose(2, 3)  # (B, T, nhead, N, d_k)
        V = self.v_shared(x).view(B, T, N, self.nhead, self.d_k).transpose(2, 3)
        
        # Apply adjacency matrix for spatial aggregation (neighbor filtering)
        # Reshape adj for broadcasting: (1, 1, 1, N, N)
        adj_mask = adj_matrix.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # (1, 1, 1, N, N)
        
        # Attention for current timestep: w^t
        scores_curr = torch.matmul(Q_curr, K[:, -1:].transpose(-2, -1)) / np.sqrt(self.d_k)  # (B, 1, nhead, N, N)
        scores_curr = scores_curr.masked_fill(adj_mask == 0, float('-inf'))  # Apply adjacency mask
        attn_curr = F.softmax(scores_curr, dim=-1)
        attn_curr = self.dropout(attn_curr)
        h_curr = torch.matmul(attn_curr, V[:, -1:])  # (B, 1, nhead, N, d_k)
        
        # Attention for previous timestep: w^{t-1}
        scores_prev = torch.matmul(Q_prev, K[:, -2:-1].transpose(-2, -1)) / np.sqrt(self.d_k)
        scores_prev = scores_prev.masked_fill(adj_mask == 0, float('-inf'))
        attn_prev = F.softmax(scores_prev, dim=-1)
        attn_prev = self.dropout(attn_prev)
        h_prev = torch.matmul(attn_prev, V[:, -2:-1])  # (B, 1, nhead, N, d_k)
        
        # Reshape and concatenate: [h̃^t ++ h̃^{t-1}]
        h_curr = h_curr.transpose(2, 3).reshape(B, 1, N, self.d_model)  # (B, 1, N, d_model)
        h_prev = h_prev.transpose(2, 3).reshape(B, 1, N, self.d_model)
        h_concat = torch.cat([h_curr, h_prev], dim=-1)  # (B, 1, N, 2*d_model)
        
        # Project back to d_model
        h_out = self.out_proj(h_concat)  # (B, 1, N, d_model)
        
        # Residual connection
        h_out = self.norm(h_out + x_current)
        
        # Feed-forward network
        h_ffn = self.ffn(h_out)
        h_final = self.norm_ffn(h_out + self.dropout(h_ffn))
        
        return h_final.squeeze(1)  # (B, N, D)
    
    def _standard_attention(self, x, adj_matrix):
        """Fallback for when T < 2"""
        B, T, N, D = x.shape
        x_flat = x.reshape(B * T, N, D)
        Q = self.q_current(x_flat).view(B * T, N, self.nhead, self.d_k).transpose(1, 2)
        K = self.k_shared(x_flat).view(B * T, N, self.nhead, self.d_k).transpose(1, 2)
        V = self.v_shared(x_flat).view(B * T, N, self.nhead, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.d_k)
        adj_mask = adj_matrix.unsqueeze(0).unsqueeze(0)
        scores = scores.masked_fill(adj_mask == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        h = torch.matmul(attn, V).transpose(1, 2).reshape(B * T, N, D)
        h = self.out_proj(torch.cat([h, h], dim=-1))
        return (h.reshape(B, T, N, D)[:, -1, :, :] + x[:, -1, :, :])  # Residual


def compute_laplacian_positional_encoding(adj_matrix, k=10):
    """
    Compute Laplacian Positional Encoding as described in BigD777 paper.
    
    Args:
        adj_matrix: (N, N) adjacency matrix
        k: Number of smallest eigenvectors to use
    
    Returns:
        pos_enc: (N, k) positional encoding matrix
    """
    # Compute degree matrix
    D = torch.diag(adj_matrix.sum(dim=1))
    
    # Compute Laplacian: L = D - A
    L = D - adj_matrix
    
    # Normalized Laplacian: L_norm = I - D^{-1/2} A D^{-1/2}
    D_inv_sqrt = torch.diag(1.0 / torch.sqrt(D.diag() + 1e-8))
    L_norm = torch.eye(adj_matrix.size(0), device=adj_matrix.device) - torch.mm(torch.mm(D_inv_sqrt, adj_matrix), D_inv_sqrt)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
    
    # Take k smallest eigenvectors (excluding the first trivial one)
    pos_enc = eigenvectors[:, 1:k+1]  # (N, k)
    
    return pos_enc

class STGTLayer(nn.Module):
    """
    Spatio-Temporal Graph Transformer Layer (STGT) from BigD777 Paper.
    Processes spatial and temporal information using Dual Temporal Attention.
    """
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.dual_attn = DualTemporalAttention(d_model, nhead, dropout)
        
    def forward(self, x, adj_matrix):
        """
        x: (Batch, Time, Nodes, Features)
        adj_matrix: (Nodes, Nodes)
        Returns: (Batch, Nodes, Features)
        """
        return self.dual_attn(x, adj_matrix)

class STGTN(nn.Module):
    """
    Spatio-Temporal Graph Transformer Network (STGTN) - BigD777 Paper Implementation.
    
    Architecture Flow:
    1. Input Embedding + Laplacian Positional Encoding
    2. Stacked STGT Layers with Dual Temporal Attention (Parallel processing of t and t-1)
    3. Encoder MLP
    4. Decoder with Auto-regressive prediction
    
    Key Differences from Original Code:
    - Uses FIXED adjacency matrix (no learnable adaptive matrix)
    - Laplacian positional encoding instead of simple learned embeddings
    - Dual temporal attention (parallel t and t-1) instead of sequential T→S blocks
    - Encoder-decoder architecture for multi-step prediction
    """
    def __init__(self, num_nodes, in_dim=1, out_len=3, d_model=64, nhead=4, 
                 num_encoder_layers=2, num_decoder_layers=2, dropout=0.1, 
                 laplacian_k=10, device='cpu'):
        super().__init__()
        self.num_nodes = num_nodes
        self.out_len = out_len
        self.d_model = d_model
        self.device = device
        self.laplacian_k = laplacian_k
        
        # Input Projection
        self.input_emb = nn.Linear(in_dim, d_model)
        
        # Laplacian Positional Encoding (will be computed from adjacency matrix)
        self.pos_emb_proj = nn.Linear(laplacian_k, d_model)
        
        # Encoder: Stacked STGT Layers
        self.encoder_layers = nn.ModuleList([
            STGTLayer(d_model, nhead, dropout) 
            for _ in range(num_encoder_layers)
        ])
        
        # Encoder MLP
        self.encoder_mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model)
        )
        
        # Decoder: Stacked STGT Layers for auto-regressive prediction
        self.decoder_layers = nn.ModuleList([
            STGTLayer(d_model, nhead, dropout)
            for _ in range(num_decoder_layers)
        ])
        
        # Output Projection
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, in_dim)  # Predict back to original feature dimension
        )
        
        # Cache for Laplacian positional encoding
        self.register_buffer('laplacian_pos_enc', None)

    def forward(self, x, adj_matrix):
        """
        x: (Batch, Time, Nodes, Features)
        adj_matrix: (Nodes, Nodes) - Fixed adjacency matrix
        
        Returns: (Batch, out_len, Nodes, Features)
        """
        B, T, N, D = x.shape
        
        # Compute Laplacian Positional Encoding once
        if self.laplacian_pos_enc is None:
            self.laplacian_pos_enc = compute_laplacian_positional_encoding(
                adj_matrix, k=self.laplacian_k
            ).to(self.device)
        
        # 1. Input Embedding
        x_emb = self.input_emb(x)  # (B, T, N, d_model)
        
        # 2. Add Laplacian Positional Encoding
        pos_enc = self.pos_emb_proj(self.laplacian_pos_enc)  # (N, d_model)
        x_emb = x_emb + pos_enc.unsqueeze(0).unsqueeze(0)  # Broadcast to (B, T, N, d_model)
        
        # 3. Encoder: Process through STGT layers
        h = x_emb
        encoder_outputs = []
        for layer in self.encoder_layers:
            # Each layer processes the full sequence and outputs last timestep
            h_out = layer(h, adj_matrix)  # (B, N, d_model)
            encoder_outputs.append(h_out)
            # Update h by appending the new output as next timestep
            h = torch.cat([h[:, 1:, :, :], h_out.unsqueeze(1)], dim=1)  # Sliding window
        
        # 4. Encoder MLP
        h_encoded = self.encoder_mlp(encoder_outputs[-1])  # (B, N, d_model)
        
        # 5. Decoder: Auto-regressive prediction
        predictions = []
        h_decoder = h  # Initialize with encoder history
        
        for step in range(self.out_len):
            # Decode one step
            h_dec_out = h_decoder
            for dec_layer in self.decoder_layers:
                h_dec_out_step = dec_layer(h_dec_out, adj_matrix)  # (B, N, d_model)
                # Update decoder history
                h_dec_out = torch.cat([h_dec_out[:, 1:, :, :], h_dec_out_step.unsqueeze(1)], dim=1)
            
            # Predict next timestep
            pred = self.output_layer(h_dec_out_step)  # (B, N, 1)
            predictions.append(pred)
            
            # Update decoder input with prediction for next iteration
            # Need to embed prediction back to d_model space
            pred_emb = self.input_emb(pred)  # (B, N, d_model) - input_emb expects (*, 1)
            h_decoder = torch.cat([h_decoder[:, 1:, :, :], pred_emb.unsqueeze(1)], dim=1)
        
        # Stack predictions: (B, out_len, N, D)
        output = torch.stack(predictions, dim=1)
        
        return output
