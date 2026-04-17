"""
ASCAP Transformer — Adaptive Sparse Caption Prediction
Encoder-Decoder transformer for image captioning with bottom-up attention features.

Encoder: projects 2048-dim region features → d_model, then N_enc transformer layers
Decoder: word embeddings + positional encoding, N_dec transformer layers with cross-attention
Output:  log-softmax over vocabulary
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ==============================================================================
# Positional Encoding (sinusoidal)
# ==============================================================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=200, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)                    # (max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (B, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


# ==============================================================================
# Multi-Head Attention (with explicit d_k, d_v)
# ==============================================================================
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, h, d_k, d_v, dropout=0.1):
        super().__init__()
        self.h = h
        self.d_k = d_k
        self.d_v = d_v

        self.W_q = nn.Linear(d_model, h * d_k)
        self.W_k = nn.Linear(d_model, h * d_k)
        self.W_v = nn.Linear(d_model, h * d_v)
        self.W_o = nn.Linear(h * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        B = query.size(0)

        # Project and reshape: (B, seq, h*d) → (B, h, seq, d)
        Q = self.W_q(query).view(B, -1, self.h, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(B, -1, self.h, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(B, -1, self.h, self.d_v).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = self.dropout(F.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)

        # Concat heads: (B, h, seq, d_v) → (B, seq, h*d_v)
        context = context.transpose(1, 2).contiguous().view(B, -1, self.h * self.d_v)
        return self.W_o(context)


# ==============================================================================
# Feed-Forward Network
# ==============================================================================
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))


# ==============================================================================
# Encoder Layer
# ==============================================================================
class EncoderLayer(nn.Module):
    def __init__(self, d_model, h, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, d_k, d_v, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention + residual
        attn_out = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Feed-forward + residual
        ff_out = self.ff(x)
        x = self.norm2(x + self.dropout2(ff_out))
        return x


# ==============================================================================
# Decoder Layer
# ==============================================================================
class DecoderLayer(nn.Module):
    def __init__(self, d_model, h, d_k, d_v, d_ff, dropout=0.1):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, h, d_k, d_v, dropout)
        self.cross_attn = MultiHeadAttention(d_model, h, d_k, d_v, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, self_mask=None, cross_mask=None):
        # Masked self-attention
        attn_out = self.self_attn(x, x, x, self_mask)
        x = self.norm1(x + self.dropout1(attn_out))
        # Cross-attention with encoder output
        cross_out = self.cross_attn(x, enc_output, enc_output, cross_mask)
        x = self.norm2(x + self.dropout2(cross_out))
        # Feed-forward
        ff_out = self.ff(x)
        x = self.norm3(x + self.dropout3(ff_out))
        return x


# ==============================================================================
# Full ASCAP Model
# ==============================================================================
class ASCAPTransformer(nn.Module):
    def __init__(self, vocab_size, bos_idx, padding_idx,
                 d_model=512, d_k=64, d_v=64, h=8,
                 d_ff=2048, d_in=2048, N_enc=3, N_dec=3,
                 max_len=100, dropout=0.1):
        super().__init__()

        self.padding_idx = padding_idx
        self.d_model = d_model

        # --- Encoder side ---
        # Project bottom-up features (2048-dim) to d_model
        self.feature_proj = nn.Sequential(
            nn.Linear(d_in, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        # Encoder max_len must cover max regions (up to ~100), not caption length
        self.encoder_pos = PositionalEncoding(d_model, max_len=200, dropout=dropout)
        self.encoder_layers = nn.ModuleList([
            EncoderLayer(d_model, h, d_k, d_v, d_ff, dropout)
            for _ in range(N_enc)
        ])

        # --- Decoder side ---
        self.word_emb = nn.Embedding(vocab_size, d_model, padding_idx=padding_idx)
        self.decoder_pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.decoder_layers = nn.ModuleList([
            DecoderLayer(d_model, h, d_k, d_v, d_ff, dropout)
            for _ in range(N_dec)
        ])

        # --- Output ---
        self.output_proj = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        # Zero out embedding for padding
        self.word_emb.weight.data[self.padding_idx].zero_()

    def _generate_causal_mask(self, seq_len, device):
        """Upper-triangular causal mask: prevents attending to future tokens."""
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = (mask == 0).unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, seq_len)
        return mask

    def _generate_padding_mask(self, seq, pad_idx):
        """Mask out padding positions: (B, 1, 1, seq_len)"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)

    def encode(self, features):
        """
        features: (B, num_regions, 2048)
        returns:  (B, num_regions, d_model)
        """
        x = self.feature_proj(features)
        x = self.encoder_pos(x)
        for layer in self.encoder_layers:
            x = layer(x)
        return x

    def decode(self, input_seq, enc_output):
        """
        input_seq:  (B, tgt_len)  — token indices
        enc_output: (B, src_len, d_model)
        returns:    (B, tgt_len, vocab_size) — log probabilities
        """
        tgt_len = input_seq.size(1)
        device = input_seq.device

        # Masks
        causal_mask = self._generate_causal_mask(tgt_len, device)
        tgt_pad_mask = self._generate_padding_mask(input_seq, self.padding_idx)
        self_mask = causal_mask & tgt_pad_mask  # combine causal + padding

        # Embeddings
        x = self.word_emb(input_seq) * math.sqrt(self.d_model)
        x = self.decoder_pos(x)

        for layer in self.decoder_layers:
            x = layer(x, enc_output, self_mask=self_mask)

        logits = self.output_proj(x)
        return F.log_softmax(logits, dim=-1)

    def forward(self, features, input_seq):
        """
        features:  (B, num_regions, 2048)
        input_seq: (B, tgt_len) — teacher-forced caption tokens
        returns:   (B, tgt_len, vocab_size) — log probabilities
        """
        enc_output = self.encode(features)
        return self.decode(input_seq, enc_output)


# ==============================================================================
# Builder function (called by train_xe.py)
# ==============================================================================
def build_ascap_model(vocab_size, bos_idx, padding_idx,
                      d_model=512, d_k=64, d_v=64, h=8,
                      d_ff=2048, d_in=2048, N_enc=3, N_dec=3,
                      max_len=100, dropout=0.1):
    model = ASCAPTransformer(
        vocab_size=vocab_size,
        bos_idx=bos_idx,
        padding_idx=padding_idx,
        d_model=d_model,
        d_k=d_k,
        d_v=d_v,
        h=h,
        d_ff=d_ff,
        d_in=d_in,
        N_enc=N_enc,
        N_dec=N_dec,
        max_len=max_len,
        dropout=dropout
    )

    # Count parameters
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"ASCAP Model built | Total params: {total:,} | Trainable: {trainable:,}")

    return model
