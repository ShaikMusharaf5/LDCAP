import sys
sys.path.insert(0, '/kaggle/input/datasets/musharaf5/scap-source-code/SCAP-main')

import torch
from torch import nn

# Use full absolute path — avoids confusion between your models/ and SCAP models/
import importlib.util, os
_att_path = '/kaggle/input/datasets/musharaf5/scap-source-code/SCAP-main/models/core/attention.py'
_spec = importlib.util.spec_from_file_location('scap_attention', _att_path)
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
MultiHeadAttention = _mod.MultiHeadAttention

class FiLMSiftingAttention(nn.Module):
    """
    FiLM-conditioned cross-attention replacing the two enc_att calls
    in SummaryForgetDecoderLayer.

    Key design: Layer 1 FiLM params are conditioned on att_0 (updated
    text), not original text. This makes cross-modal interaction genuine
    rather than text-as-query-only as in base SCAP.
    """
    def __init__(self, d_model=512, h=8, dropout=0.1):
        super(FiLMSiftingAttention, self).__init__()
        d_k = d_model // h
        d_v = d_model // h
        self.film_gamma_0 = nn.Linear(d_model, d_model)
        self.film_beta_0  = nn.Linear(d_model, d_model)
        self.film_gamma_1 = nn.Linear(d_model, d_model)
        self.film_beta_1  = nn.Linear(d_model, d_model)
        self.enc_att = MultiHeadAttention(
            d_model, d_k, d_v, h,
            dropout=dropout,
            can_be_stateful=False
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, text_query, enc_output, mask_enc_att, mask_pad):
        # Layer 0 — FiLM from original text
        q0        = text_query.mean(dim=1, keepdim=True)
        gamma_0   = self.film_gamma_0(q0)
        beta_0    = self.film_beta_0(q0)
        vis_0     = enc_output[:, 0]
        vis_0_mod = gamma_0 * vis_0 + beta_0
        att_0     = self.enc_att(text_query, vis_0_mod, vis_0_mod, mask_enc_att)
        att_0     = att_0 * mask_pad

        # Layer 1 — FiLM from UPDATED text (att_0), not original
        q1        = att_0.mean(dim=1, keepdim=True)
        gamma_1   = self.film_gamma_1(q1)
        beta_1    = self.film_beta_1(q1)
        vis_1     = enc_output[:, 1]
        vis_1_mod = gamma_1 * vis_1 + beta_1
        att_1     = self.enc_att(att_0, vis_1_mod, vis_1_mod, mask_enc_att)
        att_1     = att_1 * mask_pad

        return self.layer_norm(att_1 + text_query)
