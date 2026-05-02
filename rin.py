import torch
from torch import nn

class RIN(nn.Module):
    """
    Residual Illumination Normalisation.
    Directly addresses the limitation stated in SCAP Section 6:
    challenging lighting conditions degrade visual feature quality.
    Residual design (rin_scale init=0.1) ensures normal-light images
    are unaffected at training start — correction grows only as needed.
    """
    def __init__(self, d_model=512):
        super(RIN, self).__init__()
        self.illum_mlp = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.ReLU(),
            nn.Linear(256, d_model)
        )
        self.gamma_fc  = nn.Linear(d_model, d_model)
        self.beta_fc   = nn.Linear(d_model, d_model)
        self.rin_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        illum_stats   = x.mean(dim=1)                      # [batch, d_model]
        illum_context = self.illum_mlp(illum_stats)        # [batch, d_model]
        gamma = self.gamma_fc(illum_context).unsqueeze(1)  # [batch, 1, d_model]
        beta  = self.beta_fc(illum_context).unsqueeze(1)   # [batch, 1, d_model]
        corrected = gamma * x + beta                       # [batch, seq_len, d_model]
        return x + self.rin_scale * corrected              # [batch, seq_len, d_model]
