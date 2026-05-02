import sys, importlib.util
import torch.nn.functional as F

# Load SCAP encoders (temporarily hide /kaggle/working to avoid models/ clash)
_orig = sys.path.copy()
sys.path = [p for p in sys.path if 'working' not in p]
sys.path.insert(0, '/kaggle/input/datasets/musharaf5/scap-source-code/SCAP-main')
from models.core.encoders import SummaryForgetEncoder, MultiLevelEncoder
sys.path = _orig

# Load RIN from our folder by full path
_spec = importlib.util.spec_from_file_location('rin', '/kaggle/working/models/rin.py')
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
RIN = _mod.RIN

class ASCAPEncoder(SummaryForgetEncoder):
    def __init__(self, N, padding_idx, d_in=2048, **kwargs):
        super().__init__(N, padding_idx, d_in, **kwargs)
        self.rin = RIN(d_model=self.d_model)

    def forward(self, input, attention_weights=None):
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = self.rin(out)
        return MultiLevelEncoder.forward(self, out, attention_weights=attention_weights)
