import sys, importlib.util

# Load SCAP decoders and containers
_orig = sys.path.copy()
sys.path = [p for p in sys.path if 'working' not in p]
sys.path.insert(0, '/kaggle/input/datasets/musharaf5/scap-source-code/SCAP-main')
from models.core.decoders import SummaryForgetDecoderLayer, SummaryForgetDecoder
from models.containers import ModuleList
sys.path = _orig

# Load FiLMSiftingAttention from our folder by full path
_spec = importlib.util.spec_from_file_location('film_sifting', '/kaggle/working/models/film_sifting.py')
_mod  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
FiLMSiftingAttention = _mod.FiLMSiftingAttention


class ASCAPDecoderLayer(SummaryForgetDecoderLayer):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048,
                 dropout=0.1, self_att_module=None, enc_att_module=None,
                 self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super().__init__(d_model, d_k, d_v, h, d_ff, dropout,
                         self_att_module, enc_att_module,
                         self_att_module_kwargs, enc_att_module_kwargs)
        self.film_sifting = FiLMSiftingAttention(d_model, h, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self_att * mask_pad
        cross    = self.film_sifting(self_att, enc_output, mask_enc_att, mask_pad)
        ff       = self.pwff(cross) * mask_pad
        return ff


class ASCAPDecoder(SummaryForgetDecoder):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx,
                 d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=0.1,
                 self_att_module=None, enc_att_module=None,
                 self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super().__init__(vocab_size, max_len, N_dec, padding_idx,
                         d_model, d_k, d_v, h, d_ff, dropout,
                         self_att_module, enc_att_module,
                         self_att_module_kwargs, enc_att_module_kwargs)
        for i in range(N_dec):
            self.layers[i] = ASCAPDecoderLayer(
                d_model, d_k, d_v, h, d_ff, dropout,
                self_att_module, enc_att_module,
                self_att_module_kwargs, enc_att_module_kwargs
            )
