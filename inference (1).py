"""
ASCAP Inference Engine
Extracts bottom-up region features from a PIL image using torchvision's
Faster R-CNN backbone, then runs the ASCAPTransformer to generate a caption.
"""

import os
import sys
import json
import math
import importlib.util

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import torchvision
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights


# ── load ASCAPTransformer from local models/ folder ──────────────────────────
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_BASE = os.path.dirname(os.path.abspath(__file__))
_transformer_mod = _load_module(
    "ascap_transformer",
    os.path.join(_BASE, "models", "ascap_transformer.py")
)
build_ascap_model  = _transformer_mod.build_ascap_model
ASCAPTransformer   = _transformer_mod.ASCAPTransformer


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extractor  (Faster R-CNN backbone → 2048-dim region features)
# ═══════════════════════════════════════════════════════════════════════════════
class BottomUpExtractor:
    """
    Uses torchvision Faster R-CNN (ResNet-50 FPN) to extract
    adaptive-k region features matching the training distribution.
    """

    def __init__(self, device, min_boxes=10, max_boxes=36):
        self.device    = device
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes

        # Load detection model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn(weights=weights)
        detector.eval()
        self.detector = detector.to(device)

        # Hook into the box_head to capture 2048-dim features
        self._features = []
        detector.roi_heads.box_head.register_forward_hook(self._hook)

        self.transform = T.Compose([T.ToTensor()])

    def _hook(self, module, input, output):
        self._features.append(output.detach().cpu())

    @torch.no_grad()
    def extract(self, pil_image):
        """
        Args:
            pil_image: PIL.Image (RGB)
        Returns:
            features: torch.FloatTensor  [num_boxes, 2048]
        """
        self._features.clear()
        img_tensor = self.transform(pil_image.convert("RGB")).to(self.device)

        # Run detector
        outputs = self.detector([img_tensor])
        scores  = outputs[0]['scores'].cpu()

        # Extract hooked features
        if not self._features:
            # Fallback: return zero features if hook didn't fire
            return torch.zeros(self.min_boxes, 2048)

        raw_feats = self._features[0]  # [num_proposals, 2048]

        # Filter by confidence and clamp to [min_boxes, max_boxes]
        num = raw_feats.shape[0]
        num = max(self.min_boxes, min(num, self.max_boxes))
        feats = raw_feats[:num]

        # Pad if needed
        if feats.shape[0] < self.min_boxes:
            pad   = torch.zeros(self.min_boxes - feats.shape[0], 2048)
            feats = torch.cat([feats, pad], dim=0)

        return feats.float()


# ═══════════════════════════════════════════════════════════════════════════════
# Caption Generator
# ═══════════════════════════════════════════════════════════════════════════════
class CaptionGenerator:
    def __init__(self, checkpoint_path, vocab_path, device=None):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )

        # ── Vocabulary ────────────────────────────────────────────────────────
        with open(vocab_path, 'r') as f:
            vocab_raw = json.load(f)

        self.word2idx = vocab_raw['word2idx']
        self.idx2word = vocab_raw['idx2word']   # keys are strings (JSON)
        self.vocab_size = len(self.word2idx)
        self.bos_idx    = self.word2idx.get('<BOS>', 1)
        self.eos_idx    = self.word2idx.get('<EOS>', 2)
        self.pad_idx    = self.word2idx.get('<PAD>', 0)
        self.unk_idx    = self.word2idx.get('<UNK>', 3)

        # ── Model ─────────────────────────────────────────────────────────────
        ckpt = torch.load(checkpoint_path, map_location='cpu')

        # Detect architecture from checkpoint
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        elif 'model' in ckpt:
            sd = ckpt['model']
        else:
            sd = ckpt

        # Strip DataParallel prefix
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

        # Infer architecture from checkpoint shapes
        n_enc   = sum(1 for k in sd if k.startswith('encoder_layers.') and k.endswith('.norm1.weight'))
        n_dec   = sum(1 for k in sd if k.startswith('decoder_layers.') and k.endswith('.norm1.weight'))
        max_len = sd['decoder_pos.pe'].shape[1]
        d_model = sd['decoder_pos.pe'].shape[2]

        self.model = build_ascap_model(
            vocab_size  = self.vocab_size,
            bos_idx     = self.bos_idx,
            padding_idx = self.pad_idx,
            d_model     = d_model,
            N_enc       = n_enc,
            N_dec       = n_dec,
            max_len     = max_len,
        )
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self.model.to(self.device)

        self.max_len = max_len

        # ── Feature Extractor ─────────────────────────────────────────────────
        self.extractor = BottomUpExtractor(self.device)

    @torch.no_grad()
    def generate(self, pil_image, max_new_tokens=None, beam_size=1):
        """
        Args:
            pil_image:      PIL.Image
            max_new_tokens: max tokens to generate (default: model max_len)
            beam_size:      1 = greedy, >1 = beam search
        Returns:
            caption: str
            tokens:  list[str]
        """
        max_steps = max_new_tokens or (self.max_len - 2)

        # Extract features
        feats = self.extractor.extract(pil_image)           # [K, 2048]
        feats = feats.unsqueeze(0).to(self.device)          # [1, K, 2048]

        # Encode
        enc_output = self.model.encode(feats)               # [1, K, d_model]

        if beam_size == 1:
            tokens = self._greedy(enc_output, max_steps)
        else:
            tokens = self._beam(enc_output, max_steps, beam_size)

        words   = [self.idx2word.get(str(t), '<unk>') for t in tokens]
        caption = ' '.join(w for w in words
                           if w not in ('<PAD>', '<BOS>', '<EOS>', '<UNK>', '<unk>'))
        # Capitalise first letter
        caption = caption.strip()
        if caption:
            caption = caption[0].upper() + caption[1:]
        return caption, words

    def _greedy(self, enc_output, max_steps):
        generated = torch.tensor([[self.bos_idx]], device=self.device)
        tokens = []
        for _ in range(max_steps):
            logprobs   = self.model.decode(generated, enc_output)
            next_token = logprobs[:, -1, :].argmax(dim=-1).item()
            if next_token == self.eos_idx:
                break
            tokens.append(next_token)
            generated = torch.cat(
                [generated, torch.tensor([[next_token]], device=self.device)], dim=1
            )
        return tokens

    def _beam(self, enc_output, max_steps, beam_size):
        """Simple beam search."""
        # Each beam: (score, token_list, generated_tensor)
        beams = [(0.0, [], torch.tensor([[self.bos_idx]], device=self.device))]
        completed = []

        for _ in range(max_steps):
            candidates = []
            for score, toks, gen in beams:
                logprobs = self.model.decode(gen, enc_output.expand(1, -1, -1))
                lp       = logprobs[0, -1, :]                  # [vocab]
                topk_lp, topk_idx = lp.topk(beam_size)

                for lp_val, idx in zip(topk_lp.tolist(), topk_idx.tolist()):
                    new_score = score + lp_val
                    new_toks  = toks + [idx]
                    new_gen   = torch.cat(
                        [gen, torch.tensor([[idx]], device=self.device)], dim=1
                    )
                    if idx == self.eos_idx:
                        completed.append((new_score / max(len(new_toks), 1), new_toks[:-1]))
                    else:
                        candidates.append((new_score, new_toks, new_gen))

            if not candidates:
                break
            candidates.sort(key=lambda x: x[0] / max(len(x[1]), 1), reverse=True)
            beams = candidates[:beam_size]

        if not completed:
            # Use best active beam
            completed = [(s / max(len(t), 1), t) for s, t, _ in beams]

        completed.sort(key=lambda x: x[0], reverse=True)
        return completed[0][1]
