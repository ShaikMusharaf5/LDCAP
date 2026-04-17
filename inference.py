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
import torch.nn.functional as F
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
    "ldcap_transformer",
    os.path.join(_BASE, "models", "ldcap_transformer.py")
)
build_ascap_model  = _transformer_mod.build_ascap_model
ASCAPTransformer   = _transformer_mod.ASCAPTransformer


def _load_checkpoint(checkpoint_path):
    """
    Load model checkpoints robustly across PyTorch versions.

    PyTorch 2.6 switched `torch.load` to `weights_only=True` by default, which
    breaks older checkpoints that contain a small amount of NumPy metadata.
    We try the safe path first, then allowlist the specific NumPy scalar type
    reported by the error, and only fall back to full pickle loading if needed.
    """
    load_kwargs = {"map_location": "cpu"}

    try:
        return torch.load(checkpoint_path, weights_only=True, **load_kwargs)
    except Exception as first_error:
        numpy_scalar = None
        numpy_core = getattr(np, "_core", None) or getattr(np, "core", None)
        if numpy_core is not None:
            multiarray = getattr(numpy_core, "multiarray", None)
            if multiarray is not None:
                numpy_scalar = getattr(multiarray, "scalar", None)

        if numpy_scalar is not None and hasattr(torch.serialization, "safe_globals"):
            try:
                with torch.serialization.safe_globals([numpy_scalar]):
                    return torch.load(checkpoint_path, weights_only=True, **load_kwargs)
            except Exception:
                pass

        # Trusted local checkpoint fallback: older training artifacts can contain
        # objects that the weights-only loader will not accept.
        return torch.load(checkpoint_path, weights_only=False, **load_kwargs)


def _extract_state_dict(ckpt):
    """
    Normalize a few common checkpoint layouts into a plain state_dict.
    Raises a clear error when the checkpoint belongs to a different model stack.
    """
    if not isinstance(ckpt, dict):
        return ckpt

    if 'model_state_dict' in ckpt:
        return ckpt['model_state_dict']
    if 'model' in ckpt:
        return ckpt['model']
    if 'state_dict' in ckpt:
        return ckpt['state_dict']

    # Some training pipelines save multiple submodels in one checkpoint.
    if {'rin', 'ascap', 'vocab_size'}.issubset(ckpt.keys()):
        ascap_sd = ckpt['ascap']
        if isinstance(ascap_sd, dict):
            sample_keys = list(ascap_sd.keys())[:8]
            raise ValueError(
                "Unsupported checkpoint format: this file contains a combined "
                "RIN+ASCAP pipeline, not the standalone ASCAPTransformer used by "
                "this app. Sample keys: "
                + ", ".join(sample_keys)
            )

    return ckpt


# ═══════════════════════════════════════════════════════════════════════════════
# Feature Extractor  (Faster R-CNN backbone → 2048-dim region features)
# ═══════════════════════════════════════════════════════════════════════════════
class BottomUpExtractor:
    """
    Uses torchvision Faster R-CNN (ResNet-50 FPN) to extract
    adaptive-k region features matching the training distribution.
    """

    def __init__(self, device, expected_feature_dim, min_boxes=10, max_boxes=36, score_thresh=0.25):
        self.device    = device
        self.min_boxes = min_boxes
        self.max_boxes = max_boxes
        self.score_thresh = score_thresh
        self.feature_dim = expected_feature_dim

        # Load detection model
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        detector = fasterrcnn_resnet50_fpn(weights=weights)
        detector.eval()
        self.detector = detector.to(device)

        self.transform = T.Compose([T.ToTensor()])

    def _align_feature_dim(self, feats):
        current_dim = feats.shape[1]
        if current_dim == self.feature_dim:
            return feats
        if current_dim > self.feature_dim:
            return feats[:, :self.feature_dim]

        pad = torch.zeros(
            feats.shape[0],
            self.feature_dim - current_dim,
            dtype=feats.dtype,
        )
        return torch.cat([feats, pad], dim=1)

    def _zero_features(self):
        return torch.zeros(self.min_boxes, self.feature_dim, dtype=torch.float32)

    def _select_boxes(self, boxes, scores):
        if boxes.numel() == 0:
            return boxes

        order = scores.argsort(descending=True)
        if self.score_thresh is not None:
            confident = order[scores[order] >= self.score_thresh]
            if confident.numel() >= self.min_boxes:
                order = confident

        order = order[:self.max_boxes]
        if order.numel() < min(self.min_boxes, scores.numel()):
            order = scores.argsort(descending=True)[:min(self.max_boxes, scores.numel())]

        return boxes[order]

    def _project_box_features(self, box_features):
        head = self.detector.roi_heads.box_head
        if not hasattr(head, "fc6") or not hasattr(head, "fc7"):
            return self._align_feature_dim(head(box_features))

        flattened = box_features.flatten(start_dim=1)
        fc6_feats = F.relu(head.fc6(flattened))
        fc7_feats = F.relu(head.fc7(fc6_feats))

        if fc7_feats.shape[1] == self.feature_dim:
            return fc7_feats

        combined = torch.cat([fc6_feats, fc7_feats], dim=1)
        if combined.shape[1] == self.feature_dim:
            return combined

        return self._align_feature_dim(fc7_feats)

    @torch.no_grad()
    def extract(self, pil_image):
        """
        Args:
            pil_image: PIL.Image (RGB)
        Returns:
            features: torch.FloatTensor  [num_boxes, feature_dim]
        """
        img_tensor = self.transform(pil_image.convert("RGB")).to(self.device)
        images, _ = self.detector.transform([img_tensor], None)
        features = self.detector.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = {"0": features}

        proposals, _ = self.detector.rpn(images, features, None)
        detections, _ = self.detector.roi_heads(features, proposals, images.image_sizes, None)

        if not detections:
            return self._zero_features()

        boxes = detections[0].get("boxes")
        scores = detections[0].get("scores")
        if boxes is None or scores is None or boxes.numel() == 0:
            return self._zero_features()

        boxes = self._select_boxes(boxes, scores)
        if boxes.numel() == 0:
            return self._zero_features()

        # Recompute ROI-aligned features for the final selected boxes instead of
        # using raw proposal features, which are much noisier for captioning.
        box_features = self.detector.roi_heads.box_roi_pool(features, [boxes], images.image_sizes)
        raw_feats = self._project_box_features(box_features).detach().cpu()
        feats = raw_feats[:self.max_boxes]

        # Pad if needed
        if feats.shape[0] < self.min_boxes:
            pad = torch.zeros(
                self.min_boxes - feats.shape[0],
                self.feature_dim,
                dtype=feats.dtype,
            )
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

        if 'word2idx' in vocab_raw and 'idx2word' in vocab_raw:
            self.word2idx = vocab_raw['word2idx']
            self.idx2word = vocab_raw['idx2word']   # keys are strings (JSON)
        else:
            # Support flat vocab files shaped like {"token": index}.
            self.word2idx = vocab_raw
            self.idx2word = {str(idx): word for word, idx in self.word2idx.items()}

        self.vocab_size = len(self.word2idx)
        self.bos_idx    = self.word2idx.get('<BOS>', self.word2idx.get('<sos>', 1))
        self.eos_idx    = self.word2idx.get('<EOS>', self.word2idx.get('<eos>', 2))
        self.pad_idx    = self.word2idx.get('<PAD>', self.word2idx.get('<pad>', 0))
        self.unk_idx    = self.word2idx.get('<UNK>', self.word2idx.get('<unk>', 3))

        # ── Model ─────────────────────────────────────────────────────────────
        ckpt = _load_checkpoint(checkpoint_path)

        # Detect architecture from checkpoint
        sd = _extract_state_dict(ckpt)

        # Strip DataParallel prefix
        if any(k.startswith('module.') for k in sd.keys()):
            sd = {k.replace('module.', '', 1): v for k, v in sd.items()}

        # Infer architecture from checkpoint shapes
        if 'decoder_pos.pe' not in sd or 'feature_proj.0.weight' not in sd:
            sample_keys = ", ".join(list(sd.keys())[:8])
            raise ValueError(
                "Checkpoint architecture does not match models/ldcap_transformer.py. "
                "Expected keys like 'decoder_pos.pe' and 'feature_proj.0.weight'. "
                f"Found keys such as: {sample_keys}"
            )

        n_enc = sum(1 for k in sd if k.startswith('encoder_layers.') and k.endswith('.norm1.weight'))
        n_dec = sum(1 for k in sd if k.startswith('decoder_layers.') and k.endswith('.norm1.weight'))
        max_len = sd['decoder_pos.pe'].shape[1]
        d_model = sd['decoder_pos.pe'].shape[2]
        feature_dim = sd['feature_proj.0.weight'].shape[1]

        self.model = build_ascap_model(
            vocab_size  = self.vocab_size,
            bos_idx     = self.bos_idx,
            padding_idx = self.pad_idx,
            d_model     = d_model,
            d_in        = feature_dim,
            N_enc       = n_enc,
            N_dec       = n_dec,
            max_len     = max_len,
        )
        self.model.load_state_dict(sd, strict=True)
        self.model.eval()
        self.model.to(self.device)

        self.max_len = max_len

        # ── Feature Extractor ─────────────────────────────────────────────────
        self.extractor = BottomUpExtractor(self.device, expected_feature_dim=feature_dim)
        self.blocked_token_ids = {idx for idx in (self.bos_idx, self.pad_idx) if idx is not None}

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
        feats = self.extractor.extract(pil_image)           # [K, feature_dim]
        feats = feats.unsqueeze(0).to(self.device)          # [1, K, feature_dim]

        # Encode
        enc_output = self.model.encode(feats)               # [1, K, d_model]

        if beam_size == 1:
            tokens = self._greedy(enc_output, max_steps)
        else:
            tokens = self._beam(enc_output, max_steps, beam_size)

        words   = [self.idx2word.get(str(t), '<unk>') for t in tokens]
        special_tokens = {'<PAD>', '<BOS>', '<EOS>', '<UNK>', '<pad>', '<sos>', '<eos>', '<unk>'}
        caption = ' '.join(w for w in words if w not in special_tokens)
        # Capitalise first letter
        caption = caption.strip()
        if caption:
            caption = caption[0].upper() + caption[1:]
        return caption, words

    def _clean_logprobs(self, logprobs, tokens):
        adjusted = logprobs.clone()
        for idx in self.blocked_token_ids:
            adjusted[idx] = float('-inf')

        if self.unk_idx is not None and 0 <= self.unk_idx < adjusted.numel():
            adjusted[self.unk_idx] -= 2.0

        # Repetition penalty helps avoid loops like "a man man man".
        for idx in set(tokens[-3:]):
            if 0 <= idx < adjusted.numel():
                adjusted[idx] -= 1.2

        return adjusted

    def _greedy(self, enc_output, max_steps):
        generated = torch.tensor([[self.bos_idx]], device=self.device)
        tokens = []
        for _ in range(max_steps):
            logprobs = self.model.decode(generated, enc_output)
            next_step = self._clean_logprobs(logprobs[0, -1, :], tokens)
            next_token = next_step.argmax(dim=-1).item()
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
                lp = self._clean_logprobs(logprobs[0, -1, :], toks)
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
