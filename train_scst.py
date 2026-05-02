"""
ASCAP Stage 2 — Self-Critical Sequence Training (SCST)
"""

import os
import sys
import json
import string
import math
import numpy as np
from collections import defaultdict
import importlib.util

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

# ============================================================
# DIRECT FILE IMPORTS (bypass Python module system)
# ============================================================

def load_module_from_file(module_name, file_path):
    """Load a Python module directly from file path."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

# Load modules directly from /kaggle/working/
config_module     = load_module_from_file("config",            "/kaggle/working/config.py")
dataset_module    = load_module_from_file("dataset",           "/kaggle/working/dataset.py")
transformer_module = load_module_from_file("ascap_transformer", "/kaggle/working/models/ascap_transformer.py")

# Extract what we need
ASCAPConfig        = config_module.ASCAPConfig
COCODataset        = dataset_module.COCODataset
collate_fn         = dataset_module.collate_fn
build_ascap_model  = transformer_module.build_ascap_model
ASCAPTransformer   = transformer_module.ASCAPTransformer

# CIDEr scorer
from pycocoevalcap.cider.cider import Cider

print("✅ All modules loaded successfully!")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_coco_references(karpathy_path):
    """Load ALL reference captions from Karpathy JSON."""
    with open(karpathy_path, 'r') as f:
        data = json.load(f)

    ref_dict = defaultdict(list)
    for img in data['images']:
        image_id = img['cocoid']
        for sent in img['sentences']:
            caption = sent['raw'].strip()
            ref_dict[image_id].append(caption)

    ref_dict = dict(ref_dict)
    print(f"📚 Loaded references for {len(ref_dict)} images")
    return ref_dict


def decode_sequences(seqs, idx2word, eos_idx):
    """Convert token indices to strings."""
    batch_size = seqs.size(0)
    results = []

    for i in range(batch_size):
        words = []
        for j in range(seqs.size(1)):
            idx = seqs[i, j].item()
            if idx == eos_idx:
                break
            word = idx2word.get(str(idx), idx2word.get(idx, '<unk>'))
            if word not in ['<pad>', '<bos>', '<eos>', '<start>', '<end>']:
                words.append(word)
        results.append(' '.join(words))

    return results


def compute_cider_batch(hyp_strs, ref_dict, image_ids):
    """Compute CIDEr scores for a batch."""
    gts = {}
    res = {}
    valid_ids = []

    for i, (hyp, img_id) in enumerate(zip(hyp_strs, image_ids)):
        img_id = int(img_id)
        if img_id in ref_dict:
            gts[img_id] = ref_dict[img_id]
            res[img_id] = [hyp]
            valid_ids.append(i)

    if len(valid_ids) == 0:
        return np.zeros(len(image_ids), dtype=np.float32)

    scorer = Cider()
    score, scores = scorer.compute_score(gts, res)

    result = np.zeros(len(image_ids), dtype=np.float32)
    scores_list = list(scores)
    for batch_idx, cider_score in zip(valid_ids, scores_list):
        result[batch_idx] = cider_score

    return result


# ============================================================
# GENERATION FUNCTIONS
# ============================================================

def sample_with_gradients(model, features, max_len, bos_idx, eos_idx):
    """Multinomial sampling WITH gradient tracking."""
    device = features.device
    batch_size = features.size(0)

    # ASCAPTransformer exposes .encode() and .decode() methods
    core = model.module if hasattr(model, 'module') else model

    enc_output = core.encode(features)   # (B, num_regions, d_model)
    generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)
    all_log_probs = []

    for t in range(max_len):
        logprobs = core.decode(generated, enc_output)   # (B, t+1, vocab)
        last_logprobs = logprobs[:, -1, :]

        probs = torch.exp(last_logprobs)
        probs = torch.clamp(probs, min=1e-8)
        sampled = torch.multinomial(probs, num_samples=1)

        lp = last_logprobs.gather(1, sampled)
        all_log_probs.append(lp)
        generated = torch.cat([generated, sampled], dim=1)

    sequences = generated[:, 1:]
    log_probs = torch.cat(all_log_probs, dim=1)

    return sequences, log_probs


def greedy_decode(model, features, max_len, bos_idx, eos_idx):
    """Argmax decoding, no gradients."""
    device = features.device
    batch_size = features.size(0)

    # ASCAPTransformer exposes .encode() and .decode() methods
    core = model.module if hasattr(model, 'module') else model

    with torch.no_grad():
        enc_output = core.encode(features)   # (B, num_regions, d_model)
        generated = torch.full((batch_size, 1), bos_idx, dtype=torch.long, device=device)

        for t in range(max_len):
            logprobs = core.decode(generated, enc_output)   # (B, t+1, vocab)
            last_logprobs = logprobs[:, -1, :]
            next_token = last_logprobs.argmax(dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)

        sequences = generated[:, 1:]

    return sequences


def validate_scst(model, val_loader, ref_dict, idx2word, max_len, bos_idx, eos_idx):
    """Greedy decode entire val set, compute mean CIDEr."""
    model.eval()
    all_cider_scores = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Validating"):
            features, _, image_ids = batch  # collate_fn returns (features, captions, image_ids)
            features = features.cuda()

            sequences   = greedy_decode(model, features, max_len, bos_idx, eos_idx)
            hyp_strs    = decode_sequences(sequences, idx2word, eos_idx)
            cider_scores = compute_cider_batch(hyp_strs, ref_dict, image_ids)
            all_cider_scores.extend(cider_scores.tolist())

    return np.mean(all_cider_scores)


# ============================================================
# DUAL-SAVE HELPER  (working dir + persistent Kaggle output)
# ============================================================

def save_checkpoint(payload, primary_path, persistent_dir=None):
    """
    Save checkpoint to primary_path.
    If persistent_dir is given, also copy there so files survive
    Kaggle session resets / kernel restarts.
    """
    import shutil
    # Atomic-style write: save to .tmp first, then rename
    tmp_path = primary_path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, primary_path)
    if persistent_dir:
        dest = os.path.join(persistent_dir, os.path.basename(primary_path))
        shutil.copy2(primary_path, dest)


# ============================================================
# MAIN TRAINING FUNCTION
# ============================================================

def train_scst(config):
    """Self-Critical Sequence Training (SCST) - Stage 2"""
    print("=" * 60)
    print("  ASCAP Stage 2 — Self-Critical Sequence Training")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️  Device: {device}")

    # --------------------------------------------------------
    # Config values (CORRECT attribute names)
    # --------------------------------------------------------
    checkpoint_dir   = config.CHECKPOINT_PATH       # fixed: was CHECKPOINT_DIR

    # --------------------------------------------------------
    # Ensure checkpoint directories exist (persistent + working)
    # --------------------------------------------------------
    os.makedirs(checkpoint_dir, exist_ok=True)
    # Also mirror to Kaggle's persistent output so files survive session resets
    persistent_dir = "/kaggle/output/checkpoints"
    os.makedirs(persistent_dir, exist_ok=True)
    print(f"📁 Checkpoint dir : {checkpoint_dir}")
    print(f"📁 Persistent dir : {persistent_dir}")
    feature_dir      = config.FEATURES_PATH         # fixed: was FEATURE_DIR
    karpathy_json    = config.KARPATHY_PATH         # fixed: was KARPATHY_JSON
    vocab_path       = config.VOCAB_PATH
    max_len          = config.MAX_LEN
    num_workers      = config.NUM_WORKERS
    batch_size_scst  = config.BATCH_SIZE_SCST
    lr_scst          = config.LEARNING_RATE_SCST
    epochs_scst      = config.NUM_EPOCHS_SCST       # fixed: was EPOCHS_SCST
    bos_idx          = config.BOS_IDX
    eos_idx          = config.EOS_IDX
    pad_idx          = config.PADDING_IDX           # fixed: was PAD_IDX

    # Generation length (use 20 if not specified)
    max_gen_len = getattr(config, 'MAX_GEN_LEN', 20)
    grad_clip   = getattr(config, 'GRAD_CLIP_SCST', 0.1)

    # --------------------------------------------------------
    # Load vocabulary
    # --------------------------------------------------------
    print("\n📖 Loading vocabulary...")
    with open(vocab_path, 'r') as f:
        vocab_raw = json.load(f)

    # vocab.json is nested: { "word2idx": {...}, "idx2word": {...} }
    word2idx = vocab_raw['word2idx']          # str→int
    idx2word = vocab_raw['idx2word']          # str→str  (keys are str since JSON)

    vocab_size = len(word2idx)
    print(f"   Vocabulary size: {vocab_size}")
    print(f"   BOS: {bos_idx}, EOS: {eos_idx}, PAD: {pad_idx}")

    # --------------------------------------------------------
    # Build model — pass ALL architecture params from config
    # so the model structure exactly matches the XE checkpoint
    # --------------------------------------------------------
    print("\n🏗️  Building model...")
    model = build_ascap_model(
        vocab_size  = vocab_size,
        bos_idx     = bos_idx,
        padding_idx = pad_idx,       # correct param name
        d_model     = config.D_MODEL,
        d_k         = config.D_K,
        d_v         = config.D_V,
        h           = config.N_HEADS, # correct param name
        d_ff        = config.D_FF,
        d_in        = config.D_IN,
        N_enc       = config.N_ENC,   # 2  — must match checkpoint
        N_dec       = config.N_DEC,   # 3  — must match checkpoint
        max_len     = config.MAX_LEN, # 54 — must match checkpoint
        dropout     = config.DROPOUT,
    )

    # Load XE checkpoint
    xe_path = os.path.join(checkpoint_dir, 'xe_best_model.pt')
    print(f"📂 Loading XE checkpoint from {xe_path}")
    checkpoint = torch.load(xe_path, map_location='cpu')

    if 'model_state_dict' in checkpoint:
        raw_state = checkpoint['model_state_dict']
    elif 'model' in checkpoint:
        raw_state = checkpoint['model']
    else:
        raw_state = checkpoint

    # Strip 'module.' prefix if checkpoint was saved with DataParallel
    if any(k.startswith('module.') for k in raw_state.keys()):
        print("   ⚠️  Detected DataParallel checkpoint — stripping 'module.' prefix")
        raw_state = {k.replace('module.', '', 1): v for k, v in raw_state.items()}

    model.load_state_dict(raw_state, strict=True)
    model = nn.DataParallel(model)
    model = model.to(device)
    print("   ✅ Model loaded!")

    # --------------------------------------------------------
    # Optimizer and Scaler
    # --------------------------------------------------------
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_scst)
    scaler    = GradScaler()

    # --------------------------------------------------------
    # Check for resume
    # --------------------------------------------------------
    scst_ckpt_path  = os.path.join(checkpoint_dir, 'scst_checkpoint.pt')
    best_model_path = os.path.join(checkpoint_dir, 'scst_best_model.pt')

    start_epoch = 0
    best_cider  = 0.0

    if os.path.exists(scst_ckpt_path):
        print(f"\n🔄 Resuming from {scst_ckpt_path}")
        ckpt = torch.load(scst_ckpt_path)
        model.module.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        if 'scaler_state_dict' in ckpt:
            scaler.load_state_dict(ckpt['scaler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_cider  = ckpt.get('best_cider', 0.0)
        print(f"   Resuming from epoch {start_epoch}, best CIDEr: {best_cider:.4f}")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    print("\n📊 Loading datasets...")

    # COCODataset signature: (split, config, word2idx)
    train_dataset = COCODataset(split='train', config=config, word2idx=word2idx)
    val_dataset   = COCODataset(split='val',   config=config, word2idx=word2idx)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size_scst, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True, collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset, batch_size=batch_size_scst, shuffle=False,
        num_workers=num_workers, pin_memory=True, collate_fn=collate_fn
    )

    print(f"   Train: {len(train_dataset)}, Val: {len(val_dataset)}, Batch: {batch_size_scst}")

    # --------------------------------------------------------
    # Load references
    # --------------------------------------------------------
    print("\n📚 Loading references...")
    ref_dict = load_coco_references(karpathy_json)

    # --------------------------------------------------------
    # Training Loop
    # --------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"  Starting SCST | Epochs: {epochs_scst}, LR: {lr_scst}, Clip: {grad_clip}")
    print("=" * 60)

    for epoch in range(start_epoch, epochs_scst):
        model.train()
        epoch_rewards = []
        epoch_losses  = []

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs_scst}")

        for batch in pbar:
            features, _, image_ids = batch  # collate_fn returns (features, captions, image_ids)
            features = features.cuda()

            with autocast():
                sample_seqs, sample_logprobs = sample_with_gradients(
                    model, features, max_gen_len, bos_idx, eos_idx
                )

            baseline_seqs = greedy_decode(model, features, max_gen_len, bos_idx, eos_idx)

            sample_strs   = decode_sequences(sample_seqs,   idx2word, eos_idx)
            baseline_strs = decode_sequences(baseline_seqs, idx2word, eos_idx)

            cider_sample   = compute_cider_batch(sample_strs,   ref_dict, image_ids)
            cider_baseline = compute_cider_batch(baseline_strs, ref_dict, image_ids)

            reward = torch.tensor(cider_sample - cider_baseline, dtype=torch.float32, device=device)

            with autocast():
                loss = -torch.mean(reward * sample_logprobs.sum(dim=1))

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
            scaler.step(optimizer)
            scaler.update()

            epoch_rewards.append(reward.mean().item())
            epoch_losses.append(loss.item())

            pbar.set_postfix({
                'loss':     f'{loss.item():.4f}',
                'reward':   f'{reward.mean().item():.4f}',
                'cider_s':  f'{cider_sample.mean():.3f}',
                'cider_b':  f'{cider_baseline.mean():.3f}'
            })

        # Epoch Summary
        print(f"\n📊 Epoch {epoch+1}: Reward={np.mean(epoch_rewards):.4f}, Loss={np.mean(epoch_losses):.4f}")

        # Validation
        print("\n🔍 Validating...")
        val_cider = validate_scst(model, val_loader, ref_dict, idx2word, max_gen_len, bos_idx, eos_idx)
        print(f"   Val CIDEr: {val_cider:.4f}")

        # Save best model
        if val_cider > best_cider:
            best_cider = val_cider
            save_checkpoint({
                'model_state_dict': model.module.state_dict(),
                'epoch':     epoch,
                'val_cider': val_cider
            }, best_model_path, persistent_dir)
            print(f"   ✅ New best! CIDEr: {best_cider:.4f}  (saved to working + output)")

        # Save checkpoint
        save_checkpoint({
            'model_state_dict':     model.module.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict':    scaler.state_dict(),
            'epoch':      epoch,
            'best_cider': best_cider
        }, scst_ckpt_path, persistent_dir)
        print(f"   💾 Checkpoint saved (working + output)")

    print("\n" + "=" * 60)
    print(f"  ✅ Prompt 5 PASSED — SCST complete")
    print(f"  Best CIDEr: {best_cider:.4f}")
    print("=" * 60)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    config = ASCAPConfig()

    # Set defaults if not in config
    if not hasattr(config, 'MAX_GEN_LEN'):
        config.MAX_GEN_LEN = 20
    if not hasattr(config, 'GRAD_CLIP_SCST'):
        config.GRAD_CLIP_SCST = 0.1

    train_scst(config)
