import os, math, torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from config import ASCAPConfig
from dataset import build_vocab, COCODataset, collate_fn
from models.ascap_transformer import build_ascap_model
from checkpoint_manager import (save_checkpoint, load_latest_checkpoint,
                                 clean_old_checkpoints)


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scaler,
                    scheduler, config, epoch):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    pbar = tqdm(loader, desc=f"XE Epoch {epoch+1}/{config.NUM_EPOCHS_XE}")
    for batch_idx, (features, captions, _) in enumerate(pbar):
        features   = features.cuda()
        captions   = captions.cuda()
        input_seq  = captions[:, :-1]   # teacher forcing input
        target_seq = captions[:, 1:]    # prediction target

        with autocast(enabled=config.MIXED_PRECISION):
            output = model(features, input_seq)
            # output: [B, seq_len-1, vocab_size]
            loss = F.nll_loss(
                output.reshape(-1, config.VOCAB_SIZE),
                target_seq.reshape(-1),
                ignore_index=config.PADDING_IDX
            ) / config.GRADIENT_ACCUMULATION

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.GRADIENT_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

        total_loss += loss.item() * config.GRADIENT_ACCUMULATION
        avg = total_loss / (batch_idx + 1)
        pbar.set_postfix({
            'loss': f'{avg:.4f}',
            'lr':   f'{scheduler.get_last_lr()[0]:.2e}'
        })

    return total_loss / len(loader)


def validate_one_epoch(model, loader, config, epoch):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for features, captions, _ in tqdm(
                loader, desc=f"Val  Epoch {epoch+1}/{config.NUM_EPOCHS_XE}"):
            features   = features.cuda()
            captions   = captions.cuda()
            input_seq  = captions[:, :-1]
            target_seq = captions[:, 1:]
            with autocast(enabled=config.MIXED_PRECISION):
                output = model(features, input_seq)
                loss   = F.nll_loss(
                    output.reshape(-1, config.VOCAB_SIZE),
                    target_seq.reshape(-1),
                    ignore_index=config.PADDING_IDX
                )
            total_loss += loss.item()
    return total_loss / len(loader)


def train_xe(config):
    os.makedirs(config.CHECKPOINT_PATH, exist_ok=True)

    # 1. Vocabulary
    word2idx, idx2word = build_vocab(
        config.ANNOTATION_PATH, config.VOCAB_PATH,
        karpathy_path=config.KARPATHY_PATH
    )
    config.VOCAB_SIZE = len(word2idx)
    print(f"Vocab size: {config.VOCAB_SIZE:,}")

    # 2. Model
    model = build_ascap_model(
        vocab_size   = config.VOCAB_SIZE,
        bos_idx      = config.BOS_IDX,
        padding_idx  = config.PADDING_IDX,
        d_model      = config.D_MODEL,
        d_k          = config.D_K,
        d_v          = config.D_V,
        h            = config.N_HEADS,
        d_ff         = config.D_FF,
        d_in         = config.D_IN,
        N_enc        = config.N_ENC,
        N_dec        = config.N_DEC,
        max_len      = config.MAX_LEN,
        dropout      = config.DROPOUT
    )
    model = nn.DataParallel(model).cuda()

    # 3. Optimizer + scaler
    optimizer = Adam(model.parameters(),
                     lr=config.LEARNING_RATE_XE,
                     betas=(0.9, 0.98), eps=1e-9)
    scaler    = GradScaler(enabled=config.MIXED_PRECISION)

    # 4. DataLoaders (seq_len detected here)
    train_ds = COCODataset('train', config, word2idx)
    val_ds   = COCODataset('val',   config, word2idx)
    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE_XE,
        shuffle=True,  num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.BATCH_SIZE_XE,
        shuffle=False, num_workers=config.NUM_WORKERS,
        collate_fn=collate_fn, pin_memory=True
    )

    # 5. Scheduler
    total_steps = (config.NUM_EPOCHS_XE
                   * (len(train_loader) // config.GRADIENT_ACCUMULATION))
    scheduler   = get_warmup_cosine_scheduler(
        optimizer, config.WARMUP_STEPS, total_steps
    )

    # 6. Auto-resume
    start_epoch, metrics = load_latest_checkpoint(
        model, optimizer, scaler, scheduler, stage='xe', config=config
    )
    best_val_loss = (metrics.get('val_loss', float('inf'))
                     if metrics else float('inf'))

    # 7. Training loop
    for epoch in range(start_epoch, config.NUM_EPOCHS_XE):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, scheduler, config, epoch
        )
        val_loss   = validate_one_epoch(model, val_loader, config, epoch)

        print(f"\nEpoch {epoch+1}/{config.NUM_EPOCHS_XE} | "
              f"Train: {train_loss:.4f} | Val: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                model.state_dict(),
                os.path.join(config.CHECKPOINT_PATH, 'xe_best_model.pt')
            )
            print(f"⭐ New best val loss: {best_val_loss:.4f}")

        save_checkpoint(epoch, 'xe', model, optimizer, scaler,
                        scheduler,
                        {'train_loss': train_loss, 'val_loss': val_loss},
                        config)
        clean_old_checkpoints('xe', config, keep=3)

    print("\n✅ Prompt 4 PASSED — XE training complete")
    print(f"Best val loss: {best_val_loss:.4f}")


if __name__ == '__main__':
    config = ASCAPConfig()
    train_xe(config)
