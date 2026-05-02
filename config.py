class ASCAPConfig:
    # Model Architecture
    D_MODEL     = 512
    D_K         = 64
    D_V         = 64
    N_HEADS     = 8
    D_FF        = 2048
    D_IN        = 2048
    N_ENC       = 2
    N_DEC       = 3
    MAX_LEN     = 54
    DROPOUT     = 0.1
    BOS_IDX     = 1
    EOS_IDX     = 2
    PADDING_IDX = 0
    VOCAB_SIZE  = None
    SEQ_LEN     = None
    # Stage 1 XE
    BATCH_SIZE_XE         = 50
    LEARNING_RATE_XE      = 1e-4
    NUM_EPOCHS_XE         = 3
    WARMUP_STEPS          = 1000
    GRADIENT_ACCUMULATION = 2
    # Stage 2 SCST
    BATCH_SIZE_SCST    = 25
    LEARNING_RATE_SCST = 5e-6
    NUM_EPOCHS_SCST    = 2
    # Kaggle
    MIXED_PRECISION = True
    NUM_WORKERS     = 4
    # Paths
    FEATURES_PATH   = "/kaggle/input/datasets/mariofrcrce/coco-bottom-up-features-adaptive-k/bottom-up-attention-features"
    ANNOTATION_PATH = "/kaggle/input/datasets/nadaibrahim/coco2014/captions/annotations"
    KARPATHY_PATH   = "/kaggle/input/datasets/musharaf5/coco-karpathy-split/dataset_coco.json"
    CHECKPOINT_PATH = "/kaggle/working/checkpoints"
    VOCAB_PATH      = "/kaggle/working/vocab.json"

# ============================================================
# SCST CONFIGURATION (append to existing config)
# ============================================================
# If these are not already in your ASCAPConfig class, add them:
#
# BATCH_SIZE_SCST = 25        # Half of XE (SCST uses more memory)
# LEARNING_RATE_SCST = 1e-5   # Lower LR for fine-tuning
# EPOCHS_SCST = 5
# MAX_GEN_LEN = 20            # Max generation length
# GRAD_CLIP_SCST = 0.1        # Tighter gradient clipping
