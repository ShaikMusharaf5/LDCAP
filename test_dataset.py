"""
test_dataset.py — Validate ASCAP dataset pipeline (Prompt 3)
Runs 12 checks and prints clear PASS/FAIL for each.
"""

import sys
import os
from torch.utils.data import DataLoader

from config import ASCAPConfig
from dataset import build_vocab, COCODataset, collate_fn


def main():
    config = ASCAPConfig()
    all_passed = True

    # ── 1. build_vocab() runs and vocab.json is created ──
    try:
        word2idx, idx2word = build_vocab(
            config.ANNOTATION_PATH,
            config.VOCAB_PATH,
            karpathy_path=config.KARPATHY_PATH,
        )
        assert os.path.exists(config.VOCAB_PATH), "vocab.json not found on disk"
        print("✅ Test 1 PASSED — build_vocab() succeeded, vocab.json created")
    except Exception as e:
        print(f"❌ Test 1 FAILED — build_vocab() error: {e}")
        all_passed = False
        sys.exit(1)

    # ── 2. config.VOCAB_SIZE is set ──
    try:
        config.VOCAB_SIZE = len(word2idx)
        assert config.VOCAB_SIZE is not None and config.VOCAB_SIZE > 0
        print(f"✅ Test 2 PASSED — VOCAB_SIZE = {config.VOCAB_SIZE}")
    except Exception as e:
        print(f"❌ Test 2 FAILED — {e}")
        all_passed = False

    # ── 3. COCODataset('train') loads without error ──
    try:
        train_dataset = COCODataset('train', config, word2idx)
        print("✅ Test 3 PASSED — train dataset loaded")
    except Exception as e:
        print(f"❌ Test 3 FAILED — {e}")
        all_passed = False
        sys.exit(1)

    # ── 4. config.SEQ_LEN is set and printed ──
    try:
        assert config.SEQ_LEN is not None and config.SEQ_LEN > 0
        print(f"✅ Test 4 PASSED — SEQ_LEN = {config.SEQ_LEN}")
    except Exception as e:
        print(f"❌ Test 4 FAILED — {e}")
        all_passed = False

    # ── 5. Train dataset size in expected range ──
    try:
        n_train = len(train_dataset)
        if 110_000 <= n_train <= 600_000:
            print(f"✅ Test 5 PASSED — train pairs = {n_train:,}")
        else:
            print(f"⚠️  Test 5 WARNING — train pairs = {n_train:,} "
                  f"(expected 110,000–600,000)")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 5 FAILED — {e}")
        all_passed = False

    # ── 6. COCODataset('val') loads ──
    try:
        val_dataset = COCODataset('val', config, word2idx)
        print("✅ Test 6 PASSED — val dataset loaded")
    except Exception as e:
        print(f"❌ Test 6 FAILED — {e}")
        all_passed = False
        sys.exit(1)

    # ── 7. Val dataset size in expected range ──
    try:
        n_val = len(val_dataset)
        if 4_900 <= n_val <= 5_100:
            print(f"✅ Test 7 PASSED — val images = {n_val:,}")
        else:
            print(f"⚠️  Test 7 WARNING — val images = {n_val:,} "
                  f"(expected 4,900–5,100)")
            all_passed = False
    except Exception as e:
        print(f"❌ Test 7 FAILED — {e}")
        all_passed = False

    # ── 8. DataLoader with batch_size=4 and collate_fn ──
    try:
        loader = DataLoader(
            val_dataset, batch_size=4, shuffle=False, collate_fn=collate_fn
        )
        print("✅ Test 8 PASSED — DataLoader created")
    except Exception as e:
        print(f"❌ Test 8 FAILED — {e}")
        all_passed = False
        sys.exit(1)

    # ── 9. Fetch one batch and assert shapes ──
    try:
        features, captions, image_ids = next(iter(loader))
        assert features.shape == (4, config.SEQ_LEN, 2048), \
            f"features shape {features.shape} != (4, {config.SEQ_LEN}, 2048)"
        assert captions.shape[0] == 4, \
            f"captions batch dim {captions.shape[0]} != 4"
        assert captions.shape[1] <= 54, \
            f"captions seq dim {captions.shape[1]} > 54"
        print(f"✅ Test 9 PASSED — features={tuple(features.shape)}, "
              f"captions={tuple(captions.shape)}")
    except Exception as e:
        print(f"❌ Test 9 FAILED — {e}")
        all_passed = False

    # ── 10. Decode one caption back to words ──
    try:
        cap_ids = captions[0].tolist()
        words = [idx2word.get(i, '<?>') for i in cap_ids if i != 0]
        decoded = ' '.join(words)
        print(f"✅ Test 10 PASSED — decoded caption: \"{decoded}\"")
    except Exception as e:
        print(f"❌ Test 10 FAILED — {e}")
        all_passed = False

    # ── 11. Print final summary ──
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Vocab size:    {config.VOCAB_SIZE}")
    print(f"SEQ_LEN:       {config.SEQ_LEN}")
    print(f"Train pairs:   {len(train_dataset):,}")
    print(f"Val images:    {len(val_dataset):,}")
    if 'features' in dir():
        print(f"Batch shapes:  features={tuple(features.shape)} "
              f"captions={tuple(captions.shape)}")
    print("=" * 50)

    # ── 12. Final verdict ──
    if all_passed:
        print("\n✅ Prompt 3 PASSED")
    else:
        print("\n❌ Prompt 3 FAILED — see warnings/errors above")

    # Cleanup
    train_dataset.close()
    val_dataset.close()


if __name__ == '__main__':
    main()
