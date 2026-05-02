# Zenodo Release Description: LDCAP / ASCAP Image Captioning

## Description

This release contains the code and supporting files for an image captioning project based on transformer decoding, bottom-up visual region features, and ASCAP/SCST-style training. The package is intended to support reproducibility, software reuse, and later extension by other researchers.

This repository currently provides:

- the Streamlit inference application
- the inference pipeline
- the project configuration module
- the FiLM-based sifting attention module
- the transformer model definition used by inference
- the RIN module used by the ASCAP variant
- the XE training entry script
- the SCST training entry script
- the dataset validation script
- the vocabulary file used by the current setup

This release is best understood as a research software package for:

- local inference and demonstration
- documentation of the training and inference workflow
- partial reproduction of the original experimental pipeline

## External Resources Required

The following Kaggle datasets were used in the project and should be cited as external inputs when reproducing the experiments:

1. `/kaggle/input/datasets/mariofrcrce/coco-bottom-up-features-adaptive-k`
   Purpose: pre-extracted COCO adaptive-k bottom-up visual features used during training and validation.
2. `/kaggle/input/datasets/musharaf5/coco-karpathy-split`
   Purpose: Karpathy train/validation/test split and caption reference organization.
3. `/kaggle/input/datasets/nadaibrahim/coco2014`
   Purpose: original COCO 2014 image data and annotations.
4. `/kaggle/input/datasets/musharaf5/scap-source-code`
   Purpose: external SCAP source code used by `ascap_encoder.py` and `ascap_decoder.py`.
5. `/kaggle/input/datasets/soumikrakshit/lol-dataset`
   Purpose: not directly used by the scripts currently included in this repository snapshot. Retain only if reproducing a broader experiment outside the current package.

Kaggle notebook associated with this project:

- `https://www.kaggle.com/code/musharaf5/caption`
  Purpose: companion Kaggle notebook for the training and experimentation workflow.

## Contents Of This Release

Current repository layout:

```text
LDCAP-Caption/
|-- .streamlit/
|   |-- config.toml
|-- checkpoints/
|-- models/
|   |-- __init__.py
|   |-- ldcap_transformer.py
|-- venv/
|-- __pycache__/
|-- .gitignore
|-- app.py
|-- ascap_decoder.py
|-- ascap_encoder.py
|-- config.py
|-- film_sifting.py
|-- inference (1).py
|-- inference.py
|-- README.md
|-- requirements.txt
|-- rin.py
|-- test_dataset.py
|-- test_image.jpg
|-- train_scst.py
|-- train_xe.py
|-- vocab.json
```

File and folder roles:

- `.streamlit/config.toml`: Streamlit theme and server configuration.
- `checkpoints/`: expected location for model checkpoints such as `xe_best_model.pt` and `scst_best_model.pt`.
- `checkpoints/scst_best_model.pt`: currently included trained SCST checkpoint in this release package.
- `models/ldcap_transformer.py`: transformer architecture used by the inference workflow.
- `config.py`: central experiment configuration, model hyperparameters, training settings, and Kaggle dataset/checkpoint paths.
- `film_sifting.py`: FiLM-conditioned visual cross-attention module used by the ASCAP decoder variant.
- `rin.py`: RIN module used by the ASCAP encoder variant.
- `app.py`: Streamlit user interface for image upload and caption generation.
- `inference.py`: primary inference pipeline used for checkpoint loading, vocabulary loading, Faster R-CNN feature extraction, and caption decoding.
- `inference (1).py`: alternate inference file retained in the package as a supplementary local copy.
- `train_xe.py`: stage-1 cross-entropy training entry script.
- `train_scst.py`: stage-2 self-critical sequence training entry script.
- `test_dataset.py`: dataset and vocabulary sanity-check script.
- `ascap_encoder.py` and `ascap_decoder.py`: auxiliary ASCAP components that rely on external SCAP and local custom modules.
- `vocab.json`: vocabulary used by the current project snapshot.
- `test_image.jpg`: sample image for a quick inference check.

## Software Requirements

Recommended environment:

- Python `3.10`
- `pip` `24.x`
- GPU recommended for training
- CPU acceptable for inference, with slower runtime

Pinned package versions currently recorded in `requirements.txt`:

```text
streamlit==1.35.0
torch==2.6.0
torchvision==0.21.0
Pillow==10.4.0
numpy==1.26.4
tqdm==4.67.1
```

Additional training-only package:

```text
pycocoevalcap
```

## Installation

### Local installation

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install pycocoevalcap
```

Linux or macOS:

```bash
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install pycocoevalcap
```

If a CUDA-enabled build of PyTorch is required, install the matching `torch` and `torchvision` build first, then install the remaining dependencies.

### Kaggle environment

The training scripts are written around Kaggle-style paths such as `/kaggle/input/...` and `/kaggle/working/...`. A typical Kaggle reproduction flow is:

1. Attach the five Kaggle datasets listed above.
2. Copy or clone this repository into `/kaggle/working/`.
3. Confirm that all training-side file paths point to the correct Kaggle inputs in your execution environment.
4. Run validation, XE training, and SCST training in sequence.

## Reproducibility Package Scope

This release is organized as the complete software package for reproducing the workflow documented in this repository snapshot, including:

- inference and Streamlit deployment
- configuration for the documented Kaggle training setup
- reference Kaggle notebook workflow
- dataset validation
- XE training entry
- SCST training entry
- ASCAP-related helper modules included in this package
- trained SCST checkpoint and vocabulary used by the current setup

The software assumes the dataset and runtime paths used by the training scripts and Kaggle inputs listed above.

## Expected Data And Path Configuration

The training scripts use the following logical resources:

- `config.ANNOTATION_PATH`
- `config.VOCAB_PATH`
- `config.KARPATHY_PATH`
- `config.FEATURES_PATH`
- `config.CHECKPOINT_PATH`

These should be configured to match the dataset locations in the target runtime environment.

The bundled `config.py` currently points to the Kaggle dataset paths listed in this README and uses `/kaggle/working/checkpoints` and `/kaggle/working/vocab.json` for training outputs.

## Reproduction Workflow

### 1. Prepare the runtime layout

Place the repository in the working layout expected by the scripts and make sure all configured paths resolve correctly in the target environment.

### 2. Validate dataset loading

Run:

```bash
python test_dataset.py
```

This script is intended to verify:

- vocabulary generation
- dataset loading
- sequence length setup
- dataloader construction
- one-batch feature and caption shapes

### 3. Run stage-1 XE training

Run:

```bash
python train_xe.py
```

Expected function:

- build vocabulary
- build datasets
- build the ASCAP model
- train with cross-entropy supervision
- validate on the validation split
- save the best XE checkpoint

Expected output:

```text
checkpoints/xe_best_model.pt
```

### 4. Run stage-2 SCST training

Run:

```bash
python train_scst.py
```

Expected function:

- load `xe_best_model.pt`
- generate sampled and greedy captions
- compute CIDEr reward with `pycocoevalcap`
- fine-tune with self-critical sequence training
- save the best SCST checkpoint

Expected outputs:

```text
checkpoints/scst_best_model.pt
checkpoints/scst_checkpoint.pt
```

Important path note:

`train_scst.py` contains hardcoded Kaggle-style imports, including:

- `/kaggle/working/config.py`
- `/kaggle/working/dataset.py`
- `/kaggle/working/models/ascap_transformer.py`

If the code is executed outside Kaggle, those imports must be adapted or the same directory structure must be mirrored locally.

## Inference Workflow

Local inference is possible with the current repository if a compatible checkpoint and matching vocabulary are available.

Recommended placement:

```text
LDCAP-Caption/
|-- checkpoints/
|   |-- scst_best_model.pt
|-- vocab.json
```

Checkpoint currently present in this repository:

```text
checkpoints/scst_best_model.pt
```

The inference app automatically searches for checkpoints in:

- `checkpoints/`
- the repository root
- the parent folder
- the grandparent folder

Preferred checkpoint names:

- `xe_best_model.pt`
- `scst_best_model.pt`
- `rin_ascap_pipeline.pth`

## Running The Streamlit Application

Launch locally with:

```bash
streamlit run app.py
```

Default local address:

```text
http://localhost:8501
```

Basic usage:

1. Start the application.
2. Upload a `jpg`, `jpeg`, `png`, or `webp` image.
3. Select decoding mode.
4. Select maximum token length.
5. Generate a caption.

## Hosting

### Local network hosting

```bash
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```

Other devices on the same network can access:

```text
http://YOUR_LOCAL_IP:8501
```

### Streamlit cloud-style hosting

This is possible, but users should verify:

- final checkpoint size
- memory use during Faster R-CNN feature extraction
- ability to download required torchvision detector weights
- startup time

## Known Limitations Of This Release

- The training workflow is path-sensitive and assumes the documented Kaggle-style runtime layout.
- `pycocoevalcap` is not included in `requirements.txt` and must be installed separately for SCST training.
- Some scripts depend on Kaggle-hardcoded file paths.
- A dedicated evaluation script for final benchmark reporting is not included.
- No `environment.yml` or lock file is included.

## Reuse Note

This release is intended to support reproducibility, reuse, and citation of the software package, trained checkpoint, and documented workflow included in this repository snapshot.
