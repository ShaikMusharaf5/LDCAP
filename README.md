# LDCAP

LDCAP is a local Streamlit app for image caption generation using a transformer-based captioning model with bottom-up visual features extracted from Faster R-CNN.

It provides:
- a polished web interface for image upload and caption generation
- automatic checkpoint and vocabulary discovery
- greedy and beam-search decoding
- caption history with runtime and token breakdown

## Project Overview

This project is organized around two main runtime files:

- `app.py`: Streamlit UI, model discovery, generation controls, and caption history
- `inference.py`: checkpoint loading, vocabulary loading, bottom-up feature extraction, and caption decoding

The current codebase expects a model compatible with the transformer defined in `models/ldcap_transformer.py`.

## Features

- Upload `jpg`, `jpeg`, `png`, and `webp` images
- Generate captions with greedy or beam-search decoding
- Automatically detect vocabulary files
- Automatically detect model checkpoints from common locations
- Display caption history, generation time, and token count
- Cache the loaded model for faster repeated use in Streamlit

## Tech Stack

- Python
- Streamlit
- PyTorch
- Torchvision
- Pillow
- NumPy

## Repository Structure

```text
LDCAP/
|- app.py
|- inference.py
|- requirements.txt
|- README.md
|- vocab.json
|- .streamlit/
|  |- config.toml
|- checkpoints/
|  |- scst_best_model.pt
|- models/
   |- __init__.py
   |- ldcap_transformer.py
```

## Requirements

- Python 3.10 or newer recommended
- `pip`
- Enough disk space for model checkpoints
- Internet access on first run if Torchvision needs to download detector weights

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPOSITORY.git
cd YOUR_REPOSITORY
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv venv
venv\Scripts\activate
```

macOS/Linux:

```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

If you want GPU support, install the correct CUDA-enabled PyTorch build from the official PyTorch site first, then install the remaining dependencies.

## Required Files

The app needs:

- a model checkpoint
- a vocabulary JSON file

### Vocabulary

Supported vocab layouts:

- `{"word2idx": {...}, "idx2word": {...}}`
- a flat `{"token": index}` dictionary

Recommended filename:

```text
vocab.json
```

Recommended locations:

- project root
- `models/`
- `checkpoints/`

### Checkpoint

The app currently prefers these filenames when auto-detecting a model:

- `xe_best_model.pt`
- `scst_best_model.pt`
- `rin_ascap_pipeline.pth`

It also searches for other checkpoint-like files with these extensions:

- `.pth`
- `.pt`
- `.ckpt`
- `.bin`

Search locations:

- `checkpoints/`
- project root
- parent folder of the project
- grandparent folder of the project

This makes it possible to keep a checkpoint either inside the repo or in a nearby folder on your machine.

## How To Run

Start the Streamlit app:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

## How To Use

1. Launch the app with `streamlit run app.py`
2. Upload an image
3. Choose a decode mode: `Greedy`, `Beam Search (x3)`, or `Beam Search (x5)`
4. Set the maximum token length
5. Click `Generate Caption`
6. Review the caption, tokens, and generation history

## Model Loading Behavior

At startup, the app:

1. searches for a checkpoint
2. searches for a vocab file
3. loads `CaptionGenerator` from `inference.py`
4. builds the transformer from `models/ldcap_transformer.py`
5. extracts image region features using Torchvision Faster R-CNN
6. runs decoding to generate the caption

The model is cached using `st.cache_resource`, so repeated runs in the same session do not reload the full model every time.

## Current Inference Pipeline

`inference.py` currently does the following:

- loads checkpoints safely across modern PyTorch versions
- normalizes several common checkpoint layouts
- strips `module.` prefixes from DataParallel checkpoints
- extracts region features from Faster R-CNN detections
- supports greedy decoding and beam search
- applies basic decoding cleanup to reduce repeated or invalid tokens

## Notes About Caption Quality

The UI can load a checkpoint successfully even when caption quality is still poor.

This usually happens when:

- the checkpoint was trained with a different feature extraction pipeline
- the model architecture is only partially compatible
- the vocabulary does not exactly match training
- the checkpoint is an earlier training stage and not the best final model

For example, a checkpoint may load correctly because tensor sizes match, but still produce weak captions if its training-time visual features do not match the inference-time features in this app.

## Example Local Setup

```text
LDCAP/
|- app.py
|- inference.py
|- vocab.json
|- checkpoints/
|  |- scst_best_model.pt
|- models/
   |- ldcap_transformer.py
```

Another valid layout:

```text
Final year/
|- xe_best_model.pt
|- New folder/
   |- LDCAP/
      |- app.py
      |- inference.py
      |- vocab.json
      |- checkpoints/
      |- models/
```

Because the app searches parent folders, it can still find `xe_best_model.pt` in that setup.



## Future Improvements

- manual model selector in the UI
- clearer display of which checkpoint is currently loaded
- better alignment between training-time and inference-time feature extraction
- cloud deployment flow for large checkpoints
- improved caption post-processing



