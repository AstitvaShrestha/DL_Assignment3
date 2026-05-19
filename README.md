# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

**Github Link**:- https://github.com/AstitvaShrestha/DL_Assignment3/  
**WandB Link**:- https://api.wandb.ai/links/da25s013-iitm/uyfz3rnj  


---

# Features

- Scaled Dot-Product Attention
- Multi-Head Attention
- Sinusoidal Positional Encoding
- Encoder-Decoder Transformer Architecture
- Padding + Causal Masking
- Label Smoothing
- Noam Learning Rate Scheduler
- Greedy Autoregressive Decoding
- W&B Experiment Tracking
- BLEU Score Evaluation

---

# Dataset

Dataset used:

- Multi30k Dataset  
  https://huggingface.co/datasets/bentrevett/multi30k

Dataset statistics:
- ~29,000 training sentence pairs
- 1,014 validation sentence pairs
- 1,000 test sentence pairs

Language pair:
- German → English

---

# Project Structure

```text
.
├── dataset.py
├── model.py
├── train.py
├── lr_scheduler.py
├── best_model.pth
├── src_vocab.pkl
├── tgt_vocab.pkl
├── requirements.txt
└── README.md
```

---

# Installation

Create a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Required Libraries

Main libraries used:
- torch
- datasets
- spacy
- numpy
- tqdm
- wandb
- nltk
- matplotlib
- gdown

---

# Training

Run training:

```bash
python train.py
```

Example configuration:

```bash
python train.py \
    --d_model 512 \
    --N 6 \
    --num_heads 8 \
    --d_ff 2048 \
    --dropout 0.2 \
    --batch_size 64 \
    --num_epochs 25
```

---

# Training Configuration

Best performing configuration:

| Hyperparameter | Value |
|---|---|
| d_model | 512 |
| Encoder/Decoder Layers (N) | 6 |
| Attention Heads | 8 |
| FFN Dimension | 2048 |
| Dropout | 0.2 |
| Warmup Steps | 4000 |
| Batch Size | 64 |
| Epochs | 25 |

---

# BLEU Score

Final performance:

| Metric | Score |
|---|---|
| Validation BLEU (Peak) | 32.83 |
| Test BLEU | 31.9 |

Evaluation performed using corpus-level BLEU score.

---

# Positional Encoding

Implemented sinusoidal positional encoding exactly as described in the original Transformer paper.

---

# Noam Learning Rate Scheduler

Implemented from scratch using PyTorch scheduler APIs.

---

# Inference

The model performs greedy autoregressive decoding token-by-token.

Example:

```python
from model import Transformer

model = Transformer()

translation = model.infer(
    "ein mann spielt gitarre"
)

print(translation)
```

---

# Tokenization

Tokenization implemented using:

```python
spacy.blank("de")
spacy.blank("en")
```

This avoids external spaCy model downloads and ensures portability in Gradescope/autograder environments.

---

# Implementation Notes

- Entire implementation written using basic PyTorch building blocks.
- `torch.nn.MultiheadAttention` was NOT used.
- Attention masking implemented manually.
- Layer normalization implemented using `nn.LayerNorm`.
- Greedy decoding implemented manually.

---