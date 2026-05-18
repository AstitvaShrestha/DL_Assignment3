# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

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
    --d_model 256 \
    --N 4 \
    --num_heads 4 \
    --d_ff 1024 \
    --dropout 0.1 \
    --batch_size 64 \
    --num_epochs 20
```

---

# Training Configuration

Best performing configuration:

| Hyperparameter | Value |
|---|---|
| d_model | 256 |
| Encoder/Decoder Layers (N) | 4 |
| Attention Heads | 4 |
| FFN Dimension | 1024 |
| Dropout | 0.1 |
| Warmup Steps | 4000 |
| Batch Size | 64 |
| Epochs | 20 |

---

# BLEU Score

Final performance:

| Metric | Score |
|---|---|
| Validation BLEU | ~32.83 |
| Test BLEU | ~30.5 |

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

# Weights & Biases

Project link:

https://wandb.ai/da25s013-iitm/da6401-assignment3

Tracked metrics:
- Training Loss
- Validation Loss
- Validation BLEU
- Learning Rate
- Test BLEU

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

# References

- Attention Is All You Need  
  https://arxiv.org/abs/1706.03762

- Multi30k Dataset  
  https://huggingface.co/datasets/bentrevett/multi30k

- Assignment Skeleton  
  https://github.com/MiRL-IITM/da6401_assignment_3

---
