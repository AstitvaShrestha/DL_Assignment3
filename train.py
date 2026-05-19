"""
train.py — Training Pipeline, Inference & Evaluation
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────────┐
  │  greedy_decode(model, src, src_mask, max_len, start_symbol)         │
  │      → torch.Tensor  shape [1, out_len]  (token indices)            │
  │                                                                     │
  │  evaluate_bleu(model, test_dataloader, tgt_vocab, device)           │
  │      → float  (corpus-level BLEU score, 0–100)                      │
  │                                                                     │
  │  save_checkpoint(model, optimizer, scheduler, epoch, path) → None   │
  │  load_checkpoint(path, model, optimizer, scheduler)        → int    │
  └─────────────────────────────────────────────────────────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional

from model import Transformer, make_src_mask, make_tgt_mask
from nltk.translate.bleu_score import corpus_bleu

import wandb

from dataset import Multi30kDataset, collate_fn

from lr_scheduler import NoamScheduler
import argparse
import random
import numpy as np

# ══════════════════════════════════════════════════════════════════════
#  LABEL SMOOTHING LOSS  
# ══════════════════════════════════════════════════════════════════════

class LabelSmoothingLoss(nn.Module):
    """
    Label smoothing as in "Attention Is All You Need"

    Smoothed target distribution:
        y_smooth = (1 - eps) * one_hot(y) + eps / (vocab_size - 1)

    Args:
        vocab_size (int)  : Number of output classes.
        pad_idx    (int)  : Index of <pad> token — receives 0 probability.
        smoothing  (float): Smoothing factor ε (default 0.1).
    """

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()

        self.vocab_size = vocab_size

        self.pad_idx = pad_idx

        self.smoothing = smoothing

        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits : shape [batch * tgt_len, vocab_size]  (raw model output)
            target : shape [batch * tgt_len]              (gold token indices)

        Returns:
            Scalar loss value.
        """
        # TODO: Task 3.1

        # Log probabilities
        log_probs = torch.log_softmax(
            logits,
            dim=-1
        )


        # Create smoothed target distribution
        with torch.no_grad():

            true_dist = torch.zeros_like(log_probs)

            true_dist.fill_(
                self.smoothing / (self.vocab_size - 2)
            )


            # Assign confidence to correct class
            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                self.confidence
            )


            # Zero out PAD token probability
            true_dist[:, self.pad_idx] = 0


            # Ignore PAD positions entirely
            pad_mask = (target == self.pad_idx)

            true_dist[pad_mask] = 0


        # KL-divergence style loss
        loss = torch.sum(
            -true_dist * log_probs,
            dim=1
        )


        # Ignore PAD tokens in averaging
        non_pad_mask = (target != self.pad_idx)

        loss = loss.masked_select(non_pad_mask)

        return loss.mean()


# ══════════════════════════════════════════════════════════════════════
#   TRAINING LOOP  
# ══════════════════════════════════════════════════════════════════════

def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
    global_step=0,         
    log_grad_steps=1000,    # <-- log only first 1000 steps
) -> float:
    """
    Run one epoch of training or evaluation.

    Args:
        data_iter  : DataLoader yielding (src, tgt) batches of token indices.
        model      : Transformer instance.
        loss_fn    : LabelSmoothingLoss (or any nn.Module loss).
        optimizer  : Optimizer (None during eval).
        scheduler  : NoamScheduler instance (None during eval).
        epoch_num  : Current epoch index (for logging).
        is_train   : If True, perform backward pass and scheduler step.
        device     : 'cpu' or 'cuda'.

    Returns:
        avg_loss : Average loss over the epoch (float).

    """

    # ------------------------------------------------
    # Set train/eval mode
    # ------------------------------------------------
    if is_train:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    # ------------------------------------------------
    # Iterate over batches
    # ------------------------------------------------
    for batch_idx, (src, tgt) in enumerate(data_iter):

        # ------------------------------------------------
        # Move tensors to device
        # ------------------------------------------------
        src = src.to(device)

        tgt = tgt.to(device)

        # ------------------------------------------------
        # Teacher forcing shift
        # ------------------------------------------------
        tgt_input = tgt[:, :-1]

        tgt_output = tgt[:, 1:]

        # ------------------------------------------------
        # Create masks
        # ------------------------------------------------
        src_mask = make_src_mask(
            src
        ).to(device)

        tgt_mask = make_tgt_mask(
            tgt_input
        ).to(device)

        # ------------------------------------------------
        # Forward pass
        # ------------------------------------------------
        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask
        )

        # ------------------------------------------------
        # Reshape for loss
        #
        # logits:
        # [batch, tgt_len, vocab]
        #
        # target:
        # [batch, tgt_len]
        # ------------------------------------------------
        logits = logits.reshape(
            -1,
            logits.shape[-1]
        )

        tgt_output = tgt_output.reshape(-1)

        
        # ------------------------------------------------
        # Compute loss
        # ------------------------------------------------
        loss = loss_fn(
            logits,
            tgt_output
        )

        # ------------------------------------------------
        # Backpropagation
        # ------------------------------------------------
        if is_train:

            optimizer.zero_grad()

            loss.backward()

            # ------------------------------------------------
            # Gradient clipping
            # ------------------------------------------------
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            # # ── Gradient norm logging ──────────────────────
            # if global_step < log_grad_steps:
            #     q_norms, k_norms = [], []

            #     for name, param in model.named_parameters():
            #         if param.grad is None:
            #             continue
            #         if "W_q" in name and "weight" in name:
            #             q_norms.append(param.grad.norm().item())
            #         if "W_k" in name and "weight" in name:
            #             k_norms.append(param.grad.norm().item())

            #     if q_norms and k_norms:
            #         wandb.log({
            #             "grad_norm/Q_mean": sum(q_norms) / len(q_norms),
            #             "grad_norm/K_mean": sum(k_norms) / len(k_norms),
            #             "grad_norm/Q_max":  max(q_norms),
            #             "grad_norm/K_max":  max(k_norms),
            #             "step": global_step,
            #         })
            # # ───────────────────────────────────────────────

            optimizer.step()

            # ------------------------------------------------
            # Noam scheduler step
            # ------------------------------------------------
            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()
        global_step += 1   

    # ------------------------------------------------
    # Average epoch loss
    # ------------------------------------------------
    avg_loss = total_loss / len(data_iter)

    return avg_loss, global_step


# ══════════════════════════════════════════════════════════════════════
#   GREEDY DECODING  
# ══════════════════════════════════════════════════════════════════════

def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Generate a translation token-by-token using greedy decoding.

    Args:
        model        : Trained Transformer.
        src          : Source token indices, shape [1, src_len].
        src_mask     : shape [1, 1, 1, src_len].
        max_len      : Maximum number of tokens to generate.
        start_symbol : Vocabulary index of <sos>.
        end_symbol   : Vocabulary index of <eos>.
        device       : 'cpu' or 'cuda'.

    Returns:
        ys : Generated token indices, shape [1, out_len].
             Includes start_symbol; stops at (and includes) end_symbol
             or when max_len is reached.

    """
    # TODO: Task 3.3 — implement token-by-token greedy decoding
    """
    Greedy autoregressive decoding.
    """

    model.eval()

    src = src.to(device)

    src_mask = src_mask.to(device)

    # ------------------------------------------------
    # Encode source sentence
    # ------------------------------------------------
    with torch.no_grad():

        memory = model.encode(
            src,
            src_mask
        )

    # ------------------------------------------------
    # Initialize decoder input with <sos>
    # ------------------------------------------------
    ys = torch.ones(
        1,
        1,
        dtype=torch.long
    ).fill_(start_symbol).to(device)

    # ------------------------------------------------
    # Autoregressive decoding loop
    # ------------------------------------------------
    for _ in range(max_len - 1):

        # ------------------------------------------------
        # Target mask
        # ------------------------------------------------
        tgt_mask = make_tgt_mask(
            ys
        ).to(device)

        with torch.no_grad():

            out = model.decode(
                memory,
                src_mask,
                ys,
                tgt_mask
            )

        # ------------------------------------------------
        # Last token logits
        # ------------------------------------------------
        prob = out[:, -1, :]

        # ------------------------------------------------
        # Greedy token selection
        # ------------------------------------------------
        next_word = torch.argmax(
            prob,
            dim=-1
        ).item()

        # ------------------------------------------------
        # Append generated token
        # ------------------------------------------------
        next_word_tensor = torch.ones(
            1,
            1,
            dtype=torch.long
        ).fill_(next_word).to(device)

        ys = torch.cat(
            [ys, next_word_tensor],
            dim=1
        )

        # ------------------------------------------------
        # Stop at EOS
        # ------------------------------------------------
        if next_word == end_symbol:
            break

    return ys


# ══════════════════════════════════════════════════════════════════════
#   BLEU EVALUATION  
# ══════════════════════════════════════════════════════════════════════

def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:
    """
    Evaluate translation quality with corpus-level BLEU score.

    Args:
        model           : Trained Transformer (in eval mode).
        test_dataloader : DataLoader over the test split.
                          Each batch yields (src, tgt) token-index tensors.
        tgt_vocab       : Vocabulary object with idx_to_token mapping.
                          Must support  tgt_vocab.itos[idx]  or
                          tgt_vocab.lookup_token(idx).
        device          : 'cpu' or 'cuda'.
        max_len         : Max decode length per sentence.

    Returns:
        bleu_score : Corpus-level BLEU (float, range 0–100).

    """
    # TODO: Task 3 — loop test set, decode, compute and return BLEU
    
    model.eval()

    references = []

    hypotheses = []

    sos_idx = model.tgt_vocab["<sos>"]

    eos_idx = model.tgt_vocab["<eos>"]

    pad_idx = model.pad_idx

    with torch.no_grad():

        for src_batch, tgt_batch in test_dataloader:

            src_batch = src_batch.to(device)

            tgt_batch = tgt_batch.to(device)

            batch_size = src_batch.size(0)

            # ------------------------------------------------
            # Decode each sentence individually
            # ------------------------------------------------
            for i in range(batch_size):

                src = src_batch[i].unsqueeze(0)

                tgt = tgt_batch[i]

                # ------------------------------------------------
                # Source mask
                # ------------------------------------------------
                src_mask = make_src_mask(
                    src,
                    pad_idx
                ).to(device)

                # ------------------------------------------------
                # Greedy decode
                # ------------------------------------------------
                pred_tokens = greedy_decode(
                    model,
                    src,
                    src_mask,
                    max_len,
                    sos_idx,
                    eos_idx,
                    device
                )

                pred_tokens = pred_tokens.squeeze(0).tolist()

                tgt_tokens = tgt.tolist()

                # ------------------------------------------------
                # Convert prediction indices → tokens
                # ------------------------------------------------
                pred_sentence = []

                for idx in pred_tokens:

                    token = model.tgt_itos[idx]

                    if token in ["<sos>", "<eos>", "<pad>"]:
                        continue

                    pred_sentence.append(token)

                # ------------------------------------------------
                # Convert target indices → tokens
                # ------------------------------------------------
                target_sentence = []

                for idx in tgt_tokens:

                    token = model.tgt_itos[idx]

                    if token in ["<sos>", "<eos>", "<pad>"]:
                        continue

                    target_sentence.append(token)

                # ------------------------------------------------
                # BLEU formatting
                # ------------------------------------------------
                hypotheses.append(pred_sentence)

                references.append([target_sentence])

    # ------------------------------------------------
    # Compute corpus BLEU
    # ------------------------------------------------
    bleu = corpus_bleu(
        references,
        hypotheses
    )

    # ------------------------------------------------
    # Convert to percentage
    # ------------------------------------------------
    return bleu * 100


# ══════════════════════════════════════════════════════════════════════
# ❺  CHECKPOINT UTILITIES  (autograder loads your model from disk)
# ══════════════════════════════════════════════════════════════════════

def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:
    """
    Save model + optimiser + scheduler state to disk.

    The autograder will call load_checkpoint to restore your model.
    Do NOT change the keys in the saved dict.

    Args:
        model     : Transformer instance.
        optimizer : Optimizer instance.
        scheduler : NoamScheduler instance.
        epoch     : Current epoch number.
        path      : File path to save to (default 'checkpoint.pt').

    Saves a dict with keys:
        'epoch', 'model_state_dict', 'optimizer_state_dict',
        'scheduler_state_dict', 'model_config'

    model_config must contain all kwargs needed to reconstruct
    Transformer(**model_config), e.g.:
        {'src_vocab_size': ..., 'tgt_vocab_size': ...,
         'd_model': ..., 'N': ..., 'num_heads': ...,
         'd_ff': ..., 'dropout': ...}
    """
    # TODO: implement using torch.save({...}, path)
    
    checkpoint = {

        # ------------------------------------------------
        # Current epoch
        # ------------------------------------------------
        "epoch": epoch,

        # ------------------------------------------------
        # Model weights
        # ------------------------------------------------
        "model_state_dict":
            model.state_dict(),

        # ------------------------------------------------
        # Optimizer state
        # ------------------------------------------------
        "optimizer_state_dict":
            optimizer.state_dict(),

        # ------------------------------------------------
        # Scheduler state
        # ------------------------------------------------
        "scheduler_state_dict":
            scheduler.state_dict()
            if scheduler is not None
            else None,

        # ------------------------------------------------
        # Model reconstruction config
        # ------------------------------------------------
        "model_config": {

            "src_vocab_size":
                len(model.src_vocab),

            "tgt_vocab_size":
                len(model.tgt_vocab),

            "d_model":
                model.d_model,

            "N":
                len(model.encoder.layers),

            "num_heads":
                model.encoder.layers[0]
                .self_attn.num_heads,

            "d_ff":
                model.encoder.layers[0]
                .ffn.linear1.out_features,

            "dropout":
                model.encoder.layers[0]
                .dropout.p,
        }
    }

    torch.save(checkpoint, path)


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:
    """
    Restore model (and optionally optimizer/scheduler) state from disk.

    Args:
        path      : Path to checkpoint file saved by save_checkpoint.
        model     : Uninitialised Transformer with matching architecture.
        optimizer : Optimizer to restore (pass None to skip).
        scheduler : Scheduler to restore (pass None to skip).

    Returns:
        epoch : The epoch at which the checkpoint was saved (int).

    """
    # TODO: implement restore logic

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    # ------------------------------------------------
    # Restore model weights
    # ------------------------------------------------
    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    # ------------------------------------------------
    # Restore optimizer
    # ------------------------------------------------
    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

    # ------------------------------------------------
    # Restore scheduler
    # ------------------------------------------------
    if (
        scheduler is not None and
        checkpoint["scheduler_state_dict"] is not None
    ):

        scheduler.load_state_dict(
            checkpoint["scheduler_state_dict"]
        )

    # ------------------------------------------------
    # Return saved epoch
    # ------------------------------------------------
    return checkpoint["epoch"]

def compute_confidence(model, data_iter, device):
    """
    Compute average softmax probability assigned to correct token
    across the validation set.
    """
    model.eval()
    total_confidence = 0.0
    total_tokens = 0

    with torch.no_grad():
        for src, tgt in data_iter:
            src = src.to(device)
            tgt = tgt.to(device)

            tgt_input  = tgt[:, :-1]
            tgt_output = tgt[:, 1:]

            src_mask = make_src_mask(src).to(device)
            tgt_mask = make_tgt_mask(tgt_input).to(device)

            logits = model(src, tgt_input, src_mask, tgt_mask)
            logits = logits.reshape(-1, logits.shape[-1])
            tgt_output = tgt_output.reshape(-1)

            probs = torch.softmax(logits, dim=-1)

            correct_probs = probs.gather(
                1, tgt_output.unsqueeze(1)
            ).squeeze(1)

            non_pad_mask = (tgt_output != 1)
            correct_probs = correct_probs[non_pad_mask]

            total_confidence += correct_probs.sum().item()
            total_tokens += non_pad_mask.sum().item()

    return total_confidence / total_tokens if total_tokens > 0 else 0.0

# ══════════════════════════════════════════════════════════════════════
#   EXPERIMENT ENTRY POINT
# ══════════════════════════════════════════════════════════════════════

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--N", type=int, default=6)
    parser.add_argument("--num_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--learning_rate", type=float, default=1.0)
    parser.add_argument("--warmup_steps", type=int, default=4000)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--no_noam", dest="use_noam", action="store_false")
    parser.set_defaults(use_noam=True)
    
    parser.add_argument("--fixed_lr", type=float, default=1e-4, help="Fixed learning rate when not using Noam scheduler.")
    
    parser.add_argument("--no_scale", action="store_true", help="Disable attention scaling")

    parser.add_argument("--learned_pe", action="store_true", help="Use learned positional embeddings instead of sinusoidal")
    
    parser.add_argument("--smoothing", type=float, default=0.1, help="Label smoothing factor (0.0 = standard cross entropy)")
    
    parser.add_argument("--run_name", type=str, default=None, help="W&B run name")
    
    return parser.parse_args()

def run_training_experiment() -> None:
    """
    Set up and run the full training experiment.

    Steps:
        1. Init W&B:   wandb.init(project="da6401-a3", config={...})
        2. Build dataset / vocabs from dataset.py
        3. Create DataLoaders for train / val splits
        4. Instantiate Transformer with hyperparameters from config
        5. Instantiate Adam optimizer (β1=0.9, β2=0.98, ε=1e-9)
        6. Instantiate NoamScheduler(optimizer, d_model, warmup_steps=4000)
        7. Instantiate LabelSmoothingLoss(vocab_size, pad_idx, smoothing=0.1)
        8. Training loop:
               for epoch in range(num_epochs):
                   run_epoch(train_loader, model, loss_fn,
                             optimizer, scheduler, epoch, is_train=True)
                   run_epoch(val_loader, model, loss_fn,
                             None, None, epoch, is_train=False)
                   save_checkpoint(model, optimizer, scheduler, epoch)
        9. Final BLEU on test set:
               bleu = evaluate_bleu(model, test_loader, tgt_vocab)
               wandb.log({'test_bleu': bleu})
    """
    # TODO: implement full experiment

    # Parse command-line arguments
    args = parse_args()

    # Reproducibility
    seed = 42

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Initialize W&B
    wandb.init(
        project="da6401-assignment3",
        entity="da25s013-iitm",
        name=args.run_name, 
        config=vars(args)
    )

    device = args.device

    print(f"Using device: {device}")

    # Datasets
    train_dataset = Multi30kDataset(
        split="train"
    )

    val_dataset = Multi30kDataset(
        split="validation"
    )

    test_dataset = Multi30kDataset(
        split="test"
    )

    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    use_scale = not args.no_scale

    # Model
    model = Transformer(
        src_vocab_size=len(train_dataset.src_vocab),
        tgt_vocab_size=len(train_dataset.tgt_vocab),
        d_model=args.d_model,
        N=args.N,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_scale=use_scale,
        learned_pe=args.learned_pe
    ).to(device)

    if args.use_noam:
        # Optimizer
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.98),
            eps=1e-9
        )


        # Noam Scheduler
        scheduler = NoamScheduler(
            optimizer,
            d_model=args.d_model,
            warmup_steps=args.warmup_steps
        )

    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.fixed_lr,   # e.g. 1e-4
            betas=(0.9, 0.98),
            eps=1e-9
        )

        scheduler = None  # run_epoch already handles scheduler=None gracefully

    # Loss function
    loss_fn = LabelSmoothingLoss(
        vocab_size=len(train_dataset.tgt_vocab),
        pad_idx=1,
        smoothing=args.smoothing
    )

    # best_val_loss = float("inf")
    best_val_bleu = 0.0

    global_step = 0
    # Training loop
    for epoch in range(args.num_epochs):


        # Train epoch
        train_loss, global_step = run_epoch(
            data_iter=train_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch_num=epoch,
            is_train=True,
            device=device,
            global_step=global_step,
            log_grad_steps=1000,
        )

        # Validation epoch
        val_loss, _ = run_epoch(
            data_iter=val_loader,
            model=model,
            loss_fn=loss_fn,
            optimizer=None,
            scheduler=None,
            epoch_num=epoch,
            is_train=False,
            device=device,
            global_step=global_step,
            log_grad_steps=1000,
        )

        val_bleu = evaluate_bleu(
            model=model,
            test_dataloader=val_loader,
            tgt_vocab=train_dataset.tgt_vocab,
            device=device,
        )
        
        # Compute confidence on val set
        confidence = compute_confidence(
            model, val_loader, device
        )

        
        print(
            f"Epoch {epoch+1}/{args.num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val BLEU: {val_bleu:.2f}"
        )

        # ------------------------------------------------
        # Log metrics to W&B
        # ------------------------------------------------
        wandb.log({
            "epoch": epoch+1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_bleu": val_bleu,
            "pred_confidence": confidence,
            "learning_rate":
                optimizer.param_groups[0]["lr"],
        })

        # Save best checkpoint
        if val_bleu > best_val_bleu:

            best_val_bleu = val_bleu
        # if val_loss < best_val_loss:

        #     best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                path="best_model.pth",
            )

            print("Saved best checkpoint.")


    # Load best model before BLEU evaluation
    load_checkpoint(
        path="best_model.pth",
        model=model
    )

    # ------------------------------------------------
    # Final BLEU evaluation
    # ------------------------------------------------
    bleu = evaluate_bleu(
        model=model,
        test_dataloader=test_loader,
        tgt_vocab=train_dataset.tgt_vocab,
        device=device,
    )

    print(f"\nTest BLEU: {bleu:.2f}")

    wandb.log({
        "test_bleu": bleu
    })

    wandb.finish()


if __name__ == "__main__":
    run_training_experiment()
