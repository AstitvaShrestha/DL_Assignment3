"""
model.py — Transformer Architecture Skeleton
DA6401 Assignment 3: "Attention Is All You Need"

AUTOGRADER CONTRACT (DO NOT MODIFY SIGNATURES):
  ┌─────────────────────────────────────────────────────────────────┐
  │  scaled_dot_product_attention(Q, K, V, mask) → (out, weights)  │
  │  MultiHeadAttention.forward(q, k, v, mask)   → Tensor          │
  │  PositionalEncoding.forward(x)               → Tensor          │
  │  make_src_mask(src, pad_idx)                 → BoolTensor      │
  │  make_tgt_mask(tgt, pad_idx)                 → BoolTensor      │
  │  Transformer.encode(src, src_mask)           → Tensor          │
  │  Transformer.decode(memory,src_m,tgt,tgt_m)  → Tensor          │
  └─────────────────────────────────────────────────────────────────┘
"""

import math
import copy
import os
import pickle
import gdown
from typing import Optional, Tuple

import spacy
import torch
import torch.nn as nn
import torch.nn.functional as F


# ══════════════════════════════════════════════════════════════════════
#   STANDALONE ATTENTION FUNCTION  
#    Exposed at module level so the autograder can import and test it
#    independently of MultiHeadAttention.
# ══════════════════════════════════════════════════════════════════════

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute Scaled Dot-Product Attention.

        Attention(Q, K, V) = softmax( Q·Kᵀ / √dₖ ) · V

    Args:
        Q    : Query tensor,  shape (..., seq_q, d_k)
        K    : Key tensor,    shape (..., seq_k, d_k)
        V    : Value tensor,  shape (..., seq_k, d_v)
        mask : Optional Boolean mask, shape broadcastable to
               (..., seq_q, seq_k).
               Positions where mask is True are MASKED OUT
               (set to -inf before softmax).

    Returns:
        output : Attended output,   shape (..., seq_q, d_v)
        attn_w : Attention weights, shape (..., seq_q, seq_k)
    """

    # Dimension of key vectors
    d_k = Q.size(-1)

    # Compute raw attention scores, K.transpose(-2, -1): [batch, heads, d_k, seq_k]
    # scores:[batch, heads, seq_q, seq_k]
    scores = torch.matmul(Q, K.transpose(-2, -1))
    
    scores = scores / math.sqrt(d_k) # Scale scores. Prevents large dot products

    # Apply mask. True values are masked out
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))


    # Softmax over key dimension
    attention_weights = torch.softmax(scores, dim=-1)

    # Weighted sum of values
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


# ══════════════════════════════════════════════════════════════════════
# ❷  MASK HELPERS 
#    Exposed at module level so they can be tested independently and
#    reused inside Transformer.forward.
# ══════════════════════════════════════════════════════════════════════

def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a padding mask for the encoder (source sequence).

    Args:
        src     : Source token-index tensor, shape [batch, src_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, 1, src_len]
        True  → position is a PAD token (will be masked out)
        False → real token
    """
    
    # True where token == <pad>
    mask = (src == pad_idx)

    # Add singleton dimensions for broadcasting
    # [batch, src_len] → [batch, 1, 1, src_len]
    mask = mask.unsqueeze(1).unsqueeze(2)  

    return mask

def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:
    """
    Build a combined padding + causal (look-ahead) mask for the decoder.

    Args:
        tgt     : Target token-index tensor, shape [batch, tgt_len]
        pad_idx : Vocabulary index of the <pad> token (default 1)

    Returns:
        Boolean mask, shape [batch, 1, tgt_len, tgt_len]
        True → position is masked out (PAD or future token)
    """

    batch_size, tgt_len = tgt.shape

    # Padding mask [batch, 1, 1, tgt_len]
    pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    # Causal mask
    # Upper triangular part becomes True
    # Shape: [tgt_len, tgt_len]
    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), dtype=torch.bool),
        diagonal=1
    )

    # Add batch/head dimensions: [1, 1, tgt_len, tgt_len]
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

    # Combine masks, True means MASKED OUT
    mask = pad_mask | causal_mask

    return mask


# ══════════════════════════════════════════════════════════════════════
#  MULTI-HEAD ATTENTION 
# ══════════════════════════════════════════════════════════════════════

class MultiHeadAttention(nn.Module):
    """
    Multi-Head Attention as in "Attention Is All You Need", §3.2.2.

        MultiHead(Q,K,V) = Concat(head_1,...,head_h) · W_O
        head_i = Attention(Q·W_Qi, K·W_Ki, V·W_Vi)

    You are NOT allowed to use torch.nn.MultiheadAttention.

    Args:
        d_model   (int)  : Total model dimensionality. Must be divisible by num_heads.
        num_heads (int)  : Number of parallel attention heads h.
        dropout   (float): Dropout probability applied to attention weights.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model   = d_model
        self.num_heads = num_heads
        self.d_k       = d_model // num_heads   # depth per head
        
        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Final output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(p=dropout)

    
    def forward(
        self,
        query: torch.Tensor,
        key:   torch.Tensor,
        value: torch.Tensor,
        mask:  Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query : shape [batch, seq_q, d_model]
            key   : shape [batch, seq_k, d_model]
            value : shape [batch, seq_k, d_model]
            mask  : Optional BoolTensor broadcastable to
                    [batch, num_heads, seq_q, seq_k]
                    True → masked out (attend nowhere)

        Returns:
            output : shape [batch, seq_q, d_model]

        """

        batch_size = query.size(0)

        # Linear projections: [batch, seq_len, d_model]
        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)
        
        # Split into heads
        # [batch, seq_len, d_model] → [batch, seq_len, heads, d_k] → [batch, heads, seq_len, d_k]
        Q = Q.view(batch_size, -1, self.num_heads, self.d_k)
        K = K.view(batch_size, -1, self.num_heads, self.d_k)
        V = V.view(batch_size, -1, self.num_heads, self.d_k)

        Q = Q.transpose(1, 2)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)

        # Apply scaled dot-product attention
        attention_output, attention_weights = scaled_dot_product_attention(Q, K, V, mask)


        # Concatenate heads
        # [batch, heads, seq_len, d_k] → [batch, seq_len, heads, d_k] → [batch, seq_len, d_model]
        attention_output = attention_output.transpose(1, 2)

        attention_output = attention_output.contiguous().view(batch_size, -1, self.d_model)

        # Output projection
        output = self.W_o(attention_output)

        return output

# ══════════════════════════════════════════════════════════════════════
#   POSITIONAL ENCODING  
# ══════════════════════════════════════════════════════════════════════

class PositionalEncoding(nn.Module):
    """
    Sinusoidal Positional Encoding as in "Attention Is All You Need", §3.5.

    Args:
        d_model  (int)  : Embedding dimensionality.
        dropout  (float): Dropout applied after adding encodings.
        max_len  (int)  : Maximum sequence length to pre-compute (default 5000).
    """

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        
        self.dropout = nn.Dropout(p=dropout)

        # Positional encoding matrix. Shape: [max_len, d_model]
        pe = torch.zeros(max_len, d_model)

        # Position indices [max_len, 1]
        position = torch.arange(0, max_len).unsqueeze(1)

        # Frequency scaling term
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        
        pe[:, 0::2] = torch.sin(position * div_term) # Apply sin to even indices
        pe[:, 1::2] = torch.cos(position * div_term) # Apply cos to odd indices

        # Add batch dimension [1, max_len, d_model]
        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe) # Not a parameter, but should be saved/loaded with the model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : Input embeddings, shape [batch, seq_len, d_model]

        Returns:
            Tensor of same shape [batch, seq_len, d_model]
            = x  +  PE[:, :seq_len, :]  

        """
        
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len]
        
        return self.dropout(x)


# ══════════════════════════════════════════════════════════════════════
#  FEED-FORWARD NETWORK 
# ══════════════════════════════════════════════════════════════════════

class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-Forward Network, §3.3:

        FFN(x) = max(0, x·W₁ + b₁)·W₂ + b₂

    Args:
        d_model (int)  : Input / output dimensionality (e.g. 512).
        d_ff    (int)  : Inner-layer dimensionality (e.g. 2048).
        dropout (float): Dropout applied between the two linears.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: Task 2.3 — define:

        self.linear1 = nn.Linear(d_model, d_ff)
        
        self.linear2 = nn.Linear(d_ff, d_model)
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x : shape [batch, seq_len, d_model]
        Returns:
              shape [batch, seq_len, d_model]
        
        """
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER LAYER  
# ══════════════════════════════════════════════════════════════════════

class EncoderLayer(nn.Module):
    """
    Single Transformer encoder sub-layer:
        x → [Self-Attention → Add & Norm] → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO:instantiate:
        
        # Self-attention module
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            shape [batch, src_len, d_model]

        """
        
        # Self-attention sublayer
        attn_output = self.self_attn(x, x, x, src_mask)

        # Residual + LayerNorm
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward sublayer
        ffn_output = self.ffn(x)

        # Residual + LayerNorm
        x = self.norm2(x + self.dropout(ffn_output))

        return x


# ══════════════════════════════════════════════════════════════════════
#   DECODER LAYER 
# ══════════════════════════════════════════════════════════════════════

class DecoderLayer(nn.Module):
    """
    Single Transformer decoder sub-layer:
        x → [Masked Self-Attn → Add & Norm]
          → [Cross-Attn(memory) → Add & Norm]
          → [FFN → Add & Norm]

    Args:
        d_model   (int)  : Model dimensionality.
        num_heads (int)  : Number of attention heads.
        d_ff      (int)  : FFN inner dimensionality.
        dropout   (float): Dropout probability.
    """

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        # TODO: instantiate:

        # Masked self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Encoder-decoder cross attention
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feed-forward network
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        # LayerNorms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(p=dropout)
        


    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : Encoder output, shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            shape [batch, tgt_len, d_model]
        """
        
        # Masked self-attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)

        x = self.norm1(x + self.dropout(self_attn_output))

        # Cross-attention - Query comes from decoder and Key/Value come from encoder
        cross_attn_output = self.cross_attn(x, memory, memory, src_mask)

        x = self.norm2(x + self.dropout(cross_attn_output))

        # Feed-forward network
        ffn_output = self.ffn(x)

        x = self.norm3(x + self.dropout(ffn_output))

        return x


# ══════════════════════════════════════════════════════════════════════
#  ENCODER & DECODER STACKS
# ══════════════════════════════════════════════════════════════════════

class Encoder(nn.Module):
    """Stack of N identical EncoderLayer modules with final LayerNorm."""

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()
        
        # Clone N independent encoder layers
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        # Final layer normalization
        self.norm = nn.LayerNorm(layer.self_attn.d_model)


    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x    : shape [batch, src_len, d_model]
            mask : shape [batch, 1, 1, src_len]
        Returns:
            shape [batch, src_len, d_model]
        """
        
        # Pass through all encoder layers
        for layer in self.layers:
            x = layer(x, mask)

        # Final normalization
        return self.norm(x)


class Decoder(nn.Module):
    """Stack of N identical DecoderLayer modules with final LayerNorm."""

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()
        
        # Clone N independent decoder layers
        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x:        torch.Tensor,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x        : shape [batch, tgt_len, d_model]
            memory   : shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]
        Returns:
            shape [batch, tgt_len, d_model]
        """
        
        # Pass through all decoder layers
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        # Final normalization
        return self.norm(x)
        


# ══════════════════════════════════════════════════════════════════════
#   FULL TRANSFORMER  
# ══════════════════════════════════════════════════════════════════════

class Transformer(nn.Module):
    """
    Full Encoder-Decoder Transformer for sequence-to-sequence tasks.

    Args:
        src_vocab_size (int)  : Source vocabulary size.
        tgt_vocab_size (int)  : Target vocabulary size.
        d_model        (int)  : Model dimensionality (default 512).
        N              (int)  : Number of encoder/decoder layers (default 6).
        num_heads      (int)  : Number of attention heads (default 8).
        d_ff           (int)  : FFN inner dimensionality (default 2048).
        dropout        (float): Dropout probability (default 0.1).
    """

    def __init__(
        self,
        src_vocab_size: int,
        tgt_vocab_size: int,
        d_model:   int   = 512,
        N:         int   = 6,
        num_heads: int   = 8,
        d_ff:      int   = 2048,
        dropout:   float = 0.1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()
        # TODO: Instantiate 
        # init should also load the model weights if checkpoint path provided, download the .pth file like this
        
        # ------------------------------------------------
        # Optional checkpoint download
        # ------------------------------------------------
        # if checkpoint_path is not None:

        #     gdown.download(
        #         id="<.pth drive id>",
        #         output=checkpoint_path,
        #         quiet=False
        #     )

        # ------------------------------------------------
        # Internal constants
        # ------------------------------------------------
        self.d_model = d_model

        self.pad_idx = 1

        self.max_seq_length = 5000

        # ------------------------------------------------
        # Load vocabularies
        # ------------------------------------------------
        with open("src_vocab.pkl", "rb") as f:
            self.src_vocab, self.src_itos = pickle.load(f)

        with open("tgt_vocab.pkl", "rb") as f:
            self.tgt_vocab, self.tgt_itos = pickle.load(f)

        # ------------------------------------------------
        # Load spaCy tokenizers
        # ------------------------------------------------
        self.de_tokenizer = spacy.load("de_core_news_sm")

        self.en_tokenizer = spacy.load("en_core_web_sm")

        # ------------------------------------------------
        # Embedding layers
        # ------------------------------------------------
        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=self.pad_idx
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=self.pad_idx
        )

        # ------------------------------------------------
        # Positional encoding
        # ------------------------------------------------
        self.positional_encoding = PositionalEncoding(
            d_model,
            dropout,
            self.max_seq_length
        )

        # ------------------------------------------------
        # Encoder
        # ------------------------------------------------
        encoder_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder = Encoder(
            encoder_layer,
            N
        )

        # ------------------------------------------------
        # Decoder
        # ------------------------------------------------
        decoder_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.decoder = Decoder(
            decoder_layer,
            N
        )

        # ------------------------------------------------
        # Final vocabulary projection
        # ------------------------------------------------
        self.fc_out = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        # ------------------------------------------------
        # Load checkpoint weights if provided
        # ------------------------------------------------
        if checkpoint_path is not None:

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu"
            )

            self.load_state_dict(
                checkpoint["model_state_dict"]
            )

    # ── AUTOGRADER HOOKS ── keep these signatures exactly ─────────────

    def encode(
        self,
        src:      torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full encoder stack.

        Args:
            src      : Token indices, shape [batch, src_len]
            src_mask : shape [batch, 1, 1, src_len]

        Returns:
            memory : Encoder output, shape [batch, src_len, d_model]
        """
    
        # Source embeddings
        src = self.src_embedding(src)

        src = src * math.sqrt(self.d_model)

        src = self.positional_encoding(src)

        # Encoder forward
        memory = self.encoder(
            src,
            src_mask
        )

        return memory
    

    def decode(
        self,
        memory:   torch.Tensor,
        src_mask: torch.Tensor,
        tgt:      torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Run the full decoder stack and project to vocabulary logits.

        Args:
            memory   : Encoder output,  shape [batch, src_len, d_model]
            src_mask : shape [batch, 1, 1, src_len]
            tgt      : Token indices,   shape [batch, tgt_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """

        # Target embeddings
        tgt = self.tgt_embedding(tgt)

        tgt = tgt * math.sqrt(self.d_model)

        tgt = self.positional_encoding(tgt)


        # Decoder forward
        decoder_output = self.decoder(
            tgt,
            memory,
            src_mask,
            tgt_mask
        )
        
        # Vocabulary projection
        logits = self.fc_out(decoder_output)

        return logits

    def forward(
        self,
        src:      torch.Tensor,
        tgt:      torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full encoder-decoder forward pass.

        Args:
            src      : shape [batch, src_len]
            tgt      : shape [batch, tgt_len]
            src_mask : shape [batch, 1, 1, src_len]
            tgt_mask : shape [batch, 1, tgt_len, tgt_len]

        Returns:
            logits : shape [batch, tgt_len, tgt_vocab_size]
        """
        # Encoder
        memory = self.encode(
            src,
            src_mask
        )

        # Decoder
        logits = self.decode(
            memory,
            src_mask,
            tgt,
            tgt_mask
        )

        return logits


    def infer(self, src_sentence: str) -> str:
        """
        Translates a German sentence to English using greedy autoregressive decoding.
        
        Args:
            src_sentence: The raw German text.
            
            
        Returns:
            The fully translated English string, detokenized and clean.
        """
        
        self.eval()  # Set model to evaluation mode

        device = next(self.parameters()).device  # Get model device

        # Tokenize German sentence
        tokens = [
            token.text.lower()
            for token in self.de_tokenizer(src_sentence)
        ]

        # Add <sos> and <eos>
        tokens = ["<sos>"] + tokens + ["<eos>"]

        # Numercalize
        src_indices = [
            self.src_vocab.get(token, self.src_vocab["<unk>"])
            for token in tokens
        ]

        # Create source tensor: [1, src_len]
        src_tensor = torch.tensor(src_indices, dtype=torch.long).unsqueeze(0).to(device)

        # Source mask
        src_mask = make_src_mask(src_tensor, self.pad_idx).to(device)

        # Encode source sentence
        with torch.no_grad():
            memory = self.encode(src_tensor, src_mask)

        
        # Initialize decoder input
        tgt_indices = [self.tgt_vocab["<sos>"]]
        max_decode_len = 100  # Prevent infinite loops

        # Autoregressive decoding loop
        for _ in range(max_decode_len):
            tgt_tensor = torch.tensor(
                tgt_indices,
                dtype=torch.long
            ).unsqueeze(0).to(device)

            tgt_mask = make_tgt_mask(
                tgt_tensor,
                self.pad_idx
            ).to(device)

            with torch.no_grad():

                output = self.decode(
                    memory,
                    src_mask,
                    tgt_tensor,
                    tgt_mask
                )

            # Last token prediction
            next_token_logits = output[:, -1, :]

            next_token = torch.argmax(
                next_token_logits,
                dim=-1
            ).item()

            tgt_indices.append(next_token)

            # Stop at EOS
            if next_token == self.tgt_vocab["<eos>"]:
                break


        # Convert indices back to tokens
        tgt_tokens = [
            self.tgt_itos[idx]
            for idx in tgt_indices
        ]

        # Remove special tokens
        filtered_tokens = []

        for token in tgt_tokens:

            if token in ["<sos>", "<eos>", "<pad>"]:
                continue

            filtered_tokens.append(token)

        # Detokenize 
        translated_sentence = " ".join(filtered_tokens)

        return translated_sentence
    

# #   MAIN TEST BLOCK (for quick dummy testing)
# if __name__ == "__main__":
#     import torch
#     import os
#     # Dummy vocab files for test (if not present, create minimal ones)
#     if not os.path.exists("src_vocab.pkl"):
#         import pickle
#         src_vocab = {"<sos>":0, "<pad>":1, "<eos>":2, "<unk>":3, "hallo":4}
#         src_itos = ["<sos>", "<pad>", "<eos>", "<unk>", "hallo"]
#         with open("src_vocab.pkl", "wb") as f:
#             pickle.dump((src_vocab, src_itos), f)
#     if not os.path.exists("tgt_vocab.pkl"):
#         import pickle
#         tgt_vocab = {"<sos>":0, "<pad>":1, "<eos>":2, "<unk>":3, "hello":4}
#         tgt_itos = ["<sos>", "<pad>", "<eos>", "<unk>", "hello"]
#         with open("tgt_vocab.pkl", "wb") as f:
#             pickle.dump((tgt_vocab, tgt_itos), f)

#     # Make sure spacy models are available
#     try:
#         import spacy
#         spacy.load("de_core_news_sm")
#     except Exception:
#         os.system("python -m spacy download de_core_news_sm")
#     try:
#         spacy.load("en_core_web_sm")
#     except Exception:
#         os.system("python -m spacy download en_core_web_sm")

#     # Model params
#     src_vocab_size = 5
#     tgt_vocab_size = 5
#     d_model = 8
#     N = 2
#     num_heads = 2
#     d_ff = 16
#     dropout = 0.1

#     # Instantiate model
#     model = Transformer(
#         src_vocab_size=src_vocab_size,
#         tgt_vocab_size=tgt_vocab_size,
#         d_model=d_model,
#         N=N,
#         num_heads=num_heads,
#         d_ff=d_ff,
#         dropout=dropout
#     )

#     # Dummy input tensors
#     batch = 2
#     src_len = 4
#     tgt_len = 4
#     src = torch.randint(0, src_vocab_size, (batch, src_len))
#     tgt = torch.randint(0, tgt_vocab_size, (batch, tgt_len))
#     src_mask = make_src_mask(src, pad_idx=1)
#     tgt_mask = make_tgt_mask(tgt, pad_idx=1)

#     # Forward pass
#     out = model(src, tgt, src_mask, tgt_mask)
#     print("Output shape:", out.shape)
#     # Test inference with a dummy sentence
#     try:
#         print("Inference output:", model.infer("hallo"))
#     except Exception as e:
#         print("Inference failed:", e)