"""
CSE 676 – Deep Learning  |  Spring 2026
Assignment 2: TinyGPT
=====================

Instructions
------------
1.  Implement the TODO sections:
      Part 1 (TODOs 1.1–1.6): model components and training pipeline.
    Do NOT modify anything outside the TODO blocks.

2.  Do NOT rename classes or functions, and do NOT change any signature
    (argument names, order, or return type contract).  The autograder
    imports this file and calls functions by name.

3.  Once Part 1 is complete, run:
        python gpt_template.py --mode all

    This trains the baseline model and generates text samples.  Inspect
    training_log.json and generated_text.json to write your report.

4.  Validate your JSON before submitting:
        python -m json.tool training_log.json
        python -m json.tool generated_text.json
"""

import argparse
import json
import math
import os
import random
import time
import urllib.request
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Seed utilities  (do not modify)
# ---------------------------------------------------------------------------

SEED = 42   # fixed seed used for all random operations in this assignment


def set_all_seeds(seed: int) -> None:
    """Seed Python's random, NumPy, and PyTorch in one call."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Baseline configuration  (do not modify)
# ---------------------------------------------------------------------------

BASELINE_CONFIG = {
    "block_size":      128,   # context window length
    "embed_dim":        64,   # token embedding dimension D
    "num_heads":         4,   # number of attention heads
    "num_layers":        4,   # number of transformer blocks
    "mlp_dim":         128,   # MLP hidden dimension (2 × embed_dim)
    "dropout":         0.1,
    "lr":             3e-4,
    "batch_size":       64,
    "epochs":            5,
    "steps_per_epoch": 200,   # gradient steps per epoch
}

CHECKPOINT_EPOCHS = (1, 3, 5)
DATA_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master"
    "/data/tinyshakespeare/input.txt"
)


# ---------------------------------------------------------------------------
# Dataset  (provided – do not modify)
# ---------------------------------------------------------------------------

class CharDataset(Dataset):
    """
    Sliding-window character-level dataset.

    Each item is a pair (x, y) where:
      x = data[idx : idx + block_size]     input context
      y = data[idx+1 : idx + block_size+1] target (shifted by one position)
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data       = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        x = self.data[idx     : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# ---------------------------------------------------------------------------
# TODO 1.1  get_dataset
# ---------------------------------------------------------------------------

def get_dataset(
    data_root:  str   = "data",
    block_size: int   = 128,
    train_frac: float = 0.9,
):
    """
    Download TinyShakespeare if needed, build a character vocabulary, and
    return training / validation CharDataset objects with vocabulary metadata.

    Steps
    -----
    1.  Create data_root with os.makedirs if it does not exist.
    2.  Download DATA_URL to os.path.join(data_root, 'input.txt') if the
        file is not already present.  Use urllib.request.urlretrieve.
    3.  Read the full text from 'input.txt'.
    4.  Build the vocabulary: chars = sorted(set(text)), vocab_size = len(chars).
        Construct stoi (char→int) and itos (int→char) dicts.
    5.  Encode the entire text as a LongTensor using stoi.
    6.  Split: first train_frac of the data is training; rest is validation.
    7.  Wrap each split in CharDataset(split_data, block_size) and return.

    Args:
        data_root  (str):   Directory to store/read 'input.txt'.
        block_size (int):   Context window length.
        train_frac (float): Fraction of data for training.

    Returns:
        train_dataset (CharDataset)
        val_dataset   (CharDataset)
        vocab_size    (int)
        stoi          (dict[str, int])
        itos          (dict[int, str])
    """
    # TODO 1.1: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 1.2  CausalSelfAttention
# ---------------------------------------------------------------------------

class CausalSelfAttention(nn.Module):
    """
    Multi-head causal self-attention.

    Each token at position i may attend only to tokens at positions ≤ i.
    Positions i+1, i+2, … are masked to −∞ before the softmax so they
    receive zero attention weight.

    Args:
        embed_dim  (int):   Total embedding dimension D (must be divisible by num_heads).
        num_heads  (int):   Number of attention heads h.
        block_size (int):   Maximum sequence length; used to pre-allocate the mask.
        dropout    (float): Dropout applied to attention weights and the residual output.

    Required attributes (autograder checks these names):
        self.qkv        nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj   nn.Linear(embed_dim, embed_dim, bias=False)
        self.attn_drop  nn.Dropout
        self.resid_drop nn.Dropout
        self.mask       lower-triangular buffer shape (1, 1, block_size, block_size)

    Do NOT use nn.MultiheadAttention.

    Forward:
        x : Tensor (B, T, D)   where T ≤ block_size
    Returns:
        Tensor (B, T, D)
    """

    def __init__(
        self,
        embed_dim:  int,
        num_heads:  int,
        block_size: int,
        dropout:    float = 0.0,
    ):
        super().__init__()
        # TODO 1.2 – __init__:
        # assert embed_dim % num_heads == 0
        # store self.num_heads, self.head_dim (= embed_dim // num_heads), self.embed_dim
        # create self.qkv, self.out_proj, self.attn_drop, self.resid_drop
        # create the causal mask and register it as a buffer named "mask"
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 1.2 – forward:
        # 1. Unpack B, T, C from x.shape
        # 2. Project x through self.qkv and split into q, k, v (each shape (B, T, C))
        # 3. Reshape each to (B, num_heads, T, head_dim)
        # 4. Compute scaled dot-product: (q @ k^T) / sqrt(head_dim)
        # 5. Apply the causal mask: fill positions where mask == 0 with -inf
        # 6. Softmax over the last dimension, then apply self.attn_drop
        # 7. Multiply by v, reshape back to (B, T, C)
        # 8. Apply self.out_proj and self.resid_drop
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 1.3  GPTBlock
# ---------------------------------------------------------------------------

class GPTBlock(nn.Module):
    """
    A single GPT transformer block (decoder-only, pre-norm).

    Pre-norm means LayerNorm is applied BEFORE the sub-layer (not after):
        x ← x + CausalSelfAttention(LayerNorm(x))
        x ← x + MLP(LayerNorm(x))

    The MLP is a two-layer feed-forward network:
        Linear(embed_dim → mlp_dim) → GELU → Linear(mlp_dim → embed_dim) → Dropout

    Args:
        embed_dim  (int):   Embedding dimension D.
        num_heads  (int):   Number of attention heads.
        block_size (int):   Maximum sequence length.
        mlp_dim    (int):   Hidden dimension of the MLP.
        dropout    (float): Dropout probability.

    Required sub-module names (autograder checks):
        self.norm1  nn.LayerNorm(embed_dim)
        self.attn   CausalSelfAttention(...)
        self.norm2  nn.LayerNorm(embed_dim)
        self.mlp    nn.Sequential(Linear, GELU, Linear, Dropout)

    Forward:
        x : Tensor (B, T, D)
    Returns:
        Tensor (B, T, D)
    """

    def __init__(
        self,
        embed_dim:  int,
        num_heads:  int,
        block_size: int,
        mlp_dim:    int,
        dropout:    float = 0.0,
    ):
        super().__init__()
        # TODO 1.3 – __init__: create norm1, attn, norm2, mlp
        raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO 1.3 – forward: apply pre-norm residual connections
        raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 1.4  GPT
# ---------------------------------------------------------------------------

class GPT(nn.Module):
    """
    TinyGPT: a decoder-only transformer language model.

    Architecture (in order):
        1. Token embedding    nn.Embedding(vocab_size, embed_dim)
        2. Position embedding nn.Embedding(block_size, embed_dim)
        3. Dropout on summed embeddings
        4. num_layers × GPTBlock
        5. Final LayerNorm
        6. Linear LM head (weight-tied with token embedding)

    Weight tying  ←  this is important:
        After creating self.head, set
            self.head.weight = self.token_embedding.weight
        Both the input embedding and the output projection then share one
        matrix of shape (vocab_size, embed_dim), reducing parameter count
        and typically improving perplexity.

    Args:
        vocab_size (int):   Number of unique tokens (characters).
        block_size (int):   Maximum context length.
        embed_dim  (int):   Embedding dimension D.
        num_heads  (int):   Number of attention heads per block.
        num_layers (int):   Number of GPTBlocks.
        mlp_dim    (int):   MLP hidden dimension.
        dropout    (float): Dropout probability.

    Required attribute names (autograder checks):
        self.token_embedding  nn.Embedding
        self.pos_embedding    nn.Embedding
        self.drop             nn.Dropout
        self.blocks           nn.ModuleList of GPTBlock
        self.norm             nn.LayerNorm
        self.head             nn.Linear (weight-tied)

    Forward:
        idx : LongTensor (B, T)  token indices, T ≤ block_size
    Returns:
        logits : FloatTensor (B, T, vocab_size)
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        embed_dim:  int,
        num_heads:  int,
        num_layers: int,
        mlp_dim:    int,
        dropout:    float = 0.0,
    ):
        super().__init__()
        self.block_size = block_size
        # TODO 1.4 – __init__: create all sub-modules and tie weights
        raise NotImplementedError

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        # TODO 1.4 – forward:
        # 1. Unpack B, T; assert T <= self.block_size
        # 2. Compute token and position embeddings; add them; apply dropout
        # 3. Pass through each block in self.blocks
        # 4. Apply self.norm and self.head
        raise NotImplementedError


def build_model(config: dict, vocab_size: int) -> "GPT":
    """Construct a GPT from a config dict and vocab_size.  (Do not modify.)"""
    return GPT(
        vocab_size = vocab_size,
        block_size = config["block_size"],
        embed_dim  = config["embed_dim"],
        num_heads  = config["num_heads"],
        num_layers = config["num_layers"],
        mlp_dim    = config["mlp_dim"],
        dropout    = config["dropout"],
    )


# ---------------------------------------------------------------------------
# TODO 1.5  train_model
# ---------------------------------------------------------------------------

def train_model(
    model:          "GPT",
    train_dataset:  CharDataset,
    val_dataset:    CharDataset,
    config:         dict,
    checkpoint_dir: str = "checkpoints",
    log_path:       str = "training_log.json",
) -> dict:
    """
    Train the GPT model for next-character prediction.

    Requirements
    ------------
    Optimizer:   AdamW, lr = config["lr"], weight_decay = 1e-2
    LR schedule: CosineAnnealingLR with T_max = config["epochs"]
    Loss:        cross-entropy over ALL token positions in the batch
                 (flatten logits to (B*T, V) and targets to (B*T,))
    Gradient clipping: torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    Each epoch:  sample batches from a shuffled DataLoader until
                 config["steps_per_epoch"] gradient steps are done.
    Validation:  after each epoch, evaluate on up to 50 batches from
                 val_dataset (no gradient).
    Checkpoints: save at every epoch in CHECKPOINT_EPOCHS.
    Logging:     after all epochs, write training_log.json.

    JSON schema (see assignment.pdf Section 5 for full spec)
    --------------------------------------------------------
    {
      "seed"           : <int>,
      "config"         : { ... },
      "history"        : [
         {"epoch": 1, "train_loss": <float>, "val_loss": <float>,
          "epoch_time_sec": <float>},
         ...
      ],
      "final_val_loss" : <float>,
      "total_params"   : <int>
    }

    Checkpoint format
    -----------------
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {k: config[k] for k in
                   ["block_size","embed_dim","num_heads","num_layers","mlp_dim","dropout"]},
        "epoch": <int>,
    }, os.path.join(checkpoint_dir, f"gpt_epoch_{epoch}.pt"))

    Returns
    -------
    dict : the training log (same as what is written to log_path)
    """
    # TODO 1.5: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# TODO 1.6  generate
# ---------------------------------------------------------------------------

def generate(
    model:          "GPT",
    itos:           dict,
    stoi:           dict,
    prompt:         str,
    max_new_tokens: int   = 200,
    temperature:    float = 1.0,
) -> str:
    """
    Generate text autoregressively from a prompt string.

    Algorithm (repeat max_new_tokens times):
        1. Encode the current context to a LongTensor (1, T).
           Crop to the last block_size tokens if T > block_size.
        2. Forward pass → logits of shape (1, T, vocab_size).
        3. Take logits for the last position: logits[:, -1, :].
        4. Divide by temperature, then softmax to get probabilities.
        5. Sample one token with torch.multinomial(probs, num_samples=1).
        6. Append the new token index to the context.
    Decode the full context back to a string via itos and return it.

    Args:
        model          : trained GPT (set to eval mode AND wrap the generation
                         loop in torch.no_grad() inside this function — both
                         are required to disable dropout and avoid building an
                         unnecessary computation graph during inference)
        itos           : index-to-character dict
        stoi           : character-to-index dict
        prompt         : initial conditioning string (included in output)
        max_new_tokens : number of new characters to generate
        temperature    : > 1 increases randomness, < 1 sharpens the distribution;
                         temperature = 1.0 is standard sampling

    Returns:
        str : full generated text (prompt + new characters)
    """
    # TODO 1.6: implement
    raise NotImplementedError


# ---------------------------------------------------------------------------
# Run-all pipeline  (do not modify)
# ---------------------------------------------------------------------------

GENERATION_PROMPT = "HAMLET:\n"


def run_all(config: dict = None):
    """Train TinyGPT and generate text samples."""
    if config is None:
        config = BASELINE_CONFIG.copy()

    seed = SEED
    set_all_seeds(seed)

    print("Loading dataset …")
    train_ds, val_ds, vocab_size, stoi, itos = get_dataset(
        data_root="data", block_size=config["block_size"]
    )
    print(
        f"  vocab_size={vocab_size}, "
        f"train_samples={len(train_ds)}, "
        f"val_samples={len(val_ds)}"
    )
    config["vocab_size"] = vocab_size

    model        = build_model(config, vocab_size)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {total_params:,}")

    print("\nTraining …")
    log = train_model(
        model, train_ds, val_ds,
        config,
        checkpoint_dir="checkpoints",
        log_path="training_log.json",
    )

    print("\nGenerating text samples …")
    os.makedirs("generated", exist_ok=True)
    samples = {}

    for ckpt_epoch in CHECKPOINT_EPOCHS:
        ckpt_path = os.path.join("checkpoints", f"gpt_epoch_{ckpt_epoch}.pt")
        if not os.path.isfile(ckpt_path):
            continue

        ckpt       = torch.load(ckpt_path, map_location="cpu")
        ckpt_model = build_model(ckpt["config"], vocab_size)
        ckpt_model.load_state_dict(ckpt["model_state_dict"])

        epoch_samples: dict = {"epoch": ckpt_epoch, "prompt": GENERATION_PROMPT}
        for temp in [0.5, 1.0, 1.5]:
            set_all_seeds(seed + ckpt_epoch * 100 + int(temp * 10))
            text = generate(
                ckpt_model, itos, stoi, GENERATION_PROMPT,
                max_new_tokens=300, temperature=temp,
            )
            epoch_samples[f"temperature_{temp}"] = text

        samples[f"checkpoint_{ckpt_epoch}"] = epoch_samples
        preview = epoch_samples["temperature_1.0"][:80].replace("\n", " ")
        print(f"  epoch {ckpt_epoch}, T=1.0: {preview} …")

    with open("generated_text.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("\nDone!")
    print("  training_log.json   – training curve")
    print("  generated_text.json – text samples")
    print("  checkpoints/        – saved model weights")
    return log, samples


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="TinyGPT – CSE 676 Assignment 2"
    )
    parser.add_argument(
        "--mode",
        choices=["train", "all"],
        default="all",
    )
    args = parser.parse_args()

    run_all()
