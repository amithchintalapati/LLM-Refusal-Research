"""
Model loading utilities and architecture constants for Qwen-1.8B-Chat refusal experiments.

Provides load_model(), which downloads the tokenizer and model weights, configures
padding, and returns the two architecture constants (N_LAYERS, HIDDEN_SIZE) that
hook and direction functions need.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "Qwen/Qwen-1_8B-Chat"


def load_model(model_path: str = MODEL_PATH):
    """
    Load the Qwen-1.8B-Chat tokenizer and causal LM.

    Configures left-padding (required for batch generation) and uses Qwen's
    dedicated pad token rather than EOS.  The model is loaded in float16,
    frozen, and placed on the best available device via device_map="auto".

    Args:
        model_path: HuggingFace model identifier.  Defaults to MODEL_PATH
                    ("Qwen/Qwen-1_8B-Chat").

    Returns:
        Tuple (tokenizer, model, N_LAYERS, HIDDEN_SIZE) where:
        - tokenizer:   AutoTokenizer with left-padding and Qwen pad token set.
        - model:       AutoModelForCausalLM in eval mode, float16, gradient-disabled.
        - N_LAYERS:    int — number of transformer blocks (24 for Qwen-1.8B-Chat).
        - HIDDEN_SIZE: int — residual-stream dimensionality (2048 for Qwen-1.8B-Chat).
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, use_fast=False
    )
    tokenizer.padding_side = "left"
    tokenizer.pad_token = "<|extra_0|>"
    tokenizer.pad_token_id = tokenizer.eod_id

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        use_flash_attn=False,
        fp16=True,
    ).eval()
    model.requires_grad_(False)

    n_layers = model.config.num_hidden_layers
    hidden_size = model.config.hidden_size

    return tokenizer, model, n_layers, hidden_size
