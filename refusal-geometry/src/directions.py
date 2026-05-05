"""
Refusal-direction extraction from model activations.

get_mean_activations() is the core routine: it registers a lightweight pre-forward
hook on every transformer block, runs a set of prompts through the model in batches,
and accumulates the mean of the last-token hidden state at each layer's input.

A refusal direction is then obtained externally by subtracting two such tensors:
    refusal_directions = get_mean_activations(harmful) - get_mean_activations(harmless)

The resulting tensor has shape [n_layers, hidden_size]; each row is the refusal
direction for that layer.  Layer selection (choosing which row to use) is handled
in the evaluation module via eval_refusal_with_ablation.
"""

import torch
from tqdm.notebook import tqdm

from .data import tokenize
from .hooks import add_hooks


def get_mean_activations(model, tokenizer, prompts, n_layers, hidden_size, batch_size=8):
    """
    Compute the mean residual-stream activation at each layer across a set of prompts.

    Activations are measured at the **last token position** of each prompt — the
    position from which the model will generate its first output token and therefore
    the site where the refuse/don't-refuse decision is made.

    A pre-forward hook on each transformer block records the first element of the
    module's input tuple (the hidden-state tensor before that block runs).  Arithmetic
    accumulates in float64 to avoid half-precision rounding across many prompts.

    Args:
        model:       The Qwen AutoModelForCausalLM instance (already on the target device).
        tokenizer:   Matching tokenizer (as returned by load_model).
        prompts:     List of raw instruction strings.  Each is wrapped in the Qwen
                     chat template before tokenization.
        n_layers:    Number of transformer blocks (N_LAYERS from load_model).
        hidden_size: Residual-stream width (HIDDEN_SIZE from load_model).
        batch_size:  Number of prompts per forward pass (default 8).

    Returns:
        Float64 tensor of shape [n_layers, hidden_size] on the model's device.
        Entry [i] is the mean last-token hidden state entering block i,
        averaged over all prompts.
    """
    mean_acts = torch.zeros(n_layers, hidden_size, dtype=torch.float64, device=model.device)

    def make_hook(layer_idx):
        def hook_fn(module, input):
            acts = input[0].clone().to(torch.float64)
            mean_acts[layer_idx] += acts[:, -1, :].sum(dim=0)
        return hook_fn

    hooks = [(model.transformer.h[i], make_hook(i)) for i in range(n_layers)]

    for i in tqdm(range(0, len(prompts), batch_size), desc="Forward passes", leave=False):
        batch = prompts[i : i + batch_size]
        tokens = tokenize(batch, tokenizer)
        with add_hooks(hooks):
            model(
                input_ids=tokens.input_ids.to(model.device),
                attention_mask=tokens.attention_mask.to(model.device),
            )

    mean_acts /= len(prompts)
    return mean_acts
