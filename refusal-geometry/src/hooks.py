"""
PyTorch forward-hook factories and context managers for activation intervention experiments.

Three hook types are provided:
  - Ablation hooks  (make_ablation_hook, make_ablation_output_hook)
    Project a direction out of the residual stream or a sublayer's output,
    effectively removing the model's ability to express that direction.
  - Addition hooks  (make_addition_hook)
    Inject a scaled direction into the residual stream.

Two context managers register and cleanly deregister hooks around a forward pass:
  - add_hooks       — for pre-forward hooks only (register_forward_pre_hook)
  - add_full_hooks  — for both pre-forward and forward hooks simultaneously

get_full_ablation_hooks() builds the complete set of hooks used in the paper's
full-ablation intervention: block inputs + attention outputs + MLP outputs.
"""

import contextlib


# ---------------------------------------------------------------------------
# Hook factories
# ---------------------------------------------------------------------------


def make_ablation_hook(direction):
    """
    Create a pre-forward hook that projects ``direction`` out of the residual stream.

    At each forward call the hook computes the scalar projection of each token's
    hidden state onto the (unit-normalised) direction and subtracts it, leaving
    activations orthogonal to the direction.  Arithmetic is performed in the
    dtype of the incoming activations to avoid unintended accumulation of
    float32/float16 casts.

    Args:
        direction: 1-D tensor of shape [hidden_size] — the direction to remove.

    Returns:
        Callable suitable for ``module.register_forward_pre_hook``.
    """
    dir_norm = direction / (direction.norm() + 1e-8)

    def hook_fn(module, input):
        acts = input[0]
        d = dir_norm.to(acts)
        proj = (acts @ d).unsqueeze(-1) * d
        return (acts - proj,) + input[1:] if len(input) > 1 else (acts - proj,)

    return hook_fn


def make_ablation_output_hook(direction):
    """
    Create a post-forward hook that projects ``direction`` out of a sublayer's output.

    Behaves identically to make_ablation_hook but operates on the module's
    *output* rather than its input.  Handles both bare tensors and tuple outputs
    (attention and MLP sublayers of Qwen return tuples whose first element is the
    hidden-state tensor).

    Args:
        direction: 1-D tensor of shape [hidden_size] — the direction to remove.

    Returns:
        Callable suitable for ``module.register_forward_hook``.
    """
    dir_norm = direction / (direction.norm() + 1e-8)

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            acts = output[0]
        else:
            acts = output
        d = dir_norm.to(acts)
        proj = (acts @ d).unsqueeze(-1) * d
        acts = acts - proj
        if isinstance(output, tuple):
            return (acts,) + output[1:]
        return acts

    return hook_fn


def make_addition_hook(direction, coeff=1.0):
    """
    Create a pre-forward hook that adds ``coeff * direction`` to the residual stream.

    Used to steer the model towards refusal on otherwise harmless prompts.

    Args:
        direction: 1-D tensor of shape [hidden_size] — the direction to add.
        coeff:     Scalar multiplier (default 1.0).  Negative values subtract.

    Returns:
        Callable suitable for ``module.register_forward_pre_hook``.
    """
    def hook_fn(module, input):
        acts = input[0]
        d = direction.to(acts)
        return (acts + coeff * d,) + input[1:] if len(input) > 1 else (acts + coeff * d,)

    return hook_fn


# ---------------------------------------------------------------------------
# Context managers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def add_hooks(hooks):
    """
    Temporarily register pre-forward hooks and remove them on exit.

    Args:
        hooks: Iterable of (module, hook_fn) pairs.  Each hook_fn is registered
               via ``module.register_forward_pre_hook``.

    Yields:
        None.  All hooks are removed whether the body raises or not.
    """
    handles = [mod.register_forward_pre_hook(fn) for mod, fn in hooks]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


@contextlib.contextmanager
def add_full_hooks(pre_hooks, output_hooks):
    """
    Temporarily register both pre-forward and forward hooks, then remove them.

    Args:
        pre_hooks:    Iterable of (module, hook_fn) for ``register_forward_pre_hook``.
        output_hooks: Iterable of (module, hook_fn) for ``register_forward_hook``.

    Yields:
        None.  All hooks are removed whether the body raises or not.
    """
    handles = []
    handles += [mod.register_forward_pre_hook(fn) for mod, fn in pre_hooks]
    handles += [mod.register_forward_hook(fn) for mod, fn in output_hooks]
    try:
        yield
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Full-ablation hook builder
# ---------------------------------------------------------------------------


def get_full_ablation_hooks(model, direction, n_layers):
    """
    Build the complete set of ablation hooks used in Arditi et al. (2024).

    The full intervention ablates ``direction`` from three sites per layer:
      1. The block's pre-forward input  (residual stream entering the block)
      2. The attention sublayer's output
      3. The MLP sublayer's output

    This ensures the direction cannot re-enter the residual stream through
    any of the three additive paths within a transformer block.

    Args:
        model:     The Qwen AutoModelForCausalLM instance.
        direction: 1-D tensor of shape [hidden_size] — the direction to ablate.
        n_layers:  Number of transformer blocks (N_LAYERS from load_model).

    Returns:
        Tuple (pre_hooks, output_hooks).
        Pass both to ``add_full_hooks(pre_hooks, output_hooks)`` to activate.
    """
    hooks_pre = [
        (model.transformer.h[i], make_ablation_hook(direction))
        for i in range(n_layers)
    ]
    hooks_out = []
    for i in range(n_layers):
        hooks_out.append((model.transformer.h[i].attn, make_ablation_output_hook(direction)))
        hooks_out.append((model.transformer.h[i].mlp, make_ablation_output_hook(direction)))
    return hooks_pre, hooks_out
