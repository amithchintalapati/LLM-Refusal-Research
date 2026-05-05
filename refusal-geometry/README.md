# refusal-geometry

Experiments on the geometry of LLM refusal behaviour, built on **Qwen-1.8B-Chat** and
structured across two notebooks.

---

## Background

### Arditi et al. (2024) — *Refusal in Language Models Is Mediated by a Single Direction*

[Paper](https://arxiv.org/abs/2406.11717)

The central finding is deceptively simple: safety fine-tuning (RLHF/DPO) encodes
refusal in a **single linear direction** in the model's residual stream.  The authors
show that:

- **Ablating** (projecting out) this direction from all layers causes the model to
  comply with requests it would otherwise refuse.
- **Adding** the direction to activations for benign prompts causes the model to
  refuse them as if they were harmful.

The refusal direction is extracted as the difference in mean activations between harmful
and harmless prompts, and the best layer is selected by measuring how much each layer's
direction suppresses refusal when ablated.

### Joad et al. (2026) — cross-category geometry extension

This project extends Arditi et al. by asking: **does a single refusal direction capture
all categories of harmful content, or does each category have a distinct direction?**

Using [SorryBench-202406](https://huggingface.co/datasets/sorry-bench/sorry-bench-202406)
(Zeng et al., 2024) — 450 harmful prompts across 44 categories grouped into four policy
domains — we compute per-domain refusal directions and examine:

1. **Cosine similarity matrix**: how aligned are the four domain directions?
2. **Cross-category ablation transfer**: if we ablate domain A's direction, how well does
   that bypass refusal on domain B's prompts?

---

## Repository layout

```
refusal-geometry/
├── README.md
├── requirements.txt
├── src/
│   ├── __init__.py
│   ├── model.py        # load_model() — tokenizer, model, N_LAYERS, HIDDEN_SIZE
│   ├── data.py         # tokenize(), HARMFUL/HARMLESS_PROMPTS, load_sorry_bench()
│   ├── hooks.py        # hook factories, add_hooks, add_full_hooks, get_full_ablation_hooks
│   ├── directions.py   # get_mean_activations()
│   └── evaluation.py   # refusal_score(), generate(), generate_with_full_ablation(), is_refusal()
├── notebooks/
│   ├── 01_replication.ipynb   # Arditi et al. replication
│   └── 02_geometry.ipynb      # Per-category directions, cosine sim, cross-category transfer
└── results/
    └── figures/               # Saved plots
```

---

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU required.** Qwen-1.8B-Chat must run on a CUDA GPU (tested on T4 and A100
> via Google Colab).  Adjust `torch` to match your CUDA version:
> <https://pytorch.org/get-started/locally/>

### 2. HuggingFace token (notebook 02 only)

SorryBench is a gated dataset.  To use it:

1. Accept the terms at <https://huggingface.co/datasets/sorry-bench/sorry-bench-202406>
2. Create a Read token at <https://huggingface.co/settings/tokens>
3. Export it before running notebook 02:

   ```python
   import os
   os.environ["HF_TOKEN"] = "hf_..."
   ```

   In Google Colab, add it as a Secret named `HF_TOKEN` (🔑 sidebar).

---

## Running the notebooks

Both notebooks are designed for **Google Colab** (or any Jupyter environment with a
CUDA GPU).  Open them in order:

### `notebooks/01_replication.ipynb` — Arditi et al. replication

Demonstrates the core finding end-to-end:

1. Loads Qwen-1.8B-Chat via `src/model.py`.
2. Computes mean activations for 32 harmful and 32 harmless prompts
   (`src/directions.py`).
3. Selects the best layer by sweeping ablation refusal scores
   (`src/evaluation.py`).
4. Shows that **ablating** the refusal direction causes the model to answer
   harmful requests.
5. Shows that **adding** the direction causes the model to refuse harmless requests.

### `notebooks/02_geometry.ipynb` — Cross-category geometry

Extends the replication using SorryBench:

1. Loads the same model and SorryBench via `src/data.py`.
2. Computes a refusal direction for each of the four policy domains
   (HateSpeech, CrimeAssistance, Inappropriate, Advice).
3. Plots a **cosine similarity heatmap** of the four directions.
4. Runs a **cross-category ablation transfer** experiment: ablating domain A's
   direction and measuring the bypass rate on domain B's prompts.

---

## Key results (Qwen-1.8B-Chat, layer 14)

| Experiment | Finding |
|---|---|
| Layer selection | Layer 14 drops refusal score from +1.64 → −6.41 |
| Ablation | Model complies with all 4 tested harmful prompts |
| Addition | Model refuses all 4 tested harmless prompts |
| Cosine similarity | All four domain directions are highly aligned (≥ 0.85) |
| Cross-category transfer | Ablating any single domain's direction transfers well to others |

---

## Citation

```bibtex
@article{arditi2024refusal,
  title   = {Refusal in Language Models Is Mediated by a Single Direction},
  author  = {Arditi, Andy and Obeso, Oscar and Syed, Aaquib and Cunningham, Hoagy
             and Filan, Daniel and Biderman, Stella and Nanda, Neel},
  journal = {arXiv preprint arXiv:2406.11717},
  year    = {2024}
}

@article{zeng2024sorrybench,
  title   = {Sorry-Bench: Systematically Evaluating Large Language Model Safety
             Refusal Behaviors},
  author  = {Zeng, Tinghao and Xu, Boyuan and Huang, Yangsibo and Bhatt, Semih
             and Yu, Chiyuan and Shi, Tianyu and Hu, Zhuowen and Chen, Weijia
             and Li, Yangjun and Li, Luxi and others},
  journal = {arXiv preprint arXiv:2406.14598},
  year    = {2024}
}
```
