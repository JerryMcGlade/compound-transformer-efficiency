#!/usr/bin/env python3
"""
moe_dispersion_probe.py
-----------------------
Measures the per-expert token-count distribution N_e produced by a trained MoE
router, to settle the load-bearing empirical question behind the contribution:

    Is the expert-GEMM shape distribution D_MoE actually DISPERSED
    (so routing-aware algorithm selection has real headroom),
    or is the router close to BALANCED
    (so it collapses back toward static, FalconGEMM-style selection)?

It runs the router on a batch of varied prompts in two regimes:
  - PREFILL : full prompt sequences (many tokens per forward pass)
  - DECODE  : one token per sequence (the autoregressive-step shape)
and, for every MoE layer, tallies how many tokens are routed to each expert.
The M-dimension of each expert's GEMM is exactly that token count, so this
distribution IS the thing the multiplication backend sees.

Outputs:
  - moe_dispersion.png : histograms of N_e (prefill vs decode)
  - console report     : per-layer and aggregate dispersion stats + a verdict

NOTE ON DEPLOYMENT PADDING: this probe measures the *underlying* routing
distribution from raw forward passes. A production serving stack may pad each
expert to a fixed "capacity" (dropping overflow) to regularize shapes. If so,
the backend sees flat shapes even when the underlying routing is dispersed.
That does not make a dispersed result here meaningless -- it relocates the
argument to "padding wastes work *because* the true distribution is dispersed."
A near-balanced result here, by contrast, weakens the thesis regardless of
padding. Read the verdict with that distinction in mind.

Hardware: the default model needs ~14 GB. If your 16 GB Mac is tight, run this
on a free Colab GPU (Runtime > Change runtime type > T4) -- it fits comfortably.

Usage:
  pip install torch transformers matplotlib numpy accelerate
  python moe_dispersion_probe.py
  python moe_dispersion_probe.py --model allenai/OLMoE-1B-7B-0924
  python moe_dispersion_probe.py --prompts my_prompts.txt   # one prompt per line
"""

import argparse
import sys
import numpy as np


# ----------------------------------------------------------------------
# Stats / reporting (pure numpy -- unit-testable without a model)
# ----------------------------------------------------------------------
def layer_stats(counts):
    """counts: 1-D array of length n_experts (tokens routed to each expert).
    Returns a dict of dispersion statistics for one MoE layer."""
    counts = np.asarray(counts, dtype=np.float64)
    mean = counts.mean()
    nonzero = counts[counts > 0]
    return {
        "mean": mean,
        "std": counts.std(),
        "cv": (counts.std() / mean) if mean > 0 else 0.0,   # coefficient of variation
        "min": counts.min(),
        "max": counts.max(),
        # ratio of the largest to the smallest *active* expert GEMM -- what the
        # backend experiences as shape spread (guard against div-by-zero):
        "max_over_min": (counts.max() / nonzero.min()) if nonzero.size else float("inf"),
        "n_idle": int((counts == 0).sum()),
    }


def aggregate_report(per_layer_counts, label):
    """per_layer_counts: list of 1-D arrays (one per MoE layer).
    Prints a report and returns (mean_cv, all_counts_flat)."""
    stats = [layer_stats(c) for c in per_layer_counts]
    cvs = np.array([s["cv"] for s in stats])
    mom = np.array([s["max_over_min"] for s in stats if np.isfinite(s["max_over_min"])])
    all_counts = np.concatenate([np.asarray(c) for c in per_layer_counts]) if per_layer_counts else np.array([])

    print(f"\n=== {label} ===")
    print(f"  MoE layers measured      : {len(stats)}")
    print(f"  mean coeff. of variation : {cvs.mean():.3f}   (0 = perfectly balanced)")
    print(f"  CV range across layers   : {cvs.min():.3f} .. {cvs.max():.3f}")
    if mom.size:
        print(f"  median max/min GEMM ratio: {np.median(mom):.2f}x")
    if all_counts.size:
        print(f"  N_e overall              : min={all_counts.min():.0f}  "
              f"median={np.median(all_counts):.0f}  max={all_counts.max():.0f}")
    return cvs.mean(), all_counts


def verdict(prefill_cv, decode_cv):
    cv = max(prefill_cv, decode_cv)
    print("\n=== VERDICT (heuristic -- interpret, don't treat as law) ===")
    if cv < 0.10:
        msg = ("NEAR-BALANCED. Underlying routing is close to uniform. Claim 1 "
               "(dispersion) is weak even before deployment padding; the "
               "routing-aware thesis likely needs to pivot to imbalance/overflow/"
               "communication regimes rather than shape dispersion.")
    elif cv < 0.40:
        msg = ("MODERATELY DISPERSED. There is real shape spread but it is not "
               "dramatic. Routing-aware selection has headroom; whether it beats "
               "static selection end-to-end is now an empirical (not conceptual) "
               "question -- proceed to Claim 2.")
    else:
        msg = ("STRONGLY DISPERSED. The router produces widely varying expert "
               "GEMM shapes. Claim 1 holds for the underlying distribution; the "
               "remaining question is whether your target serving stack pads it "
               "away (if it does, that padding-waste becomes your contribution).")
    print("  max CV across regimes:", f"{cv:.3f}")
    print(" ", msg)


def plot(prefill_counts, decode_counts, path="moe_dispersion.png"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pf = np.concatenate([np.asarray(c) for c in prefill_counts]) if prefill_counts else np.array([])
    dc = np.concatenate([np.asarray(c) for c in decode_counts]) if decode_counts else np.array([])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, data, title in ((axes[0], pf, "Prefill"), (axes[1], dc, "Decode (1 token/seq)")):
        if data.size:
            ax.hist(data, bins=40, color="#4C78A8", edgecolor="white")
            ax.axvline(data.mean(), color="#E45756", linestyle="--", linewidth=1.5,
                       label=f"mean={data.mean():.0f}")
            ax.legend()
        ax.set_title(f"{title}: tokens routed per expert (N_e)")
        ax.set_xlabel("N_e  (= expert GEMM M-dimension)")
        ax.set_ylabel("frequency across experts x layers")
    fig.suptitle("MoE expert-GEMM shape distribution  (D_MoE)")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nSaved histogram -> {path}")


# ----------------------------------------------------------------------
# Model probing (needs torch + transformers)
# ----------------------------------------------------------------------
DEFAULT_PROMPTS = [
    "Explain how a suspension bridge distributes load across its cables.",
    "What is the role of ATP in muscle contraction?",
    "Summarize the causes of the 2008 financial crisis.",
    "Write a haiku about the ocean at dawn.",
    "Derive the time complexity of merge sort.",
    "How do mRNA vaccines train the immune system?",
    "Translate 'good morning, how are you?' into French.",
    "Describe the plot of Hamlet in two sentences.",
    "What distinguishes a Roth IRA from a traditional IRA?",
    "Explain backpropagation to a first-year student.",
    "List three causes of coastal erosion.",
    "What happens thermodynamically when water boils?",
    "Give a recipe outline for sourdough bread.",
    "Why is the sky blue during the day but red at sunset?",
    "Compare TCP and UDP at a high level.",
    "What is the biomechanical function of the Achilles tendon?",
]


def get_counts(model, tokenizer, prompts, device, n_experts, top_k):
    """Returns (prefill_counts, decode_counts): each a list over MoE layers of
    1-D numpy arrays of length n_experts."""
    import torch

    def counts_from_router_logits(router_logits, mask_flat):
        """router_logits: tuple over layers, each (n_tokens, n_experts).
        mask_flat: 1-D bool tensor selecting real (non-pad) token rows."""
        per_layer = []
        for lg in router_logits:
            lg = lg[mask_flat]                       # drop padding rows
            topk = torch.topk(lg, k=top_k, dim=-1).indices.reshape(-1)
            c = torch.bincount(topk, minlength=n_experts).float().cpu().numpy()
            per_layer.append(c)
        return per_layer

    # ---- PREFILL: full sequences ----
    enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True,
                    max_length=64).to(device)
    with torch.no_grad():
        out = model(**enc, output_router_logits=True)
    mask_flat = enc["attention_mask"].reshape(-1).bool()
    prefill = counts_from_router_logits(out.router_logits, mask_flat)

    # ---- DECODE proxy: last real token of each sequence, one token per seq ----
    last_ids = []
    for i in range(enc["input_ids"].shape[0]):
        valid = enc["input_ids"][i][enc["attention_mask"][i].bool()]
        last_ids.append(valid[-1])
    dec_in = torch.stack(last_ids).unsqueeze(1).to(device)   # (B, 1)
    with torch.no_grad():
        out_d = model(input_ids=dec_in, output_router_logits=True)
    mask_d = torch.ones(dec_in.shape[0], dtype=torch.bool, device=device)
    decode = counts_from_router_logits(out_d.router_logits, mask_d)

    return prefill, decode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--prompts", default=None, help="file with one prompt per line")
    ap.add_argument("--top_k", type=int, default=None, help="override num_experts_per_tok")
    args = ap.parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        sys.exit("Install deps first:  pip install torch transformers matplotlib numpy accelerate")

    prompts = DEFAULT_PROMPTS
    if args.prompts:
        with open(args.prompts) as f:
            prompts = [ln.strip() for ln in f if ln.strip()]

    device = ("cuda" if torch.cuda.is_available()
              else "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()
              else "cpu")
    dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    print(f"Loading {args.model} on {device} ({dtype}) ...")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()

    cfg = model.config
    n_experts = getattr(cfg, "num_experts", None) or getattr(cfg, "num_local_experts", None)
    top_k = args.top_k or getattr(cfg, "num_experts_per_tok", None)
    if not n_experts or not top_k:
        sys.exit("Could not read num_experts / num_experts_per_tok from config; "
                 "pass --top_k and check this model exposes output_router_logits.")
    print(f"  n_experts={n_experts}  top_k={top_k}")

    prefill, decode = get_counts(model, tokenizer, prompts, device, n_experts, top_k)

    pf_cv, _ = aggregate_report(prefill, "PREFILL")
    dc_cv, _ = aggregate_report(decode, "DECODE")
    verdict(pf_cv, dc_cv)
    plot(prefill, decode)


if __name__ == "__main__":
    main()
