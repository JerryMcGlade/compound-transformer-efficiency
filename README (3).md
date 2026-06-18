# Routing-Induced GEMM Shape Dispersion in Mixture-of-Experts Inference

A small empirical study of how a trained Mixture-of-Experts (MoE) router spreads
tokens across its experts, and what that spread implies for the matrix-multiply
shapes an inference backend actually has to compute.

> **What this is:** an honest measurement, on one open model, of a known
> phenomenon (MoE load imbalance), framed around the GEMM shapes it produces.
> **What this is not:** a new algorithm, a benchmarked speedup, or a claim that
> any of this is unknown to the field. The contribution here is the measurement
> and its careful interpretation.

---

## TL;DR

In an MoE layer, each expert's feed-forward GEMM has an **M-dimension equal to
the number of tokens routed to that expert**. So the distribution of per-expert
token counts *is* the distribution of GEMM shapes the hardware sees. I measured
it directly on `allenai/OLMoE-1B-7B-0924` (64 experts, top-8 routing, 16 MoE
layers, on a single T4).

- **Prefill is genuinely dispersed.** Per-expert token counts run from 0 to 131
  (mean ≈ 21), with a layer-averaged coefficient of variation (CV) of **0.74**
  and a median **55×** gap between the largest and smallest *active* expert GEMM.
  No single shape dominates. This is stable across all 16 layers (CV 0.57–1.07).
- **Decode, as measured here, is an artifact — and I disregard it.** My decode
  probe fed only 16 tokens across 64 experts, so most experts got zero. The
  headline CV looks huge (2.58) but it is driven by sparsity, not by genuine
  imbalance. Honest decode measurement needs a realistic concurrent batch; see
  *Limitations*.
- **The dispersed regime is also the regime that matters.** Prefill GEMMs are
  compute-bound; decode GEMMs are tiny and memory-bandwidth-bound. A
  lower-arithmetic-cost GEMM pays off in prefill and barely matters in decode —
  so the regime that came back dispersed is exactly the one where shape-aware
  GEMM handling would help.

---

## Why this matters

MoE models cut per-token cost by routing each token to a small subset of experts
(here, 8 of 64). The expensive operation inside an active expert is a GEMM whose
M-dimension is the routed token count. Systems that deploy lower-complexity or
shape-specialized GEMM kernels for LLMs — e.g. FalconGEMM, MxMoE — choose an
algorithm based on the operator's shape, because a cheaper algorithm only wins on
some shapes and hardware.

That raises a prior question those systems mostly assume rather than measure:
**is the per-expert GEMM shape distribution actually dispersed enough that
shape-aware selection has room to help, or is it tight enough that one kernel is
fine?** Routing-induced load imbalance is well documented (it is why
auxiliary load-balancing losses and expert-capacity caps exist), but I wanted to
see and quantify the shape distribution itself for a concrete model.

---

## Method

- Load a trained MoE with `output_router_logits=True`.
- For each MoE layer, take the router logits, compute the top-k expert
  assignment per token, and tally tokens per expert. Padding tokens are masked
  out so they don't pollute the counts.
- **Prefill** regime: a batch of varied prompts run as full sequences.
- **Decode** regime: one token per sequence (the autoregressive-step shape).
- Report per-layer and aggregate dispersion stats (CV, max/min ratio over active
  experts, min/median/max counts) and plot the histogram of `N_e`.

Run on a single T4 GPU. Model: `allenai/OLMoE-1B-7B-0924` (64 experts, top-8).

---

## Results

### Prefill (the trustworthy result)

| metric | value |
|---|---|
| MoE layers measured | 16 |
| mean coefficient of variation | **0.737** |
| CV range across layers | 0.573 – 1.069 |
| median max/min active-GEMM ratio | **55×** |
| `N_e` (min / median / max) | 0 / 18 / 131 |

The histogram (`moe_dispersion.png`, left panel) shows a broad distribution: a
dense band from roughly 5–40 tokens, a mean near 21, and a long tail past 130,
with a number of experts near zero. No dominant shape.

### Decode (an artifact — reported only to be honest about it)

| metric | value |
|---|---|
| `N_e` (min / median / max) | 0 / **0** / 16 |
| mean coefficient of variation | 2.576 *(not meaningful — see below)* |

The decode probe used 16 tokens (one per prompt) across 64 experts, so the
overwhelming majority of experts received zero tokens (median `N_e` = 0). The
large CV is a **small-sample / sparsity artifact**, not evidence of load
imbalance. Real autoregressive decoding runs many sequences concurrently; with
only 16 tokens you cannot characterize 64 experts. I therefore treat the decode
number as noise rather than a finding. (The histogram's right panel makes this
obvious: a spike at zero with a stray bar at 16.)

### Interpretation

Prefill GEMMs process large token batches and are compute-bound; decode GEMMs
process a sliver of tokens per step and are memory-bandwidth-bound. A GEMM
algorithm that trades multiplications for cheaper structure only helps when the
operation is compute-bound — i.e. in prefill. So the regime that came back
clearly dispersed (prefill) is also the regime where shape-aware GEMM handling
could matter, and the regime I failed to measure well (decode) is the one where
it would matter least. The gap in the data is also the less important gap.

---

## Limitations

I'd rather state these plainly than have a reader find them.

- **Small prompt set.** 16 short prompts. The prefill signal is clear and stable
  across layers, but a publishable claim wants a larger, more representative
  workload. This is directional evidence, not a benchmark.
- **One model.** OLMoE-1B-7B only. Different expert counts, top-k, and training
  recipes will give different distributions.
- **Decode undersampled.** As above — the decode regime needs a realistic
  concurrent batch (e.g. 128+ sequences) to mean anything.
- **Raw routing, not deployed shapes.** This measures the underlying routing
  distribution. A production serving stack may pad experts to a fixed capacity,
  regularizing the shapes the backend actually sees. A dispersed raw
  distribution shifts the argument toward "padding wastes work," but that is a
  separate measurement.
- **FLOPs are not latency.** Dispersed shapes say nothing on their own about
  whether a cheaper GEMM is faster in wall-clock; that requires kernel-level
  benchmarking on real hardware.
- **Analysis, not a system.** No speedup is implemented or claimed here.

---

## Reproduce

```bash
pip install torch transformers matplotlib numpy accelerate
python moe_dispersion_probe.py
```

On a 16 GB consumer GPU/Mac the default model is tight; a free Colab T4 fits it
comfortably. The script prints the PREFILL and DECODE stats and a verdict line,
and writes `moe_dispersion.png`. Options:

```bash
python moe_dispersion_probe.py --model allenai/OLMoE-1B-7B-0924
python moe_dispersion_probe.py --prompts my_prompts.txt   # one prompt per line
```

---

## Next steps (if continued)

- Re-run prefill on a larger, representative prompt set and across several MoE
  models to test how general the dispersion is.
- Measure decode honestly with a realistic concurrent batch.
- If the dispersion holds, quantify how often the *shape-optimal* GEMM algorithm
  changes across the measured distribution — the actual condition under which
  shape-aware selection (vs. a single kernel) would pay off — and benchmark
  wall-clock, not just FLOPs.
