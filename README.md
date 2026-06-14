# Nano-xDiT

A lightweight single-GPU [Wan](https://github.com/Wan-Video/Wan2.1) video-DiT inference engine with TeaCache / First-Block-Cache step-skipping — a minimal [xDiT](https://github.com/xdit-project/xDiT), built from scratch.

## Key Features

* 🎬 **Single-GPU Wan inference** — diffusers `WanTransformer3DModel` text-to-video, no distributed / sequence-parallel machinery
* 🚀 **Feature caching** — TeaCache and First-Block-Cache skip whole denoising steps for ~1.6–2.9× speedup
* 📖 **Readable codebase** — the whole engine is ~700 lines of Python
* 🔬 **Pluggable for research** — an explicit, instrumentable denoising loop plus a cache framework where a new strategy is one `CachePolicy` subclass; switch granularity (whole-stack ↔ per-block) with one argument
* ✅ **Faithful** — official ali-vilab/TeaCache Wan2.1 coefficients verbatim; with skipping disabled the output is bit-for-bit identical to no-cache

## Installation

```bash
pip install git+https://github.com/Antlera/nano-xDiT.git
```

## Quick Start

See `example.py`. The pipeline mirrors diffusers' `WanPipeline`, plus an `enable_cache` call:

```python
import torch
from nanoxdit import NanoWanPipeline

pipe = NanoWanPipeline.from_pretrained(
    "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", torch_dtype=torch.bfloat16, device="cuda"
)

# TeaCache step-skipping (rel_l1_thresh: larger => faster, lower fidelity)
pipe.enable_cache(
    policy="teacache",             # or "fbcache", or your own registered policy
    num_inference_steps=30,
    rel_l1_thresh=0.2,
    wan_variant="t2v-1.3B",        # picks official TeaCache coefficients
)

video = pipe(
    prompt="A cat and a dog baking a cake together in a kitchen.",
    negative_prompt="blurry, low quality",
    height=480, width=832, num_frames=33,
    num_inference_steps=30, guidance_scale=5.0,
)[0]

print(pipe.cache.stats)   # per-CFG-branch compute/skip counts
```

To run the engine without caching (baseline), just skip `enable_cache`.

## Benchmark

How much TeaCache speeds up a single-GPU Wan run (cache vs. no cache). See `bench.py` for the methodology.

**Test Configuration:**
- Hardware: 1× NVIDIA RTX PRO 6000 (Blackwell)
- Model: Wan2.1-T2V-1.3B (bf16)
- Resolution: 480×832, 33 frames, 30 steps, CFG 5.0, UniPC (flow_shift 3.0)
- Prompt: fixed; same seed across runs

**Performance Results:**
| Config | Time (s) | Speedup | Steps skipped | Peak VRAM (GB) |
|---|---|---|---|---|
| baseline (no cache) | 28.6 | 1.00× | 0% | 15.5 |
| TeaCache thr=0.1 | 18.3 | 1.57× | 37% | 15.6 |
| TeaCache thr=0.2 | 12.7 | 2.26× | 57% | 15.6 |
| TeaCache thr=0.3 | 9.8 | 2.90× | 67% | 15.6 |

Skipping a step still pays the non-block work (patch-embed, RoPE, condition embedder, output norm/unpatchify) that diffusers recomputes every step — about 17% of a full forward — which caps the achievable speedup, matching the official TeaCache behavior.

## How it works

`apply_cache_on_transformer` swaps the transformer's block stack for a small wrapper (via `mock.patch.object`, leaving the diffusers forward otherwise untouched). Each denoising step the wrapper decides whether the block stack has changed enough to recompute, or whether to reuse the previous step's cached residual:

- **TeaCache** measures the relative-L1 change of the timestep embedding (`e0`), rescales it with the official per-model polynomial, accumulates it, and skips while under `rel_l1_thresh`.
- **First-Block-Cache** runs only the first block and uses its output residual as the change signal for the rest of the stack.

Classifier-free guidance runs as two separate forwards (conditional / unconditional); each keeps its own cache branch, mirroring the official TeaCache even/odd buffers.

## Writing a cache policy

The mechanism (running blocks, storing/reusing residuals, per-branch state, step counting, instrumentation) is fixed; a **policy** only decides *when to recompute*. Subclass `CachePolicy`, register it, and you're done:

```python
from nanoxdit.cache import CachePolicy, register_policy, relative_l1

@register_policy("my_policy")
class MyPolicy(CachePolicy):
    needs_first_block_probe = False          # True => run unit's first block first (FBCache-style)

    def __init__(self, *, rel_l1_thresh=0.1):
        self.rel_l1_thresh = rel_l1_thresh

    def reset(self, state):                   # per (unit, CFG-branch) scratch init
        state.user["prev"] = None

    def should_compute(self, ctx) -> bool:    # the only required method
        prev = ctx.state.user["prev"]
        ctx.state.user["prev"] = ctx.e0.clone()
        return prev is None or relative_l1(ctx.e0, prev) >= self.rel_l1_thresh

    # def reconstruct(self, ctx):             # optional: override for extrapolation
    #     return ctx.hidden + ctx.state.last_residual   # default = zeroth-order reuse
```

`ctx` exposes `step / num_steps / branch / unit_index / num_units / hidden / e0 / e / encoder_hidden / state` (and `first_block_residual` when probing). Then:

```python
pipe.enable_cache(policy="my_policy", num_inference_steps=30,
                  granularity="per_block",   # "stack" (default), "per_block", or an int k
                  rel_l1_thresh=0.1)
```

`granularity` controls the cache unit: `"stack"` (one decision for all blocks, as TeaCache/FBCache do), `"per_block"` (each block decides independently, with its own residual + state), or `k` (groups of `k`). TeaCache and First-Block-Cache (`nanoxdit/cache/policies/`) are ~30-line policies under this same interface; `perblock_demo` is a per-block template. With an always-compute policy the output is bit-for-bit identical to no-cache (regression anchor).

## Acknowledgements

The caching design follows [xDiT](https://github.com/xdit-project/xDiT) (Apache-2.0); the TeaCache algorithm and official Wan2.1 coefficients are from [TeaCache](https://github.com/ali-vilab/TeaCache), and First-Block-Cache from [ParaAttention](https://github.com/chengzeyi/ParaAttention). See `NOTICE`.
