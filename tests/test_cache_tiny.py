"""End-to-end sanity test for the nano-xDiT feature cache on a tiny randomly
initialised Wan transformer. No real weights / VAE / text encoder are needed.

Run directly:  python tests/test_cache_tiny.py
Or with pytest: pytest tests/test_cache_tiny.py

Checks:
  * the cache-patched transformer produces finite output of the right shape
  * a very negative threshold forces full computation and reproduces the
    no-cache baseline bit-for-bit (the skip path is never taken)
  * a large threshold makes TeaCache skip every step outside the forced window
    (use_ret_steps=True => first 5 of N steps computed, rest skipped)
  * FBCache also skips and stays finite
"""

import torch

from diffusers import WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from nanoxdit import NanoWanPipeline, apply_cache_on_transformer
from nanoxdit.cache import remove_cache_from_transformer


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float32  # fp32 for a deterministic tiny test


def build_tiny_transformer(seed=0):
    torch.manual_seed(seed)
    model = WanTransformer3DModel(
        patch_size=(1, 2, 2),
        num_attention_heads=2,
        attention_head_dim=16,   # inner_dim = 32
        in_channels=4,
        out_channels=4,
        text_dim=16,
        freq_dim=64,
        ffn_dim=64,
        num_layers=6,
        cross_attn_norm=True,
        qk_norm="rms_norm_across_heads",
        eps=1e-6,
        rope_max_seq_len=64,
    )
    return model.to(DEVICE, DTYPE).eval()


def make_inputs(batch=1, seed=1):
    g = torch.Generator(device=DEVICE).manual_seed(seed)
    latents = torch.randn(batch, 4, 4, 16, 16, generator=g, device=DEVICE, dtype=torch.float32)
    prompt = torch.randn(batch, 12, 16, generator=g, device=DEVICE, dtype=DTYPE)
    negative = torch.randn(batch, 12, 16, generator=g, device=DEVICE, dtype=DTYPE)
    return latents, prompt, negative


def run(pipe, latents, prompt, negative, steps):
    return pipe(
        prompt_embeds=prompt,
        negative_prompt_embeds=negative,
        latents=latents.clone(),
        num_inference_steps=steps,
        guidance_scale=5.0,
        output_type="latent",
    )


def main():
    steps = 10
    transformer = build_tiny_transformer()
    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=3.0)
    pipe = NanoWanPipeline(transformer=transformer, scheduler=scheduler)
    latents, prompt, negative = make_inputs()

    # ---- baseline: no cache ----
    base_out = run(pipe, latents, prompt, negative, steps)
    assert torch.isfinite(base_out).all(), "baseline produced non-finite output"
    print(f"[baseline ] shape={tuple(base_out.shape)} mean={base_out.mean():.5f}")

    # ---- TeaCache, threshold so negative it must compute every step ----
    apply_cache_on_transformer(
        transformer,
        algorithm="teacache",
        num_inference_steps=steps,
        rel_l1_thresh=-1e9,
        coefficients=[1.0, 0.0],  # identity-ish; irrelevant since we never skip
        use_ret_steps=True,
    )
    eq_out = run(pipe, latents, prompt, negative, steps)
    stats = transformer._nano_cache.stats
    skips = sum(b["skip"] for b in stats.values())
    print(f"[tea -inf ] skips={skips} stats={stats} max|Δ|={ (eq_out-base_out).abs().max():.2e}")
    assert skips == 0, "negative threshold should never skip"
    assert torch.allclose(eq_out, base_out, atol=1e-5, rtol=1e-4), "forced-compute cache != baseline"
    remove_cache_from_transformer(transformer)

    # ---- TeaCache, large threshold => skip everything outside forced window ----
    apply_cache_on_transformer(
        transformer,
        algorithm="teacache",
        num_inference_steps=steps,
        rel_l1_thresh=1e9,
        coefficients=[1.0, 0.0],
        use_ret_steps=True,
    )
    tea_out = run(pipe, latents, prompt, negative, steps)
    stats = transformer._nano_cache.stats
    print(f"[tea +inf ] stats={stats}")
    assert torch.isfinite(tea_out).all(), "TeaCache produced non-finite output"
    # Per branch: ret_steps=5 forced-compute, remaining 5 skipped.
    for branch, s in stats.items():
        assert s["calc"] == 5, f"{branch}: expected 5 forced computes, got {s['calc']}"
        assert s["skip"] == 5, f"{branch}: expected 5 skips, got {s['skip']}"
    remove_cache_from_transformer(transformer)

    # ---- FBCache, large threshold => skip remaining blocks each step ----
    apply_cache_on_transformer(
        transformer,
        algorithm="fbcache",
        num_inference_steps=steps,
        rel_l1_thresh=1e9,
    )
    fb_out = run(pipe, latents, prompt, negative, steps)
    stats = transformer._nano_cache.stats
    skips = sum(b["skip"] for b in stats.values())
    print(f"[fb  +inf ] skips={skips} stats={stats}")
    assert torch.isfinite(fb_out).all(), "FBCache produced non-finite output"
    assert skips > 0, "FBCache with large threshold should skip"
    remove_cache_from_transformer(transformer)

    print("\nAll tiny-model cache checks passed.")


if __name__ == "__main__":
    main()
