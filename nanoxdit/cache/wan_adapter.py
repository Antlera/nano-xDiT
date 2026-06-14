# Adapter that installs a TeaCache / First-Block-Cache wrapper onto a diffusers
# WanTransformer3DModel for single-GPU inference.
#
# Mechanism (mirrors xDiT's flux adapter, adapted to Wan):
#   - Build a one-element ModuleList holding the cache wrapper, which aliases the
#     real transformer.blocks.
#   - Monkeypatch transformer.forward so that, for the duration of each forward,
#     `self.blocks` is swapped (via mock.patch.object) for that one-element list.
#     Wan's forward loop `for block in self.blocks: hidden = block(hidden, enc,
#     timestep_proj, rotary_emb)` then calls the wrapper exactly once, handing it
#     timestep_proj (the TeaCache e0 signal) for free.
#   - The original forward (patch embed, condition embedder, norm_out, unpatchify)
#     is reused unchanged, so the adapter is robust to that code.
#
# The wrapper handle is stashed on the transformer as `_nano_cache` (without
# registering it as a submodule, so state_dict / .to() are unaffected).
# NanoWanPipeline drives it via set_branch()/reset().

import functools
from unittest import mock

from torch import nn

from nanoxdit.cache.teacache import TeaCacheWanBlocks
from nanoxdit.cache.fbcache import FBCacheWanBlocks


# Official ali-vilab/TeaCache polynomial coefficients for Wan 2.1, keyed by
# (task, variant, use_ret_steps). Copied verbatim from teacache_wan.py.
WAN_TEACACHE_COEFFICIENTS = {
    # text-to-video
    ("t2v", "1.3B", True): [-5.21862437e04, 9.23041404e03, -5.28275948e02, 1.36987616e01, -4.99875664e-02],
    ("t2v", "14B", True): [-3.03318725e05, 4.90537029e04, -2.65530556e03, 5.87365115e01, -3.15583525e-01],
    ("t2v", "1.3B", False): [2.39676752e03, -1.31110545e03, 2.01331979e02, -8.29855975e00, 1.37887774e-01],
    ("t2v", "14B", False): [-5784.54975374, 5449.50911966, -1811.16591783, 256.27178429, -13.02252404],
    # image-to-video (14B); variant is the target resolution
    ("i2v", "480p", True): [2.57151496e05, -3.54229917e04, 1.40286849e03, -1.35890334e01, 1.32517977e-01],
    ("i2v", "720p", True): [8.10705460e03, 2.13393892e03, -3.72934672e02, 1.66203073e01, -4.17769401e-02],
    ("i2v", "480p", False): [-3.02331670e02, 2.23948934e02, -5.25463970e01, 5.87348440e00, -2.01973289e-01],
    ("i2v", "720p", False): [-114.36346466, 65.26524496, -18.82220707, 4.91518089, -0.23412683],
}


def _parse_variant(wan_variant: str):
    """'t2v-14B' / 't2v-1.3B' / 'i2v-480p' / 'i2v-720p' -> (task, variant)."""
    task, _, variant = wan_variant.lower().partition("-")
    if task not in ("t2v", "i2v") or not variant:
        raise ValueError(
            f"Unrecognized wan_variant {wan_variant!r}; expected one of "
            "'t2v-14B', 't2v-1.3B', 'i2v-480p', 'i2v-720p'."
        )
    # Normalize casing used in the coefficient table.
    variant = {"14b": "14B", "1.3b": "1.3B", "480p": "480p", "720p": "720p"}.get(variant, variant)
    return task, variant


def apply_cache_on_transformer(
    transformer,
    *,
    algorithm: str = "teacache",
    num_inference_steps: int,
    rel_l1_thresh: float = 0.2,
    wan_variant: str | None = None,
    coefficients=None,
    use_ret_steps: bool = True,
):
    """Install a feature cache on `transformer` (a diffusers WanTransformer3DModel).

    Args:
        algorithm: "teacache" or "fbcache".
        num_inference_steps: number of denoise steps per CFG branch; the cache
            counter wraps at this value.
        rel_l1_thresh: skip threshold. Larger => more aggressive skipping.
        wan_variant: selects official TeaCache coefficients, e.g. "t2v-14B".
        coefficients: explicit TeaCache polynomial coefficients (overrides
            wan_variant).
        use_ret_steps: TeaCache forced-compute window selection (matches the
            official --use_ret_steps flag).

    Returns the same transformer with `_nano_cache` attached.
    """
    if not hasattr(transformer, "blocks") or not hasattr(transformer, "condition_embedder"):
        raise TypeError(
            "apply_cache_on_transformer expects a diffusers WanTransformer3DModel "
            "(with .blocks and .condition_embedder)."
        )

    algorithm = algorithm.lower()
    if algorithm in ("teacache", "tea"):
        if coefficients is None:
            if wan_variant is None:
                raise ValueError("TeaCache needs either `coefficients` or `wan_variant`.")
            task, variant = _parse_variant(wan_variant)
            key = (task, variant, use_ret_steps)
            if key not in WAN_TEACACHE_COEFFICIENTS:
                raise ValueError(f"No official coefficients for {key}.")
            coefficients = WAN_TEACACHE_COEFFICIENTS[key]
        cache = TeaCacheWanBlocks(
            transformer.blocks,
            num_steps=num_inference_steps,
            rel_l1_thresh=rel_l1_thresh,
            coefficients=coefficients,
            use_ret_steps=use_ret_steps,
            name=wan_variant or "wan",
        )
    elif algorithm in ("fbcache", "fb", "first_block_cache"):
        cache = FBCacheWanBlocks(
            transformer.blocks,
            num_steps=num_inference_steps,
            rel_l1_thresh=rel_l1_thresh,
            name=wan_variant or "wan",
        )
    else:
        raise ValueError(f"Unknown algorithm {algorithm!r}; expected 'teacache' or 'fbcache'.")

    cached_blocks = nn.ModuleList([cache])
    original_forward = transformer.forward
    original_ce_forward = transformer.condition_embedder.forward

    def capturing_condition_embedder(*a, **k):
        out = original_ce_forward(*a, **k)
        # out = (temb, timestep_proj, encoder_hidden_states, ...); out[0] is
        # diffusers `temb` == official TeaCache base time embedding `e`, needed
        # as the change signal when use_ret_steps=False.
        cache._e_base = out[0]
        return out

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        with mock.patch.object(self, "blocks", cached_blocks), mock.patch.object(
            self.condition_embedder, "forward", capturing_condition_embedder
        ):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)

    # Stash the handle without registering it as a submodule.
    object.__setattr__(transformer, "_nano_cache", cache)
    object.__setattr__(transformer, "_nano_original_forward", original_forward)
    return transformer


def remove_cache_from_transformer(transformer):
    """Undo apply_cache_on_transformer, restoring the original forward."""
    if hasattr(transformer, "_nano_original_forward"):
        transformer.forward = transformer._nano_original_forward
        object.__delattr__(transformer, "_nano_original_forward")
        object.__delattr__(transformer, "_nano_cache")
    return transformer
