"""Pluggable feature-cache (TeaCache / First-Block-Cache / custom) step-skipping
for single-GPU diffusers Wan inference.

Write a new strategy by subclassing CachePolicy and decorating it with
@register_policy("name"); then `apply_cache_on_transformer(transformer,
policy="name", num_inference_steps=N, granularity="stack"|"per_block", ...)`.
"""

from nanoxdit.cache.framework import (
    apply_cache_on_transformer,
    remove_cache_from_transformer,
    CachePolicy,
    register_policy,
    POLICY_REGISTRY,
    CachedUnit,
    CacheController,
    StepContext,
    UnitState,
    HistoryRecord,
    relative_l1,
)

# Importing the policies package registers the built-in policies.
from nanoxdit.cache import policies  # noqa: F401
from nanoxdit.cache.policies import WAN_TEACACHE_COEFFICIENTS

__all__ = [
    "apply_cache_on_transformer",
    "remove_cache_from_transformer",
    "CachePolicy",
    "register_policy",
    "POLICY_REGISTRY",
    "CachedUnit",
    "CacheController",
    "StepContext",
    "UnitState",
    "HistoryRecord",
    "relative_l1",
    "WAN_TEACACHE_COEFFICIENTS",
]
