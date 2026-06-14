"""Feature-cache (TeaCache / First-Block-Cache) step-skipping accelerators for
single-GPU diffusers Wan inference."""

from nanoxdit.cache.base import CachedWanBlocks
from nanoxdit.cache.teacache import TeaCacheWanBlocks
from nanoxdit.cache.fbcache import FBCacheWanBlocks
from nanoxdit.cache.wan_adapter import (
    apply_cache_on_transformer,
    remove_cache_from_transformer,
    WAN_TEACACHE_COEFFICIENTS,
)

__all__ = [
    "CachedWanBlocks",
    "TeaCacheWanBlocks",
    "FBCacheWanBlocks",
    "apply_cache_on_transformer",
    "remove_cache_from_transformer",
    "WAN_TEACACHE_COEFFICIENTS",
]
