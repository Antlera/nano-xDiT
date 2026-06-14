"""Built-in cache policies. Importing this package registers them in
POLICY_REGISTRY. Add a new policy by dropping a module here that defines a
CachePolicy subclass decorated with @register_policy("name")."""

from nanoxdit.cache.policies import teacache, fbcache, perblock  # noqa: F401  (register on import)

from nanoxdit.cache.policies.teacache import TeaCachePolicy, WAN_TEACACHE_COEFFICIENTS
from nanoxdit.cache.policies.fbcache import FBCachePolicy
from nanoxdit.cache.policies.perblock import PerBlockDemoPolicy

__all__ = ["TeaCachePolicy", "FBCachePolicy", "PerBlockDemoPolicy", "WAN_TEACACHE_COEFFICIENTS"]
