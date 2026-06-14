"""nano-xDiT: a minimal single-GPU Wan video DiT inference + feature-cache toolkit.

Extracted from xDiT (https://github.com/xdit-project/xDiT). All distributed /
parallel machinery has been stripped out; what remains is the TeaCache /
First-Block-Cache step-skipping accelerators and an explicit Wan denoising loop
for research instrumentation.
"""

from nanoxdit.cache.wan_adapter import apply_cache_on_transformer
from nanoxdit.pipeline import NanoWanPipeline

__all__ = ["apply_cache_on_transformer", "NanoWanPipeline"]
