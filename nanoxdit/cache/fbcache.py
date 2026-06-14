# First-Block-Cache (FBCache) for diffusers Wan transformers.
#
# Adapted from the ParaAttention / xDiT FBCachedTransformerBlocks idea, rewritten
# for Wan's single-stream block stack. Unlike TeaCache (which uses the timestep
# projection e0 as its change signal), FBCache always runs block[0] and uses the
# first block's output residual (block0_out - input) as the signal: if that
# residual is close enough to the previous step's, the remaining blocks are
# skipped and the cached full residual reused.
#
# There is no polynomial rescale and no accumulator; the raw relative-L1 of the
# first-block residual is compared directly against `rel_l1_thresh`. block[0] is
# recomputed every step (that is the cost of the signal), so FBCache skips at
# most num_layers-1 of the layers on a cache hit.

from nanoxdit.cache.base import CachedWanBlocks


class FBCacheWanBlocks(CachedWanBlocks):
    def predict(self, st, hidden_states, encoder_hidden_states, temb, rotary_emb):
        # Always run the first block to obtain the change signal.
        first_block = self.blocks[0]
        hidden_after_first = first_block(hidden_states, encoder_hidden_states, temb, rotary_emb)
        first_residual = hidden_after_first - hidden_states

        if st.prev_modulated is None:
            should_calc = True
        else:
            rel = self.relative_l1(first_residual, st.prev_modulated)
            # Recompute when the first-block residual has changed enough.
            should_calc = rel >= self.rel_l1_thresh

        # Canonical FBCache (ParaAttention / xDiT) refreshes the reference signal
        # ONLY on recompute steps, so the comparison is always "now vs the last
        # computed step". (TeaCache differs: it refreshes e0 every forward.)
        if should_calc:
            st.prev_modulated = first_residual.clone()

        # On recompute, continue the stack from block[0]'s output (start_idx 1).
        # On skip, the base class reuses the cached full residual and ignores
        # hidden_after_first.
        return should_calc, hidden_after_first, 1
