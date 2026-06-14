# TeaCache for diffusers Wan transformers.
#
# Port of the official ali-vilab/TeaCache Wan implementation
# (teacache_forward in TeaCache4Wan2.1). The modulated input is the timestep
# projection e0 (== diffusers `timestep_proj`, shape [B, 6, inner_dim]). A
# polynomial rescales the per-step relative-L1 change of e0; the rescaled value
# is accumulated, and while the accumulator stays below `rel_l1_thresh` the
# whole block stack is skipped and the previous residual reused.
#
# Step-window equivalence with the official code
# ----------------------------------------------
# The official implementation keeps a single counter `cnt` that increments once
# per transformer forward, i.e. twice per denoise step (conditional + then
# unconditional), and selects the even/odd buffer via `cnt % 2`. Its forced-
# compute window is therefore expressed in that doubled space:
#     use_ret_steps=True :  ret_steps = 5*2,  cutoff_steps = sample_steps*2
#     use_ret_steps=False:  ret_steps = 1*2,  cutoff_steps = sample_steps*2 - 2
# nano-xDiT instead keeps one independent counter per CFG branch (each counts
# 0..sample_steps-1), so the windows are halved to the per-branch values
#     use_ret_steps=True :  ret_steps = 5,  cutoff_steps = sample_steps
#     use_ret_steps=False:  ret_steps = 1,  cutoff_steps = sample_steps - 1
# which selects exactly the same steps for forced computation.

import numpy as np

from nanoxdit.cache.base import CachedWanBlocks


class TeaCacheWanBlocks(CachedWanBlocks):
    def __init__(
        self,
        blocks,
        *,
        num_steps: int,
        rel_l1_thresh: float,
        coefficients,
        use_ret_steps: bool = True,
        name: str = "wan",
    ):
        super().__init__(blocks, num_steps=num_steps, rel_l1_thresh=rel_l1_thresh, name=name)
        self.coefficients = list(coefficients)
        self.rescale = np.poly1d(self.coefficients)
        self.use_ret_steps = bool(use_ret_steps)
        self.configure_steps(num_steps)

    def configure_steps(self, num_steps: int) -> None:
        """(Re)compute the per-branch forced-compute window for `num_steps`."""
        self.num_steps = int(num_steps)
        if self.use_ret_steps:
            self.ret_steps = 5
            self.cutoff_steps = self.num_steps        # cutoff never fires
        else:
            self.ret_steps = 1
            self.cutoff_steps = self.num_steps - 1

    def predict(self, st, hidden_states, encoder_hidden_states, temb, rotary_emb):
        # Official TeaCache change signal (teacache_wan.py: `modulated_inp = e0
        # if self.use_ref_steps else e`). `temb` here is diffusers `timestep_proj`
        # == e0; the base time embedding `e` (diffusers `temb` from the condition
        # embedder) is captured by the adapter as self._e_base.
        if self.use_ret_steps:
            modulated = temb
        else:
            if self._e_base is None:
                raise RuntimeError(
                    "use_ret_steps=False needs the base time embedding `e`, which "
                    "is captured by apply_cache_on_transformer. Install the cache "
                    "through that function (not by constructing TeaCacheWanBlocks "
                    "directly)."
                )
            modulated = self._e_base

        forced = st.cnt < self.ret_steps or st.cnt >= self.cutoff_steps
        if forced or st.prev_modulated is None:
            should_calc = True
            st.accumulated = 0.0
        else:
            rel = self.relative_l1(modulated, st.prev_modulated)
            st.accumulated += float(self.rescale(rel))
            if st.accumulated < self.rel_l1_thresh:
                should_calc = False
            else:
                should_calc = True
                st.accumulated = 0.0

        # The official code refreshes previous_e0 on every forward, including
        # skipped steps, so the next comparison is always against the latest e0.
        st.prev_modulated = modulated.clone()

        # TeaCache decides before any block runs: recompute the full stack from
        # the original input (start_idx 0).
        return should_calc, hidden_states, 0
