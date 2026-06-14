# Base feature-cache wrapper for a diffusers Wan transformer block stack.
#
# Adapted (and heavily slimmed) from xDiT's
# xfuser/model_executor/cache/utils.py. All sequence-parallel machinery
# (get_sp_group / all_reduce / runtime_state) has been removed: nano-xDiT is
# single-GPU only, so is_parallelized is always False and the relative-L1
# distance is computed locally.
#
# Design
# ------
# diffusers' WanTransformer3DModel.forward runs, after patch-embed + condition
# embedder:
#
#     for block in self.blocks:
#         hidden_states = block(hidden_states, encoder_hidden_states,
#                               timestep_proj, rotary_emb)
#
# We replace `self.blocks` (temporarily, via mock.patch.object in the adapter)
# with a one-element ModuleList holding a CachedWanBlocks instance. Because the
# wrapper's forward signature matches a single Wan block, it receives
# `timestep_proj` (the official TeaCache "e0" modulated input) for free, decides
# whether to recompute the whole stack or reuse the cached residual, loops over
# the real blocks internally, and returns the final hidden_states tensor. Wan
# blocks return only hidden_states (encoder_hidden_states is read-only across
# the stack), so the wrapper does too.
#
# Classifier-free guidance in diffusers' WanPipeline runs the conditional and
# unconditional passes as two SEPARATE transformer forwards. This maps directly
# onto the official TeaCache even/odd dual cache: we keep one independent cache
# branch per pass, keyed by a string the pipeline sets via set_branch().

from abc import ABC, abstractmethod

import torch
from torch import nn


class _BranchState:
    """Per-CFG-branch cache state (one for the conditional pass, one for the
    unconditional pass). Mirrors the official TeaCache even/odd buffers."""

    __slots__ = ("cnt", "accumulated", "prev_modulated", "prev_residual", "calc", "skip")

    def __init__(self):
        self.cnt = 0                       # denoise-step counter within this branch
        self.accumulated = 0.0             # accumulated rescaled rel-L1 distance
        self.prev_modulated = None         # previous modulated input (e0 or first-block residual)
        self.prev_residual = None          # cached full residual: final - input
        self.calc = 0                      # #steps actually recomputed (instrumentation)
        self.skip = 0                      # #steps skipped via cache    (instrumentation)


class CachedWanBlocks(nn.Module, ABC):
    """Abstract feature-cache wrapper around a Wan transformer block stack.

    Subclasses implement `predict()`, which inspects the current step and
    decides whether the full stack must be recomputed. The base class owns the
    per-branch bookkeeping and the recompute / skip residual mechanics, which
    are identical for TeaCache and First-Block-Cache.
    """

    def __init__(self, blocks: nn.ModuleList, *, num_steps: int, rel_l1_thresh: float, name: str = "wan"):
        super().__init__()
        # Reference to the real Wan blocks. They remain registered under
        # transformer.blocks; we only alias them here, so the model's state_dict
        # is unaffected (the wrapper itself is NOT part of the transformer tree).
        self.blocks = blocks
        self.num_steps = int(num_steps)
        self.rel_l1_thresh = float(rel_l1_thresh)
        self.name = name

        self.branch = "cond"
        self._states: dict[str, _BranchState] = {}

        # Official TeaCache base time embedding `e` (diffusers `temb`), captured
        # each forward by the adapter. Only used by TeaCache when
        # use_ret_steps=False (the official code switches its change signal from
        # e0 to e there); None otherwise.
        self._e_base = None

    # ------------------------------------------------------------------ #
    # Branch / lifecycle control (driven by NanoWanPipeline)
    # ------------------------------------------------------------------ #
    def set_branch(self, name: str) -> None:
        """Select which CFG branch ("cond" / "uncond") the next forward writes to."""
        self.branch = name

    def reset(self) -> None:
        """Clear all cache state. Call once before each independent generation."""
        self._states.clear()
        self.branch = "cond"

    def configure_steps(self, num_steps: int) -> None:
        """Sync the per-branch step count the counter wraps at. TeaCache
        overrides this to also recompute its forced-compute window."""
        self.num_steps = int(num_steps)

    def _state(self) -> _BranchState:
        st = self._states.get(self.branch)
        if st is None:
            st = _BranchState()
            self._states[self.branch] = st
        return st

    @property
    def stats(self) -> dict:
        """Per-branch (calc, skip) counters for research instrumentation."""
        return {b: {"calc": s.calc, "skip": s.skip} for b, s in self._states.items()}

    # ------------------------------------------------------------------ #
    # Subclass hook
    # ------------------------------------------------------------------ #
    @abstractmethod
    def predict(self, st, hidden_states, encoder_hidden_states, temb, rotary_emb):
        """Decide whether to recompute the stack for the current step.

        Returns (should_calc, hidden_start, start_idx):
          should_calc  -- True => run blocks[start_idx:]; False => reuse residual
          hidden_start -- tensor to start the remaining-block loop from
                          (TeaCache: the original input; FBCache: block[0] output)
          start_idx    -- index into self.blocks for the remaining loop
                          (TeaCache: 0; FBCache: 1, since block[0] already ran)
        """
        raise NotImplementedError

    @staticmethod
    def relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
        """Mean-absolute relative L1 distance, computed in fp32 to match the
        official TeaCache numerics regardless of the transformer's dtype."""
        current = current.float()
        previous = previous.float()
        return ((current - previous).abs().mean() / previous.abs().mean()).item()

    # ------------------------------------------------------------------ #
    # Forward: matches a single Wan block's call signature
    # ------------------------------------------------------------------ #
    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        st = self._state()
        original_hidden = hidden_states

        should_calc, hidden_start, start_idx = self.predict(
            st, hidden_states, encoder_hidden_states, temb, rotary_emb
        )

        if not should_calc and st.prev_residual is not None:
            hidden_states = original_hidden + st.prev_residual
            st.skip += 1
        else:
            hidden_states = hidden_start
            for block in self.blocks[start_idx:]:
                hidden_states = block(hidden_states, encoder_hidden_states, temb, rotary_emb)
            # Full residual relative to the ORIGINAL input, so the skip path
            # (original_input + residual) reconstructs the stack output exactly.
            st.prev_residual = hidden_states - original_hidden
            st.calc += 1

        st.cnt = (st.cnt + 1) % self.num_steps
        return hidden_states
