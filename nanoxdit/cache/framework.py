# Pluggable feature-cache framework for single-GPU diffusers Wan transformers.
#
# Design: a single MECHANISM owns all bookkeeping (per-CFG-branch state, residual
# store/reuse, optional history, step counter, instrumentation), and a small
# POLICY decides, each step, whether a cache unit must be recomputed. Researchers
# write only a policy.
#
#   - CachedUnit   : wraps a contiguous slice of the real Wan blocks; this is the
#                    cache granularity. One unit over all blocks == TeaCache/FBCache;
#                    one unit per block == per-block caching.
#   - CacheController: holds the units + the shared policy + the current CFG branch
#                    and step count; the pipeline drives it (set_branch/reset).
#   - StepContext  : everything a policy may read for one (unit, step) decision.
#   - CachePolicy  : the extension point — implement should_compute(); optionally
#                    override reconstruct() (default = zeroth-order residual reuse)
#                    and set needs_first_block_probe (FBCache-style signals).
#
# The block stack is intercepted exactly like xDiT's flux adapter: transformer.forward
# is wrapped so that, for the duration of each forward, `self.blocks` is swapped for
# the list of CachedUnits (via unittest.mock.patch.object). Each unit's call signature
# matches a real Wan block, so it receives `timestep_proj` (== official TeaCache e0)
# for free. The original forward body (patch-embed, condition embedder, norm_out,
# unpatchify) is reused unchanged.

import functools
from collections import deque
from dataclasses import dataclass
from unittest import mock

import torch
from torch import nn


def relative_l1(current: torch.Tensor, previous: torch.Tensor) -> float:
    """Mean-absolute relative L1 distance, in fp32 (matches official TeaCache
    numerics regardless of the transformer dtype)."""
    current = current.float()
    previous = previous.float()
    return ((current - previous).abs().mean() / previous.abs().mean()).item()


@dataclass
class HistoryRecord:
    """One past COMPUTE step for a unit/branch (opt-in; for extrapolation policies)."""
    step: int
    residual: torch.Tensor
    output: torch.Tensor
    e0: torch.Tensor


class UnitState:
    """Per-(unit, CFG-branch) runtime state, owned by the mechanism.

    `user` is free scratch space for the policy (accumulators, previous signal,
    ...). `last_residual` is the cached zeroth-order residual (final - input).
    `history` is an optional bounded deque of HistoryRecord for extrapolation.
    """

    __slots__ = ("cnt", "last_residual", "history", "user", "calc", "skip")

    def __init__(self, history_len: int = 0):
        self.cnt = 0
        self.last_residual = None
        self.history = deque(maxlen=history_len) if history_len else None
        self.user = {}
        self.calc = 0
        self.skip = 0


class StepContext:
    """Read-only-ish bundle handed to the policy for one (unit, step) decision.

    `e0` is the timestep projection (official TeaCache e0); `e` is the base time
    embedding (official e), captured by the adapter. `first_block_*` are populated
    by the mechanism only when the policy sets needs_first_block_probe.
    """

    __slots__ = (
        "step", "num_steps", "branch", "unit_index", "num_units",
        "hidden", "encoder_hidden", "e0", "e", "rotary_emb", "state",
        "first_block_output", "first_block_residual",
    )

    def __init__(self, *, step, num_steps, branch, unit_index, num_units,
                 hidden, encoder_hidden, e0, e, rotary_emb, state):
        self.step = step
        self.num_steps = num_steps
        self.branch = branch
        self.unit_index = unit_index
        self.num_units = num_units
        self.hidden = hidden
        self.encoder_hidden = encoder_hidden
        self.e0 = e0
        self.e = e
        self.rotary_emb = rotary_emb
        self.state = state
        self.first_block_output = None
        self.first_block_residual = None


class CachePolicy:
    """Extension point. Subclass and implement should_compute().

    - should_compute(ctx) -> bool : True => recompute the unit; False => reuse.
      May read/write ctx.state.user for its own scratch (accumulators, prev signal).
    - reconstruct(ctx) -> Tensor  : output when skipping. Default is zeroth-order
      (input + last computed residual). Override for extrapolation (e.g. Taylor),
      using ctx.state.history (enable with history_len > 0).
    - needs_first_block_probe     : if True, the mechanism runs the unit's first
      block before should_compute and fills ctx.first_block_{output,residual}; on
      recompute the stack continues from that output (FBCache-style signals).
    """

    needs_first_block_probe = False

    def reset(self, state: UnitState) -> None:
        """Initialize per-unit scratch when a branch's state is first created or reset."""

    def should_compute(self, ctx: StepContext) -> bool:
        raise NotImplementedError

    def reconstruct(self, ctx: StepContext) -> torch.Tensor:
        return ctx.hidden + ctx.state.last_residual


# ----------------------------- registry ----------------------------- #
POLICY_REGISTRY: dict[str, type] = {}


def register_policy(name: str):
    def deco(cls):
        POLICY_REGISTRY[name] = cls
        cls.policy_name = name
        return cls
    return deco


# ----------------------------- mechanism ----------------------------- #
class CachedUnit(nn.Module):
    """Wraps a contiguous slice of the real Wan blocks as one cache unit. Its
    forward matches a single Wan block's signature so it can stand in for blocks
    inside the diffusers forward loop."""

    def __init__(self, controller, blocks, index: int):
        super().__init__()
        self.controller = controller          # plain object, not registered as submodule
        self._blocks = list(blocks)           # aliases the real blocks (not re-registered)
        self.index = index
        self._states: dict[str, UnitState] = {}

    def reset(self) -> None:
        self._states.clear()

    def _state(self) -> UnitState:
        st = self._states.get(self.controller.branch)
        if st is None:
            st = UnitState(self.controller.history_len)
            self.controller.policy.reset(st)
            self._states[self.controller.branch] = st
        return st

    def forward(self, hidden_states, encoder_hidden_states, temb, rotary_emb):
        ctrl = self.controller
        policy = ctrl.policy
        st = self._state()

        ctx = StepContext(
            step=st.cnt, num_steps=ctrl.num_steps, branch=ctrl.branch,
            unit_index=self.index, num_units=ctrl.num_units,
            hidden=hidden_states, encoder_hidden=encoder_hidden_states,
            e0=temb, e=ctrl.e_base, rotary_emb=rotary_emb, state=st,
        )

        start = 0
        if policy.needs_first_block_probe:
            fb = self._blocks[0](hidden_states, encoder_hidden_states, temb, rotary_emb)
            ctx.first_block_output = fb
            ctx.first_block_residual = fb - hidden_states
            start = 1

        compute = policy.should_compute(ctx)

        if not compute and st.last_residual is not None:
            out = policy.reconstruct(ctx)
            st.skip += 1
        else:
            h = ctx.first_block_output if start == 1 else hidden_states
            for block in self._blocks[start:]:
                h = block(h, encoder_hidden_states, temb, rotary_emb)
            out = h
            st.last_residual = out - hidden_states          # full residual vs unit input
            if st.history is not None:
                st.history.append(HistoryRecord(step=st.cnt, residual=st.last_residual, output=out, e0=temb))
            st.calc += 1

        st.cnt = (st.cnt + 1) % ctrl.num_steps
        return out


class CacheController:
    """Holds the cache units + shared policy + the current branch/step count.
    Driven by NanoWanPipeline via set_branch()/reset()/configure_steps()."""

    def __init__(self, policy: CachePolicy, num_steps: int, history_len: int = 0):
        self.policy = policy
        self.num_steps = int(num_steps)
        self.history_len = int(history_len)
        self.branch = "cond"
        self.e_base = None                    # base time embedding e, captured per forward
        self.units: list[CachedUnit] = []
        self.num_units = 0

    def set_branch(self, name: str) -> None:
        self.branch = name

    def reset(self) -> None:
        self.branch = "cond"
        self.e_base = None
        for u in self.units:
            u.reset()

    def configure_steps(self, num_steps: int) -> None:
        self.num_steps = int(num_steps)

    @property
    def stats(self) -> dict:
        """Per-branch (calc, skip), summed across units."""
        agg: dict = {}
        for u in self.units:
            for b, st in u._states.items():
                d = agg.setdefault(b, {"calc": 0, "skip": 0})
                d["calc"] += st.calc
                d["skip"] += st.skip
        return agg

    def unit_stats(self) -> list:
        """Per-unit, per-branch (calc, skip) — for per-block research."""
        return [
            {b: {"calc": st.calc, "skip": st.skip} for b, st in u._states.items()}
            for u in self.units
        ]


def _build_units(controller, blocks, granularity):
    blocks = list(blocks)
    if granularity == "stack":
        return [CachedUnit(controller, blocks, 0)]
    if granularity == "per_block":
        return [CachedUnit(controller, [b], i) for i, b in enumerate(blocks)]
    if isinstance(granularity, int) and granularity > 0:
        return [
            CachedUnit(controller, blocks[i:i + granularity], idx)
            for idx, i in enumerate(range(0, len(blocks), granularity))
        ]
    raise ValueError(f"granularity must be 'stack', 'per_block', or a positive int; got {granularity!r}")


def apply_cache_on_transformer(
    transformer,
    *,
    policy,
    num_inference_steps: int,
    granularity="stack",
    history_len: int = 0,
    **policy_kwargs,
):
    """Install a pluggable feature cache on a diffusers WanTransformer3DModel.

    Args:
        policy: a registered policy name (e.g. "teacache", "fbcache") or a
            CachePolicy instance.
        num_inference_steps: denoise steps per CFG branch (the counter wraps here).
        granularity: "stack" (one unit over all blocks), "per_block", or an int k
            (groups of k blocks).
        history_len: how many past COMPUTE steps to keep per unit (for extrapolation
            policies); 0 keeps only the last residual.
        **policy_kwargs: forwarded to the policy constructor when `policy` is a name.
    """
    if not hasattr(transformer, "blocks") or not hasattr(transformer, "condition_embedder"):
        raise TypeError("apply_cache_on_transformer expects a diffusers WanTransformer3DModel.")

    if isinstance(policy, str):
        if policy not in POLICY_REGISTRY:
            raise ValueError(f"Unknown policy {policy!r}; registered: {sorted(POLICY_REGISTRY)}")
        policy = POLICY_REGISTRY[policy](**policy_kwargs)
    elif policy_kwargs:
        raise ValueError("policy_kwargs are only used when `policy` is a registry name.")

    controller = CacheController(policy, num_steps=num_inference_steps, history_len=history_len)
    controller.units = _build_units(controller, transformer.blocks, granularity)
    controller.num_units = len(controller.units)

    cached_blocks = nn.ModuleList(controller.units)
    original_forward = transformer.forward
    original_ce_forward = transformer.condition_embedder.forward

    def capturing_condition_embedder(*a, **k):
        out = original_ce_forward(*a, **k)
        controller.e_base = out[0]            # diffusers temb == official TeaCache e
        return out

    @functools.wraps(original_forward)
    def new_forward(self, *args, **kwargs):
        with mock.patch.object(self, "blocks", cached_blocks), mock.patch.object(
            self.condition_embedder, "forward", capturing_condition_embedder
        ):
            return original_forward(*args, **kwargs)

    transformer.forward = new_forward.__get__(transformer)
    object.__setattr__(transformer, "_nano_cache", controller)
    object.__setattr__(transformer, "_nano_original_forward", original_forward)
    return transformer


def remove_cache_from_transformer(transformer):
    """Undo apply_cache_on_transformer, restoring the original forward."""
    if hasattr(transformer, "_nano_original_forward"):
        transformer.forward = transformer._nano_original_forward
        object.__delattr__(transformer, "_nano_original_forward")
        object.__delattr__(transformer, "_nano_cache")
    return transformer
