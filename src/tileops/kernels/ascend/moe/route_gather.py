"""Route tokens to experts and gather them into the masked slab.

Two launches, with the kernel boundary as the only global synchronization: the first
histograms each core's route range into GM, the second reads every core's histogram,
derives each expert's global prefix, and copies rows into the slab. A software grid
barrier inside one kernel was the previous shape; it costs a full round trip and keeps
the two passes from overlapping.

The count pass uses the hardware histogram (``dhistv2``, 64 bins per instruction) rather
than a scalar per-route increment, and the gather pass walks its routes under
``T.Pipelined`` so the row copy overlaps the next route's index read.

Every expert owns ``max_m`` slab rows, so a route's rank is its position inside its
expert's run -- the running count the gather already keeps. ``counts`` is clamped to
``max_m`` so the GEMM never reads past the slab, and a run longer than that sets the
one-element overflow flag; the caller aborts on it.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.kernels.ascend.moe.permute_align import _vector_capacity
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import GatherSlabCall, GatherSlabFwdInterface

__all__ = ["AscendMoERouteGatherKernel"]

# Bins one hardware histogram instruction covers.
_BIN_GROUP = 64
# Routes the count pass handles per vector block.
_COUNT_BLOCK = 256


def _count_program(num_tokens: int, top_k: int, num_experts: int, cores: int):
    """Build the per-core histogram pass."""
    rows_per_core = -(-num_tokens // cores)
    max_routes_this_core = rows_per_core * top_k
    e_pad = -(-num_experts // _BIN_GROUP) * _BIN_GROUP
    ids_cap = max(_COUNT_BLOCK, -(-max_routes_this_core // _COUNT_BLOCK) * _COUNT_BLOCK)

    @T.prim_func
    def main(
        flat_ids: T.Tensor((num_tokens * top_k,), "int32"),
        block_counts: T.Tensor((cores * e_pad,), "int32"),
    ):
        with T.Kernel(cores) as core:
            token_start = core * rows_per_core
            token_end = T.min(token_start + rows_per_core, num_tokens)
            rows_this_core = T.max(token_end - token_start, 0)
            route_start = token_start * top_k
            routes_this_core = rows_this_core * top_k

            ids32 = T.alloc_shared((ids_cap,), "int32")
            ids8 = T.alloc_shared((ids_cap,), "uint8")
            counts = T.alloc_shared((e_pad,), "int32")

            if rows_this_core > 0:
                T.copy(
                    flat_ids[route_start : route_start + routes_this_core],
                    ids32[0:routes_this_core],
                )

            with T.SimdVF():
                zero_vec = T.simd.vdup(T.int32(0), "int32")
                for group in T.unroll(e_pad // _BIN_GROUP):
                    T.simd.vsts(counts[group * _BIN_GROUP], zero_vec)

            if routes_this_core > 0:
                for block in range((max_routes_this_core + _COUNT_BLOCK - 1) // _COUNT_BLOCK):
                    base = block * _COUNT_BLOCK
                    if base < routes_this_core:
                        rem = T.min(_COUNT_BLOCK, routes_this_core - base)
                        with T.SimdVF():
                            m16 = T.simd.pset(16)
                            # Narrow the ids to uint8 so one histogram instruction covers 64
                            # bins; the host bounds num_experts by 256.
                            for unit in range(4):
                                values = T.simd.vld(ids32[base + unit * 64])
                                low8 = T.simd.vcvt(values, "uint8", part=0)
                                T.simd.vsts(ids8[base + unit * 64], low8, dist="PK4_B32")
                            T.simd.mem_bar("VST_VLD")
                            src = T.simd.vld(ids8[base], dist="NORM")
                            m8 = T.simd.update_mask(rem, 8)
                            freq0 = T.simd.alloc_var("uint16")
                            freq0 = T.simd.vdup(T.uint16(0), "uint16", m16)
                            T.simd.dhistv2(freq0, src, m8, bin=0)
                            wide0 = T.reinterpret(T.simd.vunpack(freq0, part="LOWER"), "int32x64")
                            T.simd.vsts(counts[0], T.simd.vadd(T.simd.vld(counts[0]), wide0))
                            for group in T.unroll(1, e_pad // _BIN_GROUP):
                                offset = group * _BIN_GROUP
                                shifted = T.simd.vsub(
                                    src, T.simd.vdup(T.uint8(offset), "uint8"), m8
                                )
                                lower = T.simd.vcmps(src, T.int32(offset), m8, op="ge")
                                upper = T.simd.vcmps(shifted, T.int32(_BIN_GROUP), m8, op="lt")
                                keep = T.simd.pand(lower, upper, m8)
                                freq = T.simd.alloc_var("uint16")
                                freq = T.simd.vdup(T.uint16(0), "uint16", m16)
                                T.simd.dhistv2(freq, shifted, keep, bin=0)
                                wide = T.reinterpret(T.simd.vunpack(freq, part="LOWER"), "int32x64")
                                T.simd.vsts(
                                    counts[group * _BIN_GROUP],
                                    T.simd.vadd(T.simd.vld(counts[group * _BIN_GROUP]), wide),
                                )

            T.copy(
                counts[0:num_experts],
                block_counts[core * e_pad : core * e_pad + num_experts],
            )

    return main


def _gather_program(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_m: int,
    hidden_size: int,
    dtype: str,
    cores: int,
):
    """Build the prefix-and-copy pass."""
    rows_per_core = -(-num_tokens // cores)
    e_pad = -(-num_experts // _BIN_GROUP) * _BIN_GROUP
    ids_cap = max(_COUNT_BLOCK, -(-(rows_per_core * top_k) // _COUNT_BLOCK) * _COUNT_BLOCK)

    @T.prim_func
    def main(
        hidden: T.Tensor((num_tokens, hidden_size), dtype),
        flat_ids: T.Tensor((num_tokens * top_k,), "int32"),
        block_counts: T.Tensor((cores * e_pad,), "int32"),
        slab: T.Tensor((num_experts, max_m, hidden_size), dtype),
        counts: T.Tensor((num_experts,), "int32"),
        inverse: T.Tensor((num_tokens * top_k,), "int32"),
        overflow: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(cores) as core:
            token_start = core * rows_per_core
            token_end = T.min(token_start + rows_per_core, num_tokens)
            rows_this_core = T.max(token_end - token_start, 0)
            route_start = token_start * top_k
            routes_this_core = rows_this_core * top_k

            ids32 = T.alloc_shared((ids_cap,), "int32")
            all_counts = T.alloc_shared((cores * e_pad,), "int32")
            totals = T.alloc_shared((e_pad,), "int32")
            prefix = T.alloc_shared((e_pad,), "int32")
            cursors = T.alloc_shared((e_pad,), "int32")
            row_buf = T.alloc_shared((hidden_size,), dtype)

            if rows_this_core > 0:
                T.copy(
                    flat_ids[route_start : route_start + routes_this_core],
                    ids32[0:routes_this_core],
                )
            T.copy(block_counts[0 : cores * e_pad], all_counts[0 : cores * e_pad])

            # ``totals`` is every core's count for an expert; ``prefix`` is the count in the
            # cores before this one, which is exactly the rank this core's first route takes.
            for g in range(e_pad // _BIN_GROUP):
                with T.SimdVF():
                    zero_vec = T.simd.vdup(T.int32(0), "int32")
                    T.simd.vsts(totals[g * _BIN_GROUP], zero_vec)
                    T.simd.vsts(prefix[g * _BIN_GROUP], zero_vec)
                    T.simd.mem_bar("VST_VLD")
                    for other in range(cores):
                        row_vec = T.simd.vld(all_counts[other * e_pad + g * _BIN_GROUP])
                        T.simd.vsts(
                            totals[g * _BIN_GROUP],
                            T.simd.vadd(T.simd.vld(totals[g * _BIN_GROUP]), row_vec),
                        )
                        if other < core:
                            T.simd.vsts(
                                prefix[g * _BIN_GROUP],
                                T.simd.vadd(T.simd.vld(prefix[g * _BIN_GROUP]), row_vec),
                            )
                        T.simd.mem_bar("VST_VLD")

            for e in T.serial(num_experts):
                cursors[e] = prefix[e]

            if core == 0:
                # The GEMM reads at most max_m rows; a longer run is reported, not cut silently.
                clamped = T.alloc_shared((e_pad,), "int32")
                over_ub = T.alloc_shared((1,), "int32")
                over_ub[0] = 0
                for e in T.serial(num_experts):
                    clamped[e] = T.min(totals[e], max_m)
                    if totals[e] > max_m:
                        over_ub[0] = 1
                T.copy(clamped[0:num_experts], counts[0:num_experts])
                T.copy(over_ub, overflow)

            for r in T.Pipelined(routes_this_core, num_stages=2):
                e = ids32[r]
                if e >= 0 and e < num_experts:
                    rank = cursors[e]
                    cursors[e] = rank + 1
                    inverse[route_start + r] = e * max_m + rank
                    if rank < max_m:
                        token_row = (route_start + r) // top_k
                        T.copy(hidden[token_row, 0:hidden_size], row_buf[0:hidden_size])
                        T.copy(row_buf[0:hidden_size], slab[e, rank, 0:hidden_size])
                else:
                    inverse[route_start + r] = -1

    return main


@functools.lru_cache(maxsize=32)
def _compile(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_m: int,
    hidden_size: int,
    dtype: str,
    cores: int,
):
    return (
        tilelang.compile(
            _count_program(num_tokens, top_k, num_experts, cores),
            target="ascend",
            out_idx=[],
        ),
        tilelang.compile(
            _gather_program(num_tokens, top_k, num_experts, max_m, hidden_size, dtype, cores),
            target="ascend",
            out_idx=[],
        ),
    )


class AscendMoERouteGatherKernel(Kernel, GatherSlabFwdInterface):
    """Route ``topk_ids`` and gather each route's token row into the masked slab."""

    devices: ClassVar[frozenset[str]] = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: GatherSlabCall) -> Optional[str]:
        if call.dtype not in (torch.float16, torch.bfloat16):
            return f"requires float16 or bfloat16, got {call.dtype}"
        if call.block_size <= 0 or call.block_size % 64:
            return f"needs block_size a positive multiple of 64, got {call.block_size}"
        if call.max_m <= 0 or call.max_m % call.block_size:
            return f"needs max_m a positive multiple of block_size, got {call.max_m}"
        if min(call.num_tokens, call.top_k, call.num_experts, call.hidden_size) <= 0:
            return "requires positive token, route, expert and hidden dimensions"
        if call.hidden_size % 16:
            return f"requires hidden_size divisible by 16, got {call.hidden_size}"
        # The count pass narrows ids to uint8, so one histogram instruction covers 64 bins.
        if not 0 < call.num_experts <= 256:
            return f"supports 1..256 experts, got {call.num_experts}"
        return super().refusal(call)

    def __init__(self, call: GatherSlabCall, *, cores: int | None = None) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        device = call.device.index if call.device is not None else torch.npu.current_device()
        capacity = _vector_capacity(device)
        if cores is None:
            cores = min(capacity, call.num_tokens)
        if type(cores) is not int or not 1 <= cores <= capacity:
            raise ValueError(f"cores must be between 1 and effective vector_core_num ({capacity})")
        self.device_index = device
        self.cores = cores
        self.aiv_capacity = capacity
        self._e_pad = -(-call.num_experts // _BIN_GROUP) * _BIN_GROUP
        self._count, self._gather = _compile(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            call.max_m,
            call.hidden_size,
            self.dtype_to_str(call.dtype),
            cores,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ):
        c = self.call
        self._require_devices(hidden_states=hidden_states, topk_ids=topk_ids)
        if tuple(hidden_states.shape) != (c.num_tokens, c.hidden_size):
            raise ValueError("hidden_states does not match the route-gather specialization")
        if hidden_states.dtype != c.dtype or not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous in the specialization's dtype")
        if (
            topk_ids.dtype != torch.int32
            or not topk_ids.is_contiguous()
            or topk_ids.numel() != c.num_routes
        ):
            raise ValueError("topk_ids must be contiguous int32 with num_tokens * top_k elements")
        device = hidden_states.device
        flat_ids = topk_ids.view(-1)
        slab = torch.empty((c.num_experts, c.max_m, c.hidden_size), dtype=c.dtype, device=device)
        counts = torch.empty(c.num_experts, dtype=torch.int32, device=device)
        inverse = torch.empty(c.num_routes, dtype=torch.int32, device=device)
        overflow = torch.zeros(1, dtype=torch.int32, device=device)
        block_counts = torch.empty((self.cores * self._e_pad,), dtype=torch.int32, device=device)
        self._count(flat_ids, block_counts)
        self._gather(hidden_states, flat_ids, block_counts, slab, counts, inverse, overflow)
        return slab, counts, inverse, overflow
