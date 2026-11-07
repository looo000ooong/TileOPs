"""Gather routed token rows into a masked expert slab on Ascend 950.

The route table is the one ``AscendMoEPermuteAlignKernel`` returns: routes grouped by
expert, each expert's run padded to a 64-route block and starting on a block boundary,
padding slots holding ``num_routes``. This kernel reads that table and copies each
route's token row into ``slab[expert, rank]``, where ``rank`` is the route's position
inside its expert's run, so the slab is the ``[E, max_m, H]`` input the masked FP8
grouped GEMM reads.

Two facts are derived on the device, never on the host:

* an expert's slab start is the first padded block naming it, found by binary search
  over the monotonic ``expert_ids`` vector;
* an expert's valid count is known to the block holding its last padded block, which
  scans that block's tail for the ``num_routes`` sentinel.

The data movement is MTE (a row copy in, a row copy out); the only scattered writes are
the per-route inverse map, which one ``SimtVF`` per padded block performs from the route
and destination lists staged in UB -- the same shape ``AscendMoEPermuteAlignKernel``
writes its own outputs with.

An expert whose routed count exceeds ``max_m`` loses the overflowing rows and sets the
one-element overflow flag; ``counts`` is clamped to ``max_m`` so the GEMM never reads
past the slab.
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops.kernels.ascend.moe.pre_permute import route_pass_configs
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import GatherSlabCall, GatherSlabFwdInterface

__all__ = ["AscendMoEGatherSlabKernel"]

# Pure AIV kernel: one block per 64-route tile, capped by the ordinary vector-core count.
_NUM_BLOCKS_CAP = 64
# Threads the scattered inverse store and the single-scalar stores launch with.
_VF_THREADS = 32


def make_gather_slab(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_m: int,
    hidden_size: int,
    dtype: str,
    block_size: int,
    num_blocks: int,
):
    """Build the gather program; every extent is a compile-time fact of the call."""
    num_routes = num_tokens * top_k
    padded_routes = num_routes + (num_experts + 1) * (block_size - 1)
    padded_blocks = -(-padded_routes // block_size)

    @T.prim_func
    def main(
        hidden: T.Tensor((num_tokens, hidden_size), dtype),
        sorted_ids: T.Tensor((padded_routes,), "int32"),
        owner: T.Tensor((padded_blocks,), "int32"),
        post_pad: T.Tensor((1,), "int32"),
        slab: T.Tensor((num_experts, max_m, hidden_size), dtype),
        counts: T.Tensor((num_experts,), "int32"),
        inverse: T.Tensor((num_routes,), "int32"),
        overflow: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(num_blocks) as block:
            row_ub = T.alloc_shared((hidden_size,), dtype)
            routes_ub = T.alloc_shared((block_size,), "int32")
            dests_ub = T.alloc_shared((block_size,), "int32")
            total = T.alloc_var("int32")
            group = T.alloc_var("int32")
            lo = T.alloc_var("int32")
            hi = T.alloc_var("int32")
            mid = T.alloc_var("int32")
            rank0 = T.alloc_var("int32")
            rows = T.alloc_var("int32")
            last = T.alloc_var("int32")
            last_block = T.alloc_var("int32")
            over = T.alloc_var("int32", init=0)
            route = T.alloc_var("int32")
            dest = T.alloc_var("int32")

            for it in T.serial(T.ceildiv(padded_blocks, num_blocks)):
                b = it * num_blocks + block
                total = post_pad[0]
                if b * block_size < total:
                    group = owner[b]
                    # First padded block of this expert: the run start in the padded table.
                    lo = 0
                    hi = b
                    while lo < hi:
                        mid = (lo + hi) // 2
                        if owner[mid] < group:
                            lo = mid + 1
                        else:
                            hi = mid
                    rank0 = b * block_size - lo * block_size
                    rows = T.min(block_size, total - b * block_size)

                    for j in T.serial(rows):
                        route = sorted_ids[b * block_size + j]
                        routes_ub[j] = route
                        dests_ub[j] = -1
                        if route < num_routes:
                            if rank0 + j < max_m:
                                dest = group * max_m + rank0 + j
                                dests_ub[j] = dest
                                # MTE3_MTE2: the row buffer may be refilled only after its
                                # previous store left.
                                T.ascend_set_flag("MTE3_MTE2", 0)
                                T.ascend_wait_flag("MTE3_MTE2", 0)
                                T.copy(hidden[route // top_k, :], row_ub)
                                T.ascend_set_flag("MTE2_MTE3", 0)
                                T.ascend_wait_flag("MTE2_MTE3", 0)
                                T.copy(row_ub, slab[group, rank0 + j, :])
                            else:
                                over = 1

                    # One vector pass scatters this block's route -> slab-row map.
                    T.ascend_set_flag("S_V", 0)
                    T.ascend_wait_flag("S_V", 0)
                    with T.SimtVF(threads=_VF_THREADS):
                        t = T.get_thread_binding()
                        for j in T.serial(T.ceildiv(block_size, _VF_THREADS)):
                            slot = j * _VF_THREADS + t
                            if slot < rows:
                                if dests_ub[slot] >= 0:
                                    inverse[routes_ub[slot]] = dests_ub[slot]
                    T.ascend_set_flag("V_S", 0)
                    T.ascend_wait_flag("V_S", 0)

                    # The block holding this expert's last padded block knows its valid count.
                    last_block = 0
                    if b + 1 < padded_blocks:
                        if owner[b + 1] != group:
                            last_block = 1
                    else:
                        last_block = 1
                    if last_block == 1:
                        last = -1
                        for j in T.serial(rows):
                            if sorted_ids[b * block_size + j] < num_routes:
                                last = j
                        with T.SimtVF(threads=_VF_THREADS):
                            t = T.get_thread_binding()
                            if t == 0:
                                counts[group] = T.min(rank0 + last + 1, max_m)

            if over == 1:
                with T.SimtVF(threads=_VF_THREADS):
                    t = T.get_thread_binding()
                    if t == 0:
                        overflow[0] = 1

    return main


@functools.lru_cache(maxsize=32)
def _compile(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_m: int,
    hidden_size: int,
    dtype: str,
    block_size: int,
    num_blocks: int,
):
    return tilelang.compile(
        make_gather_slab(
            num_tokens,
            top_k,
            num_experts,
            max_m,
            hidden_size,
            dtype,
            block_size,
            num_blocks,
        ),
        target="ascend",
        out_idx=[],
        pass_configs=route_pass_configs(),
    )


class AscendMoEGatherSlabKernel(Kernel, GatherSlabFwdInterface):
    """Copy routed token rows into the masked ``[E, max_m, H]`` expert slab."""

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
        return super().refusal(call)

    def __init__(self, call: GatherSlabCall) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        self.num_blocks = min(_NUM_BLOCKS_CAP, call.padded_blocks)
        self._kernel = _compile(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            call.max_m,
            call.hidden_size,
            self.dtype_to_str(call.dtype),
            call.block_size,
            self.num_blocks,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        sorted_token_ids: torch.Tensor,
        expert_ids: torch.Tensor,
        num_tokens_post_pad: torch.Tensor,
    ):
        c = self.call
        self._require_devices(hidden_states=hidden_states)
        if tuple(hidden_states.shape) != (c.num_tokens, c.hidden_size):
            raise ValueError("hidden_states does not match the gather specialization")
        if hidden_states.dtype != c.dtype or not hidden_states.is_contiguous():
            raise ValueError("hidden_states must be contiguous in the specialization's dtype")
        if sorted_token_ids.dtype != torch.int32 or sorted_token_ids.numel() != c.padded_routes:
            raise ValueError("sorted_token_ids must be int32 of the permute-align length")
        if expert_ids.dtype != torch.int32 or expert_ids.numel() != c.padded_blocks:
            raise ValueError("expert_ids must be int32 of one entry per padded block")
        if num_tokens_post_pad.numel() != 1 or num_tokens_post_pad.dtype != torch.int32:
            raise ValueError("num_tokens_post_pad must be one int32 element")
        device = hidden_states.device
        slab = torch.empty(
            (c.num_experts, c.max_m, c.hidden_size), dtype=c.dtype, device=device
        )
        counts = torch.zeros(c.num_experts, dtype=torch.int32, device=device)
        inverse = torch.empty(c.num_routes, dtype=torch.int32, device=device)
        overflow = torch.zeros(1, dtype=torch.int32, device=device)
        self._kernel(
            hidden_states,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_pad,
            slab,
            counts,
            inverse,
            overflow,
        )
        return slab, counts, inverse, overflow
