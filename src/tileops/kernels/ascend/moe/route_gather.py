"""Route tokens to experts and gather them into the masked slab in one launch.

``AscendMoEPermuteAlignKernel`` sorts the routes into a block-aligned table and
``AscendMoEGatherSlabKernel`` reads that table back to copy each route's token row into
``slab[expert, rank]``. The two are one pipeline: the table exists only to tell the gather
where each row goes. This kernel keeps the block-aligned routing of the first -- per-block
histograms, a grid barrier, then a scatter pass -- and performs the row copy in that same
scatter pass, so the route table never reaches GM.

An expert's slab rows are its routes' ranks, which are the counts the routing already keeps:
``bases`` is how many of an expert's routes precede this block. A rank at or past
``max_m`` is dropped and sets the one-element overflow flag, ``counts`` is clamped to
``max_m`` so the GEMM never reads past the slab, and that route's ``inverse`` entry stays
unwritten (the caller aborts on overflow).
"""

import functools
from typing import ClassVar, Optional

import tilelang
import tilelang.ascend.language as T
import torch

from tileops._csrc import csrc_include
from tileops.kernels.ascend.moe.permute_align import _vector_capacity
from tileops.kernels.ascend.moe.pre_permute import route_pass_configs
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import GatherSlabCall, GatherSlabFwdInterface

__all__ = ["AscendMoERouteGatherKernel"]

# Routes per routing tile. The scatter pass copies one row per route, so the tile also sets
# how many blocks get work: 64 routes gives one tile per 64-route block of the split gather.
_TILE = 64


def make_route_gather(
    num_tokens: int,
    top_k: int,
    num_experts: int,
    max_m: int,
    hidden_size: int,
    dtype: str,
    grid: int,
):
    """Build the fused routing-and-gather program for one specialization."""
    num_routes = num_tokens * top_k
    tiles = -(-num_routes // _TILE)
    rounds = -(-tiles // grid)
    staged = min(grid, 8)

    @T.prim_func
    def main(
        ids: T.Tensor((num_routes,), "int32"),
        hidden: T.Tensor((num_tokens, hidden_size), dtype),
        slab: T.Tensor((num_experts, max_m, hidden_size), dtype),
        counts: T.Tensor((num_experts,), "int32"),
        inverse: T.Tensor((num_routes,), "int32"),
        overflow: T.Tensor((1,), "int32"),
        workspace: T.Tensor((grid, num_experts), "int32"),
    ):
        T.func_attr({"ascend.batch_schedule": True})
        with T.Kernel(grid) as block:
            data = T.alloc_shared((2 * _TILE,), "int32")
            hist = T.alloc_shared((num_experts,), "int32")
            row = T.alloc_shared((staged, num_experts), "int32")
            totals = T.alloc_shared((num_experts,), "int32")
            bases = T.alloc_shared((num_experts,), "int32")
            route_ub = T.alloc_shared((hidden_size,), dtype)
            x = T.alloc_var("int32")
            expert = T.alloc_var("int32")
            rank = T.alloc_var("int32")
            over = T.alloc_var("int32", init=0)
            with T.SimtVF(threads=128):
                t = T.get_thread_binding()
                for j in T.serial(T.ceildiv(num_experts, 128)):
                    if j * 128 + t < num_experts:
                        hist[j * 128 + t] = 0
                        totals[j * 128 + t] = 0
                        bases[j * 128 + t] = 0
            T.ascend_set_flag("V_S", 0)
            T.ascend_wait_flag("V_S", 0)
            # First pass: each block histograms its tiles into its workspace row.
            if block < tiles:
                T.copy(
                    ids[block * _TILE : block * _TILE + T.min(_TILE, num_routes - block * _TILE)],
                    data[: T.min(_TILE, num_routes - block * _TILE)],
                )
                T.ascend_set_flag("MTE2_S", 0)
            for r in T.serial(rounds):
                tile = block + r * grid
                if tile < tiles:
                    if r + 1 < rounds and block + (r + 1) * grid < tiles:
                        nxt = (block + (r + 1) * grid) * _TILE
                        if r % 2 == 0:
                            if r >= 1:
                                T.ascend_wait_flag("S_MTE2", 1)
                            T.copy(
                                ids[nxt : nxt + T.min(_TILE, num_routes - nxt)],
                                data[_TILE : _TILE + T.min(_TILE, num_routes - nxt)],
                            )
                            T.ascend_set_flag("MTE2_S", 1)
                        else:
                            T.ascend_wait_flag("S_MTE2", 0)
                            T.copy(
                                ids[nxt : nxt + T.min(_TILE, num_routes - nxt)],
                                data[: T.min(_TILE, num_routes - nxt)],
                            )
                            T.ascend_set_flag("MTE2_S", 0)
                    if r % 2 == 0:
                        T.ascend_wait_flag("MTE2_S", 0)
                    else:
                        T.ascend_wait_flag("MTE2_S", 1)
                    length = T.min(_TILE, num_routes - tile * _TILE)
                    for j in T.serial(length):
                        x = data[(r % 2) * _TILE + j]
                        hist[x] = hist[x] + 1
                    if r + 2 < rounds and block + (r + 2) * grid < tiles:
                        if r % 2 == 0:
                            T.ascend_set_flag("S_MTE2", 0)
                        else:
                            T.ascend_set_flag("S_MTE2", 1)
            T.ascend_set_flag("S_MTE3", 0)
            T.ascend_wait_flag("S_MTE3", 0)
            T.copy(hist, workspace[block, :])
            T.ascend_set_flag("MTE3_S", 0)
            T.ascend_wait_flag("MTE3_S", 0)
            if grid > 1:
                T.call_extern("handle", "tileops::moe_grid_sync")
            # Sum the per-block histograms; ``bases`` is how many of an expert's routes sit
            # before this block, which the scatter pass turns into a slab row.
            for group in T.serial(T.ceildiv(grid, staged)):
                first = group * staged
                batch = T.min(staged, grid - first)
                T.copy(workspace[first : first + batch, :], row[:batch, :])
                T.ascend_set_flag("MTE2_S", 0)
                T.ascend_wait_flag("MTE2_S", 0)
                T.ascend_set_flag("S_V", 0)
                T.ascend_wait_flag("S_V", 0)
                with T.SimtVF(threads=128):
                    t = T.get_thread_binding()
                    for j in T.serial(T.ceildiv(num_experts, 128)):
                        e = j * 128 + t
                        if e < num_experts:
                            for k in T.serial(batch):
                                totals[e] = totals[e] + row[k, e]
                                if first + k < block:
                                    bases[e] = bases[e] + row[k, e]
                T.ascend_set_flag("V_S", 0)
                T.ascend_wait_flag("V_S", 0)
                T.ascend_set_flag("S_MTE2", 0)
                T.ascend_wait_flag("S_MTE2", 0)
            # ``hist`` counts each expert's routes seen so far; an expert's run starts at
            # ``bases`` of them, so its value *is* the route's rank in the slab.
            for e in T.serial(num_experts):
                hist[e] = bases[e]
            # Second pass: re-read the ids, scatter each route's row into its slab slot.
            if block < tiles:
                T.copy(
                    ids[block * _TILE : block * _TILE + T.min(_TILE, num_routes - block * _TILE)],
                    data[: T.min(_TILE, num_routes - block * _TILE)],
                )
                T.ascend_set_flag("MTE2_S", 0)
            for r in T.serial(rounds):
                tile = block + r * grid
                if tile < tiles:
                    if r + 1 < rounds and block + (r + 1) * grid < tiles:
                        nxt = (block + (r + 1) * grid) * _TILE
                        if r % 2 == 0:
                            if r >= 1:
                                T.ascend_wait_flag("S_MTE2", 1)
                            T.copy(
                                ids[nxt : nxt + T.min(_TILE, num_routes - nxt)],
                                data[_TILE : _TILE + T.min(_TILE, num_routes - nxt)],
                            )
                            T.ascend_set_flag("MTE2_S", 1)
                        else:
                            T.ascend_wait_flag("S_MTE2", 0)
                            T.copy(
                                ids[nxt : nxt + T.min(_TILE, num_routes - nxt)],
                                data[: T.min(_TILE, num_routes - nxt)],
                            )
                            T.ascend_set_flag("MTE2_S", 0)
                    if r % 2 == 0:
                        T.ascend_wait_flag("MTE2_S", 0)
                    else:
                        T.ascend_wait_flag("MTE2_S", 1)
                    length = T.min(_TILE, num_routes - tile * _TILE)
                    for j in T.serial(length):
                        expert = data[(r % 2) * _TILE + j]
                        rank = hist[expert]
                        hist[expert] = rank + 1
                        if rank < max_m:
                            # MTE3_MTE2: the row buffer may be refilled only after its
                            # previous store left.
                            T.ascend_set_flag("MTE3_MTE2", 0)
                            T.ascend_wait_flag("MTE3_MTE2", 0)
                            T.copy(hidden[(tile * _TILE + j) // top_k, :], route_ub)
                            T.ascend_set_flag("MTE2_MTE3", 0)
                            T.ascend_wait_flag("MTE2_MTE3", 0)
                            T.copy(route_ub, slab[expert, rank, :])
                            inverse[tile * _TILE + j] = expert * max_m + rank
                        else:
                            over = 1
                    if r + 2 < rounds and block + (r + 2) * grid < tiles:
                        if r % 2 == 0:
                            T.ascend_set_flag("S_MTE2", 0)
                        else:
                            T.ascend_set_flag("S_MTE2", 1)
            T.ascend_set_flag("S_V", 0)
            T.ascend_wait_flag("S_V", 0)
            # Valid counts and the overflow flag, one expert range per block.
            with T.SimtVF(threads=128):
                t = T.get_thread_binding()
                for e in T.serial(block * num_experts // grid, (block + 1) * num_experts // grid):
                    if t == 0:
                        counts[e] = T.min(totals[e], max_m)
            if over == 1:
                with T.SimtVF(threads=128):
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
    grid: int,
):
    configs = route_pass_configs()
    if grid > 1:
        configs[tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS] = [
            *csrc_include("ascend_moe_grid_sync.h"),
            "-U__cpp_concepts",
        ]
    return tilelang.compile(
        make_route_gather(num_tokens, top_k, num_experts, max_m, hidden_size, dtype, grid),
        target="ascend",
        out_idx=[],
        pass_configs=configs,
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
        if not 0 < call.num_experts <= 384 or call.num_experts % 8:
            return "supports 8..384 experts divisible by 8"
        return super().refusal(call)

    def __init__(self, call: GatherSlabCall, *, grid: int | None = None) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(device_index=call.device.index if call.device is not None else None)
        self.call = call
        device = call.device.index if call.device is not None else torch.npu.current_device()
        capacity = _vector_capacity(device)
        if grid is None:
            grid = 1 if call.num_routes <= _TILE else capacity
        if type(grid) is not int or not 1 <= grid <= capacity:
            raise ValueError(f"grid must be between 1 and effective vector_core_num ({capacity})")
        self.device_index = device
        self.grid = grid
        self.aiv_capacity = capacity
        self._kernel = _compile(
            call.num_tokens,
            call.top_k,
            call.num_experts,
            call.max_m,
            call.hidden_size,
            self.dtype_to_str(call.dtype),
            grid,
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
        slab = torch.empty((c.num_experts, c.max_m, c.hidden_size), dtype=c.dtype, device=device)
        counts = torch.zeros(c.num_experts, dtype=torch.int32, device=device)
        inverse = torch.empty(c.num_routes, dtype=torch.int32, device=device)
        overflow = torch.zeros(1, dtype=torch.int32, device=device)
        workspace = torch.empty((self.grid, c.num_experts), dtype=torch.int32, device=device)
        self._kernel(
            topk_ids.view(-1),
            hidden_states,
            slab,
            counts,
            inverse,
            overflow,
            workspace,
        )
        return slab, counts, inverse, overflow
