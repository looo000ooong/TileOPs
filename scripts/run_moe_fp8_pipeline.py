"""End-to-end MoE FP8 pipeline on Ascend 950, for manual bring-up on an NPU box.

Chain::

    topk -> permute-align -> gather to masked slab -> FP8 quant (1x128)
         -> FP8 grouped GEMM (gate_up, N=2F) -> silu_and_mul
         -> FP8 quant (1x128) -> FP8 grouped GEMM (down, N=H)
         -> post-permute (weighted unpermute, [T, H])

Nothing here is registered in the manifest: every kernel is built from its call spec and
called directly, so the script runs against the current working tree.

Run::

    python scripts/run_moe_fp8_pipeline.py
    python scripts/run_moe_fp8_pipeline.py --tokens 64 --experts 32 --hidden 7168 --ffn 2048
    python scripts/run_moe_fp8_pipeline.py --skip-reference      # pipeline only, no torch oracle

Shapes must satisfy the kernel contracts:
  * ``hidden % 512 == 0`` and ``hidden % 128 == 0`` (GEMM K scale blocks, quant groups)
  * ``2 * ffn % 128 == 0`` and ``ffn % 512 == 0``
  * ``max_m % 128 == 0`` (or 64), ``experts % 8 == 0`` and ``8 <= experts <= 384``
  * ``tokens * top_k % 8 == 0``
"""

import argparse
import sys

import torch

from tileops.kernels.ascend.elementwise.silu_and_mul import AscendSiluAndMulKernel
from tileops.kernels.ascend.moe.fused_topk import AscendFusedTopKKernel
from tileops.kernels.ascend.moe.gather_slab import AscendMoEGatherSlabKernel
from tileops.kernels.ascend.moe.moe_grouped_gemm_fp8 import AscendMoEGroupedGemmFP8Kernel
from tileops.kernels.ascend.moe.permute_align import AscendMoEPermuteAlignKernel
from tileops.kernels.ascend.moe.post_permute_simd import AscendMoEPostPermuteKernel
from tileops.kernels.moe.call_spec import (
    FusedTopKCall,
    GatherSlabCall,
    MGroupedGemmFP8Call,
    PermuteAlignCall,
    PostPermuteCall,
)
from tileops.ops.moe.contracts import RoutingEpilogueSpec

FP8 = torch.float8_e4m3fn
BLOCK = 64
SCALE = 128
FP8_MAX = 448.0
EPS = 1e-4


# Rows per launch block. One block owns one row per iteration, and the grid may exceed the
# physical AIV count: the scheduler queues the surplus blocks, which keeps the MTE2/V/MTE3
# pipes of every core busy instead of leaving them serialized on one row each. Measured on
# Ascend950PR at rows=8192: 64 blocks 456 us, 256 blocks 289 us, flat past 256.
_QUANT_BLOCKS = 256


def quant_fp8_rows(rows, k, dtype_str, num_cores=None):
    """Build ``(rows, k) -> (fp8, float32 scale per 128 columns)`` for one 2-D operand.

    One core owns a row at a time; the row's 128-wide groups are reduced in SimdVF. The
    bf16/fp16 load uses ``UNPK_B16`` with ``part=0``, the pattern the shipped SiLU kernel
    uses to widen 64 elements per vector, and the FP8 store uses ``PK4_B32`` as the
    validated per-token quantization example does.

    ``num_cores`` must divide ``rows``: the row loop is a fixed ``ceildiv(rows, cores)``
    trip count and a core whose last row is out of range skips the flag-paired copies,
    which leaves the kernel waiting on a flag no core sets. The default therefore picks a
    divisor of ``rows`` near ``_QUANT_BLOCKS`` rather than the raw block count.
    """
    import tilelang
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    assert k % SCALE == 0
    groups = k // SCALE
    if num_cores is None:
        num_cores = min(_QUANT_BLOCKS, rows)
        while rows % num_cores:
            num_cores -= 1
    elif rows % num_cores:
        raise ValueError(f"num_cores={num_cores} must divide rows={rows}")

    @T.prim_func
    def main(
        X: T.Tensor((rows, k), dtype_str),
        Q: T.Tensor((rows, k), T.float8_e4m3fn),
        Sc: T.Tensor((rows, groups), "float32"),
    ):
        with T.Kernel(min(num_cores, rows)) as core:
            cores = min(num_cores, rows)
            x_ub = T.alloc_shared((k,), dtype_str)
            q_ub = T.alloc_shared((k,), T.float8_e4m3fn)
            s_ub = T.alloc_shared((groups,), "float32")
            for it in T.serial(T.ceildiv(rows, cores)):
                r = it * cores + core
                if r < rows:
                    T.copy(X[r, :], x_ub)
                    T.ascend_set_flag("MTE2_V", 0)
                    T.ascend_wait_flag("MTE2_V", 0)
                    with T.SimdVF():
                        fp8_max = S.vdup(FP8_MAX, "float32")
                        eps = S.vdup(EPS, "float32")
                        for gi in range(groups):
                            a0 = S.vcvt(S.vld(x_ub[gi * SCALE], dist="UNPK_B16"), "float32", part=0)
                            a1 = S.vcvt(
                                S.vld(x_ub[gi * SCALE + 64], dist="UNPK_B16"), "float32", part=0
                            )
                            amax = S.vcmax(S.vmax(S.vabs(a0), S.vabs(a1)))
                            amax = S.vmax(amax, eps)
                            scale = S.vdiv(amax, fp8_max)
                            brc = S.vdupv(scale)
                            S.vsts(s_ub[gi], scale, dist="ONEPT_B32")
                            S.vsts(
                                q_ub[gi * SCALE],
                                S.vcvt(S.vdiv(a0, brc), "float8_e4m3fn"),
                                dist="PK4_B32",
                            )
                            S.vsts(
                                q_ub[gi * SCALE + 64],
                                S.vcvt(S.vdiv(a1, brc), "float8_e4m3fn"),
                                dist="PK4_B32",
                            )
                    T.ascend_set_flag("V_MTE3", 0)
                    T.ascend_wait_flag("V_MTE3", 0)
                    T.copy(q_ub, Q[r, :])
                    T.copy(s_ub, Sc[r, :])
                    T.ascend_set_flag("MTE3_MTE2", 0)
                    T.ascend_wait_flag("MTE3_MTE2", 0)

    return tilelang.compile(main, target="ascend", out_idx=[])


def quantize(x, program_cache, num_cores=None):
    """Quantize ``[..., k]`` to ``(fp8, scale[..., k / 128])`` with the row kernel."""
    shape = x.shape
    k = shape[-1]
    rows = x.numel() // k
    key = (rows, k, str(x.dtype).removeprefix("torch."), num_cores)
    if key not in program_cache:
        program_cache[key] = quant_fp8_rows(rows, k, key[2], num_cores)
    q = torch.empty((rows, k), dtype=FP8, device=x.device)
    s = torch.empty((rows, k // SCALE), dtype=torch.float32, device=x.device)
    program_cache[key](x.contiguous().view(rows, k), q, s)
    return q.view(*shape), s.view(*shape[:-1], k // SCALE)


def quantize_weights(w):
    """Quantize ``[E, N, K]`` weights with one dequant scale per 128x128 block.

    Weights are static, so the block reduction runs in torch rather than in a kernel: the
    GEMM reads ``b * b_scale`` with one scale per 128 rows and 128 columns, and a row-wise
    scale cannot be folded into that layout.
    """
    e, n, k = w.shape
    view = w.float().view(e, n // SCALE, SCALE, k // SCALE, SCALE)
    amax = view.abs().amax(dim=(2, 4)).clamp_min(EPS)
    scale = amax / FP8_MAX
    q = (view / scale[:, :, None, :, None]).to(FP8)
    return q.view(e, n, k), scale.contiguous()


def reference(hidden, weights, ids, w_gate_up, w_down, ffn):
    """FP32 oracle for the routed expert MLP under the routing the pipeline produced."""
    tokens, hidden_size = hidden.shape
    out = torch.zeros((tokens, hidden_size), dtype=torch.float32, device=hidden.device)
    for g in range(w_gate_up.shape[0]):
        t_idx, k_idx = (ids == g).nonzero(as_tuple=True)
        if t_idx.numel() == 0:
            continue
        gate_up = hidden[t_idx].float() @ w_gate_up[g].float().T
        act = torch.nn.functional.silu(gate_up[:, :ffn]) * gate_up[:, ffn:]
        down = act @ w_down[g].float().T
        out.index_add_(0, t_idx, down * weights[t_idx, k_idx].unsqueeze(-1).float())
    return out.to(hidden.dtype)


def run(args):
    dev = torch.device(args.device)
    torch.manual_seed(args.seed)
    tokens, top_k = args.tokens, args.top_k
    experts, hidden, ffn = args.experts, args.hidden, args.ffn
    max_m = args.max_m
    routes = tokens * top_k
    if hidden % 512 or hidden % SCALE or (2 * ffn) % SCALE or ffn % 512:
        raise SystemExit("hidden/ffn do not meet the FP8 GEMM scale-block contracts")
    if max_m % BLOCK or experts % 8 or not 8 <= experts <= 384 or routes % 8:
        raise SystemExit("experts/max_m/route count do not meet the routing contracts")

    print(
        f"T={tokens} K={top_k} E={experts} H={hidden} F={ffn} max_m={max_m} "
        f"routes={routes} device={dev}"
    )

    hidden_states = (torch.randn((tokens, hidden), dtype=torch.float32, device=dev) * 0.1).to(
        torch.bfloat16
    )
    gating = torch.randn((tokens, experts), dtype=torch.float32, device=dev)
    w_gate_up = (
        torch.randn((experts, 2 * ffn, hidden), dtype=torch.float32, device=dev) * 0.02
    ).to(torch.bfloat16)
    w_down = (
        torch.randn((experts, hidden, ffn), dtype=torch.float32, device=dev) * 0.02
    ).to(torch.bfloat16)

    # 1. routing
    topk = AscendFusedTopKKernel(
        FusedTopKCall(
            num_tokens=tokens,
            num_experts=experts,
            top_k=top_k,
            scoring_func="softmax",
            renormalize=True,
            dtype=torch.float32,
            device=dev,
        )
    )
    weights, ids = topk(gating)
    print("topk:", tuple(weights.shape), tuple(ids.shape))

    # 2. route table (block-aligned sorted ids / owner / padded count)
    align = AscendMoEPermuteAlignKernel(
        PermuteAlignCall(num_routes=routes, num_experts=experts, block_size=BLOCK, device=dev)
    )
    sorted_ids, owner, post_pad = align(ids)
    print("permute-align: post_pad =", int(post_pad.item()))

    # 3. gather into the masked slab and invert the permutation
    gather = AscendMoEGatherSlabKernel(
        GatherSlabCall(
            num_tokens=tokens,
            top_k=top_k,
            num_experts=experts,
            max_m=max_m,
            hidden_size=hidden,
            block_size=BLOCK,
            dtype=torch.bfloat16,
            device=dev,
        )
    )
    slab, counts, inverse, overflow = gather(hidden_states, sorted_ids, owner, post_pad)
    if int(overflow.item()) != 0:
        raise SystemExit(
            f"an expert's routes overflowed max_m={max_m}: {counts.tolist()}. "
            "Raise --max-m or lower --tokens."
        )
    print("gather:", tuple(slab.shape), "counts sum =", int(counts.sum().item()))

    cache = {}
    # 4. activation FP8 quant (1x128)
    a_q, a_s = quantize(slab, cache)

    # 5. gate_up FP8 GEMM: [E, max_m, H] x [E, 2F, H] -> [E, max_m, 2F]
    gu_q, gu_s = quantize_weights(w_gate_up)
    gate_up_gemm = AscendMoEGroupedGemmFP8Kernel(
        MGroupedGemmFP8Call(
            kind="masked",
            max_m=max_m,
            ab_dtype=FP8,
            num_groups=experts,
            n=2 * ffn,
            k=hidden,
            a_scale_shape=(experts, max_m, hidden // SCALE),
            b_scale_shape=(experts, 2 * ffn // SCALE, hidden // SCALE),
            device=dev,
        )
    )
    gate_up = gate_up_gemm(a_q, a_s, gu_q, gu_s, counts)
    print("gate_up:", tuple(gate_up.shape))

    # 6. gated activation, then the down FP8 quant
    act = AscendSiluAndMulKernel(experts * max_m, ffn, torch.bfloat16)(
        gate_up.view(experts * max_m, 2 * ffn)
    ).view(experts, max_m, ffn)
    act_q, act_s = quantize(act, cache)

    # 7. down FP8 GEMM: [E, max_m, F] x [E, H, F] -> [E, max_m, H]
    down_q, down_s = quantize_weights(w_down)
    down_gemm = AscendMoEGroupedGemmFP8Kernel(
        MGroupedGemmFP8Call(
            kind="masked",
            max_m=max_m,
            ab_dtype=FP8,
            num_groups=experts,
            n=hidden,
            k=ffn,
            a_scale_shape=(experts, max_m, ffn // SCALE),
            b_scale_shape=(experts, hidden // SCALE, ffn // SCALE),
            device=dev,
        )
    )
    expert_out = down_gemm(act_q, act_s, down_q, down_s, counts)

    # 8. weighted unpermute
    post = AscendMoEPostPermuteKernel(
        PostPermuteCall(
            layout_key="masked_predicated",
            max_m=max_m,
            epilogue=RoutingEpilogueSpec(routed_scaling_factor=1.0),
            input_dtype=torch.bfloat16,
            routing_weight_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            num_experts=experts,
            materialized_rows=experts * max_m,
            num_tokens=tokens,
            hidden_size=hidden,
            top_k=top_k,
            device=dev,
        )
    )
    out = post(expert_out.view(experts * max_m, hidden), inverse, weights)
    print("out:", tuple(out.shape), out.dtype)

    if args.skip_reference:
        return 0

    ref = reference(hidden_states, weights, ids, w_gate_up, w_down, ffn)
    got, want = out.float(), ref.float()
    diff = (got - want).abs()
    denom = want.abs().clamp_min(1e-6)
    print(
        f"max |diff| = {diff.max().item():.4e}  "
        f"mean |diff| = {diff.mean().item():.4e}  "
        f"max rel = {(diff / denom).max().item():.4e}"
    )
    tol = 2e-2 * (1.0 + (hidden / 1024) ** 0.5)
    ok = torch.allclose(got, want, rtol=2e-2, atol=tol)
    print(f"allclose(rtol=2e-2, atol={tol:.3f}) -> {ok}")
    return 0 if ok else 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="npu")
    p.add_argument("--tokens", type=int, default=64)
    p.add_argument("--top-k", type=int, default=8)
    p.add_argument("--experts", type=int, default=32)
    p.add_argument("--hidden", type=int, default=7168)
    p.add_argument("--ffn", type=int, default=2048)
    p.add_argument("--max-m", type=int, default=128)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--skip-reference", action="store_true")
    args = p.parse_args(argv)
    if not args.skip_reference and args.tokens * args.top_k > args.experts * args.max_m:
        print(
            "warning: routes exceed E * max_m; routing is unlikely to fit and gather will "
            "report overflow",
            file=sys.stderr,
        )
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
