"""SIMD (SimdVF) MoE post-permute weighted unpermute for Ascend 950.

Structure:

  1. Outside the vector scope, MTE stages the K source rows of one token into
     UB (row-granularity copies, up to H*itemsize bytes each) plus the K fp32
     routing weights.
  2. ONE SimdVF per (token, hidden span) keeps the FP32 accumulator in vector
     registers across the whole K loop: vld the bf16/fp16 row, vcvt the
     even/odd lane halves to fp32, vmula FMA against the broadcast weight.
     The epilogue narrows both halves back (vcvt PART_EVEN/PART_ODD, inactive
     lanes zeroed) and merges them with a bitwise vor on uint16-reinterpreted
     lanes (asc_or has no bf16 overload), then masked-stores via update_mask so
     partial 128-element tails stay in range.
  3. One MTE copy writes the finished output row back to GM.

Rows are addressed by ``inverse_indices`` only, so a masked slab
(``materialized_rows == num_experts * max_m``) and a packed layout
(``materialized_rows == num_tokens * top_k``) share this kernel.
"""

import functools

import torch

from tileops.kernels.buffer_utils import tensors_overlap
from tileops.kernels.kernel_base import Kernel
from tileops.kernels.moe.call_spec import PostPermuteCall, PostPermuteFwdInterface

__all__ = ["AscendMoEPostPermuteKernel", "torch_reference"]

# One 2048-bit vector register holds 128 16-bit elements.
VREG_LANES = 128


@functools.lru_cache(maxsize=32)
def _make_simd_unpermute(
    num_tokens,
    top_k,
    hidden_size,
    materialized_rows,
    dtype,
    scaling=1.0,
    *,
    out_dtype=None,
    num_cores=64,
):
    """One owner core per token; register-resident FP32 K reduction per span."""
    import tilelang.ascend.language as T
    from tilelang.ascend.language import simd as S

    numel = num_tokens * top_k
    out_dtype = dtype if out_dtype is None else out_dtype
    cores = min(num_tokens, num_cores)
    itemsize = 2  # fp16/bf16
    h_pad = -(-hidden_size // VREG_LANES) * VREG_LANES
    # K staged source rows + one staged output row must fit one core's UB.
    budget = 176 * 1024
    h_span = min(
        h_pad,
        max(VREG_LANES, budget // ((top_k + 1) * itemsize) // VREG_LANES * VREG_LANES),
    )
    n_spans = -(-h_pad // h_span)
    span_chunks = h_span // VREG_LANES

    @T.prim_func
    def _simd_unpermute_main(
        expert_output: T.Tensor((materialized_rows, hidden_size), dtype),
        inverse_indices: T.Tensor((numel,), "int32"),
        topk_weights: T.Tensor((num_tokens, top_k), "float32"),
        output: T.Tensor((num_tokens, hidden_size), out_dtype),
    ):
        with T.Kernel(cores) as core:
            src = T.alloc_shared((top_k, h_span), dtype)
            out_ub = T.alloc_shared((h_span,), out_dtype)
            w_ub = T.alloc_shared((top_k, 64), "float32")
            for iteration in T.serial(T.ceildiv(num_tokens, cores)):
                token_idx = T.cast(iteration, "int64") * cores + core
                if token_idx < num_tokens:
                    for span in T.serial(n_spans):
                        s0 = span * h_span
                        span_valid = T.min(hidden_size - s0, h_span)
                        for k in T.serial(top_k):
                            flat_idx = token_idx * top_k + k
                            slot = T.cast(inverse_indices[flat_idx], "int64")
                            T.copy(expert_output[slot, s0 : s0 + span_valid], src[k, 0 : span_valid])
                            w_ub[k, 0] = topk_weights[token_idx, k]
                        with T.SimdVF():
                            m32 = S.pset(32)
                            m16 = S.pset(16)
                            acc_e = S.alloc_local((1,), "float32")
                            acc_o = S.alloc_local((1,), "float32")
                            for c in T.serial(span_chunks):
                                base = c * VREG_LANES
                                acc_e[0] = S.vdup(T.float32(0.0), "float32", m32)
                                acc_o[0] = S.vdup(T.float32(0.0), "float32", m32)
                                for k in T.serial(top_k):
                                    s_vec = S.vld(src[k, base])
                                    w_vec = S.vdupv(S.vld(w_ub[k, 0]))
                                    # bf16/fp16 -> fp32: even/odd lane halves of one register.
                                    S.vmula(acc_e[0], S.vcvt(s_vec, "float32", part=0), w_vec, m32)
                                    S.vmula(acc_o[0], S.vcvt(s_vec, "float32", part=1), w_vec, m32)
                                if scaling != 1.0:
                                    acc_e[0] = S.vmuls(acc_e[0], T.float32(scaling), m32)
                                    acc_o[0] = S.vmuls(acc_o[0], T.float32(scaling), m32)
                                # fp32 -> bf16/fp16 halves (PART_EVEN/PART_ODD,
                                # inactive lanes zeroed), merged by a bitwise
                                # vor on reinterpreted uint16 lanes.
                                even_bits = T.reinterpret(
                                    S.vcvt(acc_e[0], out_dtype, part=0), "uint16x128"
                                )
                                odd_bits = T.reinterpret(
                                    S.vcvt(acc_o[0], out_dtype, part=1), "uint16x128"
                                )
                                merged = T.reinterpret(
                                    S.vor(even_bits, odd_bits, m16),
                                    out_dtype + "x128",
                                )
                                valid = T.min(hidden_size - s0 - base, VREG_LANES)
                                S.vsts(out_ub[base], merged, S.update_mask(valid, 16))
                        T.copy(out_ub[0 : span_valid], output[token_idx, s0 : s0 + span_valid])

    return _simd_unpermute_main


@functools.lru_cache(maxsize=32)
def _compile(
    num_tokens,
    top_k,
    hidden_size,
    materialized_rows,
    dtype,
    out_dtype,
    scaling,
    num_cores,
):
    import tilelang

    return tilelang.compile(
        _make_simd_unpermute(
            num_tokens,
            top_k,
            hidden_size,
            materialized_rows,
            dtype,
            scaling,
            out_dtype=out_dtype,
            num_cores=num_cores,
        ),
        target="ascend",
        out_idx=[],
    )


class AscendMoEPostPermuteKernel(Kernel, PostPermuteFwdInterface):
    """SIMD weighted inverse-permute with register-resident FP32 routing math.

    ``inverse_indices`` values stay on device and must be in
    ``[0, materialized_rows)``; rows may be referenced repeatedly, and
    ``materialized_rows`` need not equal ``T * K`` (a masked slab passes
    ``num_experts * max_m``).
    """

    devices = frozenset({"npu"})
    supported_archs = None

    @classmethod
    def refusal(cls, call: PostPermuteCall) -> "str | None":
        if call.input_dtype not in (torch.float16, torch.bfloat16):
            return f"requires float16 or bfloat16, got {call.input_dtype}"
        if call.output_dtype is not None and call.output_dtype not in (
            call.input_dtype,
            torch.float16,
            torch.bfloat16,
        ):
            return f"writes float16 or bfloat16, got {call.output_dtype}"
        if min(call.num_tokens, call.top_k, call.hidden_size, call.materialized_rows) <= 0:
            return "requires positive token, route, hidden and materialized-row counts"
        return super().refusal(call)

    def __init__(self, call: PostPermuteCall, config=None) -> None:
        reason = self.refusal(call)
        if reason is not None:
            raise ValueError(reason)
        super().__init__(
            device_index=call.device.index if call.device is not None else None
        )
        self.call = call
        self.num_tokens = call.num_tokens
        self.top_k = call.top_k
        self.hidden_size = call.hidden_size
        self.materialized_rows = call.materialized_rows
        self.dtype = call.input_dtype
        self.out_dtype = call.input_dtype if call.output_dtype is None else call.output_dtype
        self.scaling = 1.0 if call.epilogue is None else call.epilogue.routed_scaling_factor
        self.init_config(config)

    @property
    def default_config(self) -> dict:
        return {"num_cores": 64}

    def make_program(self):
        return _compile(
            self.num_tokens,
            self.top_k,
            self.hidden_size,
            self.materialized_rows,
            self.dtype_str,
            str(self.out_dtype).removeprefix("torch."),
            self.scaling,
            self.config["num_cores"],
        )

    def forward(
        self,
        expert_output: torch.Tensor,
        inverse_indices: torch.Tensor,
        topk_weights: torch.Tensor,
        out: "torch.Tensor | None" = None,
    ) -> torch.Tensor:
        self._require_devices(
            expert_output=expert_output,
            inverse_indices=inverse_indices,
            topk_weights=topk_weights,
        )
        shape = (self.num_tokens, self.hidden_size)
        rows = self.num_tokens * self.top_k
        if tuple(expert_output.shape) != (self.materialized_rows, self.hidden_size):
            raise ValueError(
                f"expert_output must be {(self.materialized_rows, self.hidden_size)}, "
                f"got {tuple(expert_output.shape)}"
            )
        if expert_output.dtype != self.dtype:
            raise ValueError(f"expert_output dtype must be {self.dtype}")
        if tuple(inverse_indices.shape) != (rows,) or inverse_indices.dtype != torch.int32:
            raise ValueError(f"inverse_indices must be int32 of {rows} elements")
        if (
            tuple(topk_weights.shape) != (self.num_tokens, self.top_k)
            or topk_weights.dtype != torch.float32
        ):
            raise ValueError("topk_weights must be float32 of (num_tokens, top_k)")
        if out is None:
            output = torch.empty(shape, dtype=self.out_dtype, device=expert_output.device)
        else:
            if tuple(out.shape) != shape:
                raise ValueError(f"out shape must be {shape}, got {tuple(out.shape)}")
            if out.dtype != self.out_dtype:
                raise ValueError(f"out dtype must be {self.out_dtype}")
            if out.device != expert_output.device or not out.is_contiguous():
                raise ValueError("out must be contiguous and on the input's device")
            for name, tensor in (
                ("expert_output", expert_output),
                ("inverse_indices", inverse_indices),
                ("topk_weights", topk_weights),
            ):
                if tensors_overlap(out, tensor):
                    raise ValueError(f"out must not overlap {name} in memory")
            output = out
        self.make_program()(expert_output, inverse_indices, topk_weights, output)
        return output


def torch_reference(
    expert_output, topk_weights, inverse_indices, routed_scaling_factor=1.0, *, out_dtype=None
):
    """CPU oracle: gather each route's row, weight it, sum in fp32, then cast."""
    tokens, top_k = topk_weights.shape
    rows = expert_output.reshape(-1, expert_output.shape[-1])
    gathered = rows[inverse_indices.long()].float().view(tokens, top_k, -1)
    out = (gathered * topk_weights.unsqueeze(-1)).sum(dim=1) * routed_scaling_factor
    return out.to(expert_output.dtype if out_dtype is None else out_dtype)


try:
    from tileops.backend import register_kernel_type

    register_kernel_type("MoEPostPermuteFwdOp", "ascend", AscendMoEPostPermuteKernel)
except ImportError:  # pragma: no cover - import guard shared with the other Ascend kernels
    pass
