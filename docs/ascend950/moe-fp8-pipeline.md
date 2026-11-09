# MoE FP8 链路（Ascend 950）背景与真机执行说明

本文件说明 `feat/ascend-moe-fp8-pipeline` 分支的目标、已实现内容、数据流 ABI、
真机执行步骤与已知风险。给真机 agent 直接照着跑。

---

## 1. 背景与目标

目标是在 Ascend 950 上打通一条 **MoE FFN 的 FP8 推理链路**：

```
topk 路由 → 路由 + 按专家重排到 masked slab（融合） → 激活 FP8 量化(1×128)
          → gated FP8 grouped GEMM (gate_up, N=2F)：epilogue 内联 silu(gate)·up
            与 1×128 量化，直接产出 fp8 + scale
          → FP8 grouped GEMM (down, N=H)
          → 加权反排列回 token 维
```

脚本保留 `--split-route-gather` 与 `--plain-gate-up` 两个开关，可退回未融合的
8-launch 链路做对照。

设计约束：

- 专家计算使用 **masked slab 布局** `[E, max_m, H]`：每个专家固定 `max_m` 行，
  `sizes[E]` 给出有效行数，GEMM 内部按 `ceil(count / tile_m)` 只算有效 tile。
- `down` 复用已有的 `AscendMoEGroupedGemmFP8Kernel`（不改）；
  `gate_up` 用新增的 `AscendMoEGroupedGemmFP8GatedKernel`，epilogue 特化。
- 两种 epilogue 是编译期特化，不是运行时分支：`down` 下游是 post-permute，要 bf16，
  所以保持 plain；`gate_up` 下游是另一个 FP8 GEMM，所以输出 fp8 + scale 最省。
  plain kernel 一行未改，融合只新增了一种特化。
- 权重量化在 host 侧用 torch 完成（离线数据，按 128×128 块），不进每 token 链路。

本分支**不接入 manifest / op 注册**：所有 kernel 由 call spec 直接构造并调用，
以减少对现有 op 层的侵入。后续要进 op 层时再补 manifest 条目。

---

## 2. 当前实现状态

| 阶段 | 实现 | 状态 |
|---|---|---|
| topk | `AscendFusedTopKKernel`（仓库已有） | 直接复用 |
| route + gather → slab | `AscendMoERouteGatherKernel`（本分支新增） | 真机验证通过 |
| 激活量化 | `quant_fp8_rows`（本分支新增，位于脚本内） | 真机验证通过 |
| gate_up GEMM + 激活 + 量化 | `AscendMoEGroupedGemmFP8GatedKernel`（本分支新增） | 真机验证通过 |
| down GEMM | `AscendMoEGroupedGemmFP8Kernel`（仓库已有） | 直接复用 |
| 反排列 | `AscendMoEPostPermuteKernel`（`post_permute_simd.py`，本分支移植） | 真机验证通过 |
| 端到端脚本 | `scripts/run_moe_fp8_pipeline.py` | 可直接跑 |

两个 GEMM 的 epilogue 在**同一个模块** `moe_grouped_gemm_fp8.py` 里，各是一个独立
`@T.prim_func`（`make_grouped_gemm_fp8` / `make_gated_grouped_gemm_fp8`），由
`MGroupedGemmFP8Call.activation` 在编译期选定，plain 类显式拒绝带 activation 的调用。
**没有共享循环体**：Ascend 的 auto-scheduler 是从 trace 出来的程序结构读调度决策的，
把 K 循环抽成 `@T.macro` 共享会让它看不到 op，实测 compute% 从 42% 掉到 7%
（gate_up 508→3049 us）。两个 builder 各自保留调优时的循环形态。

未融合的 `AscendMoEPermuteAlignKernel`（PR #44）、`AscendMoEGatherSlabKernel`、
`AscendSiluAndMulKernel` 与 plain GEMM 仍保留，由上面两个开关选中。

### 与 PR #44 的关系

本分支把 PR #44 的文件原样带入（未做 merge commit），**没有修改它的 ABI**：

- `sorted_token_ids`：按专家分组、每段 64 对齐、padding 填 `num_routes`
- `expert_ids`：每 64 行一个专家 id，尾部为 `-1`
- `num_tokens_post_pad`：padded 行总数

`counts` / `starts` / `inverse_indices` **不由 PR #44 产出**，全部在 gather kernel 里派生：

- `starts`（专家段起点）：在单调的 `expert_ids` 上二分；
- `counts`（有效行数）：由持有该专家最后一个 padded block 的 block 回扫哨兵得到；
- `inverse`（路由 → slab 行号）：`g * max_m + rank`，由 gather 的 SimdVF 散射写入。

原因：PR #44 的行号是「紧凑 packed 偏移」，而 slab 行号是 `g * max_m + rank`，
两者在 `max_m > 段长` 时不同，所以 inverse 必须由知道 `max_m` 的 gather 产出。

---

## 3. 数据流与形状 ABI

设 `T` 为 token 数、`K` 为 top-k、`E` 为专家数、`H` 为 hidden、`F` 为 ffn、
`routes = T * K`、`max_m` 为每个专家的 slab 容量。

| 阶段 | 输入 | 输出 |
|---|---|---|
| topk | `gating [T, E] f32` | `weights [T, K] f32`、`ids [T, K] i32` |
| route-gather | `ids [T,K] i32`、`hidden [T, H] bf16` | `slab [E, max_m, H] bf16`、`counts [E] i32`、`inverse [routes] i32`、`overflow [1] i32` |
| quant | `slab` | `a_q [E, max_m, H] fp8`、`a_scale [E, max_m, H/128] f32` |
| gate_up（gated） | `a_q, a_scale, w_gu_q [E,2F,H] fp8, w_gu_s [E,2F/128,H/128] f32, counts` | `act_q [E,max_m,F] fp8`、`act_s [E,max_m,F/128] f32` |
| down | `act_q, act_s, w_d_q [E,H,F] fp8, w_d_s [E,H/128,F/128] f32, counts` | `[E, max_m, H] bf16` |
| post-permute | `expert_out [E*max_m, H]`、`inverse`、`weights` | `[T, H] bf16` |

语义要点：

- slab 行 `g*max_m + rank` 与 `inverse[route]` 一一对应；
- `counts[g]` 之后的 slab 行是未定义值（GEMM 整 tile 写回），下游只通过 `inverse` 读有效行；
- `routed_scaling_factor` 只在 post-permute 里乘一次，topk 侧保持 1.0；
- `overflow=1` 表示某专家的 route 数超过 `max_m` 被截断，必须调大 `--max-m`。

---

## 4. 真机执行

### 4.1 环境

- 仓库：`TileOPs-ascend950`，分支 `feat/ascend-moe-fp8-pipeline`
- 依赖：`torch_npu`、可用的 `tilelang` ascend 后端（与仓库其它 ascend kernel 相同）
- 设备：Ascend 950

### 4.2 第一步：验证 PR #44 基线

```bash
pytest tests/test_ascend_moe_permute_align.py \
       tests/ops/test_moe_permute_align_ascend.py \
       --tileops-device=npu
```

这一步失败就不用往下跑，问题在路由 / grid sync。

### 4.3 第二步：最小形状冒烟（不比对参考）

```bash
python scripts/run_moe_fp8_pipeline.py \
    --tokens 8 --top-k 2 --experts 32 --hidden 512 --ffn 512 --max-m 64 \
    --skip-reference
```

期望输出依次为：

```
topk: (8, 2) (8, 2)
gather: (32, 64, 512) counts sum = 16
gate_up -> act_q: (32, 64, 512) act_s: (32, 64, 4)
out: (8, 512) torch.bfloat16
```

`counts sum` 必须等于 `T*K`；不等说明 route-gather 的 starts/counts 有误。
加 `--split-route-gather` 会多打印一行 `permute-align: post_pad = ...`。

### 4.4 第三步：带参考比对

```bash
python scripts/run_moe_fp8_pipeline.py \
    --tokens 8 --top-k 2 --experts 32 --hidden 512 --ffn 512 --max-m 64
```

参考实现是在脚本内用 fp32 torch 写的（相同路由下逐专家算 MLP 再加权），
判定 `allclose(rtol=2e-2, atol=2e-2*(1+sqrt(H/1024)))`。

### 4.5 第四步：放大到默认形状

```bash
python scripts/run_moe_fp8_pipeline.py
# T=64 K=8 E=32 H=7168 F=2048 max_m=128
```

### 4.6 性能测量

```bash
msprof op --kernel-name=main --launch-count 5 --warm-up 0 \
    --aic-metrics PipeUtilization \
    python scripts/run_moe_fp8_pipeline.py \
    --tokens 512 --experts 32 --hidden 4096 --ffn 1536 --max-m 256 --skip-reference
python summarize_moe_perf.py OPPROF_<stamp>
```

融合链路是 5 个 `main` launch（post-permute 的 prim_func 名为
`_simd_unpermute_main`，需单独用 `--kernel-name=_simd_unpermute_main` 补测）。
`summarize_moe_perf.py` 按 launch 数选阶段表：5 = 全融合，6 = 仅拆 route-gather，
8 = 全拆。GEMM 是 MixedKernel，搬运在 AIC 侧，必须读 `aic_mte2_ratio`。

### 官方基线 shape 实测（融合链路，5 launch）

shape 取自 manifest `FusedMoEFwdOp` 的 workload，`max_m` 取能容下最热专家的最小值
（`routes / E` 向上取整到 64 的倍数）。

| label | T | E | H | F | max_m | 总计 |
|---|---|---|---|---|---|---|
| qwen3-235b-decode | 512 | 128 | 4096 | 1536 | 64 | 2211.6 us |
| qwen3-235b-prefill | 4096 | 128 | 4096 | 1536 | 512 | 7965.0 us |
| ds-v3-grouped-decode | 512 | 256 | 7168 | 2048 | 64 | 9379.1 us |
| ds-v3-grouped-prefill | 4096 | 256 | 7168 | 2048 | 256 | 18635.2 us |

分阶段（us）：

| label | topk | route-gather | quant | gate_up | down |
|---|---|---|---|---|---|
| qwen3-decode | 5.0 | 65.9 | 346.0 | 1124.1 | 670.6 |
| qwen3-prefill | 19.9 | 298.8 | 2737.1 | 3337.8 | 1571.4 |
| ds-v3-decode | 6.6 | 86.8 | 1054.4 | 5110.1 | 3121.1 |
| ds-v3-prefill | 32.5 | 453.1 | 4227.3 | 8840.9 | 5081.3 |

**注意口径**：这些是全局专家数（128 / 256），即单卡不切 EP 的极端情况。
EP=8 部署时单 rank 只有 `E/8` 个本地专家，GEMM 的 tile 数同比减少。

### 融合的收益（同一 shape 前后对比）

在 T=512 E=32 H=4096 F=1536 max_m=256 上（这个组合不是官方基线，只用于对比）：

| 链路 | 总计 |
|---|---|
| 未融合（8 launch） | 1455.0 us |
| 仅融合 gate_up | 1184.1 us |
| 两项都融合（默认） | **1153.6 us** |

融合后 gate_up GEMM 513→507 us，route-gather 84→65 us（省掉 permute-align launch
和 slab 的中间表往返），并省掉 silu_and_mul 与第二次量化的两次 launch。

### 内存

脚本按最终 dtype 生成权重，`quantize_weights` 逐专家量化。E=256、H=7168、F=2048 时
`w_gate_up` bf16 占 14 GiB；若先建 fp32 临时张量则翻倍到 28 GiB，单卡放不下。

---

## 5. 形状约束（脚本会提前检查）

| 约束 | 原因 |
|---|---|
| `hidden % 512 == 0` | FP8 GEMM 每个 MMA 处理 128 宽 K block，L1 tile 跨 4 个 block |
| `hidden % 128 == 0`、`ffn % 128 == 0` | 量化 1×128 分组、GEMM 的 N/scale block |
| `2 * ffn % 128 == 0` | gate_up 的 N = 2F |
| `max_m % 64 == 0`（推荐 128） | GEMM 的 `tile_m ∈ {64, 128}` 且必须整除 slab |
| `experts % 8 == 0` 且 `8 <= experts <= 384` | permute-align 的 refusal |
| `experts % 32 == 0` | topk kernel 的 32-lane 分组 |
| `tokens * top_k % 8 == 0` | permute-align 的 route 数约束 |
| routes 不宜超过 `E * max_m` | 否则会触发 gather 的 overflow |

---

## 6. 已知风险与排查顺序

按风险从高到低，都是「只在真机上才会暴露」的点。

### 6.1 量化 kernel 的 bf16 扩宽（最高风险）

`quant_fp8_rows` 的数学配方照搬了
`tilelang/examples/ascend/example_per_token_cast_to_fp8.py`（fp32 输入），
但这里输入是 bf16，用 `vld(dist="UNPK_B16") + vcvt(part=0)` 扩宽
（与仓库 `kernels/ascend/elementwise/silu.py` 的写法一致）。

排查：

1. 单独打印 `q.float() * scale` 与原始 bf16 张量比较；
2. 若整体差常数因子 → 检查 `scale = amax/448` 与 `q = x/scale` 的方向；
3. 若数值顺序错乱 → 调整 `part` 与 `dist` 的组合（`part=0/1` 在
   `UNPK_B16` 下表示 64 元素分块，不是偶/奇 lane）。

### 6.2 route-gather 的行拷贝与 id 预取

融合后行拷贝（MTE2/MTE3）与 id 预取共用同一个 block 的标量循环，
`MTE3_MTE2` 保证行缓冲在上一行搬走后才被重填。tile 大小同时决定并行度：
64 route/tile 时 56 核都能分到活（512 routes → 8 tile/核），
256 route/tile 时只有 16 个 tile，多数核空转，实测慢 75%。

排查：先只跑 `--tokens 8 --top-k 2`，检查 `slab[inverse[route]] == hidden[route//K]`
逐行成立；不成立先看 `counts` 是否与 `torch.bincount(ids)` 一致。

### 6.3 PR #44 本身

若 4.2 的测试不过，先修 PR #44 的路由，不要继续调下游。

### 6.4 GEMM 的 padding 行

slab 里 `counts[g]` 之后的行没有初始化，量化后会得到无意义的 `scale`
（可能是 NaN/Inf），但只影响这些 padding 行的输出，且它们不会被 post-permute 读到。
如果真机上出现非有限值扩散（不应该），可以在 gather 里把 padding 行清零。

---

## 7. 本分支改动清单

```
新增  scripts/run_moe_fp8_pipeline.py                 端到端脚本 + 1×128 量化 kernel + torch 参考
新增  src/tileops/kernels/ascend/moe/gather_slab.py   gather to masked slab
新增  src/tileops/kernels/ascend/moe/route_gather.py  路由 + gather 融合（单 launch）
修改  src/tileops/kernels/ascend/moe/moe_grouped_gemm_fp8.py
                                                      同一模块内新增 gated prim_func
                                                      （epilogue 内联 silu·up + FP8 量化）
新增  src/tileops/kernels/ascend/moe/post_permute_simd.py  加权反排列（SimdVF，移植）
新增  docs/ascend950/moe-fp8-pipeline.md              本文件
修改  src/tileops/kernels/moe/call_spec.py            增加 GatherSlabCall / GatherSlabFwdInterface，
                                                      MGroupedGemmFP8Call 增加 activation
修改  src/tileops/kernels/ascend/moe/__init__.py      导出 post_permute
带入  src/tileops/kernels/ascend/moe/permute_align.py  PR #44
带入  src/tileops/csrc/ascend_moe_grid_sync.h          PR #44
带入  tests/test_ascend_moe_permute_align.py           PR #44
带入  tests/ops/test_moe_permute_align_ascend.py       PR #44
```

## 8. 明确未做

- manifest 条目 / `register_kernel_type`：route-gather、gated GEMM 与量化目前只能按脚本方式
  直接构造。gated GEMM 的输出签名是 (fp8, scale) 二元组，与 `MoEGroupedGemmFP8FwdOp`
  的单输出不同，进 op 层需要一条独立 manifest 条目；
- bias、EP dispatch、shared expert：均在主链路之外；
- 量化与 route-gather 融合：当前量化仍是独立 launch，多一次 `[E,max_m,H]` 的 bf16 往返；
- 量化用批量归约指令替代逐 128 列归约；
- 量化输出直接喂 GEMM 的 L1（跳过 GM 往返）。
