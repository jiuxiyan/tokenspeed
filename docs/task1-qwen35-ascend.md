# Task 1：Qwen3.5-0.8B 在 Ascend 910C 单 chip 上的计算适配

本实验在 [jiuxiyan/tokenspeed](https://github.com/jiuxiyan/tokenspeed) 的原始 Engine 上完成文本计算路径适配。代码基准为 `070dab022dc9879474b37fba2d95ab5d12f0534e`，工作分支为 `task1-qwen35-ascend`。实现、分层验证、完整推理和性能测量均使用同一运行路径。

最终实现已通过 251 项算子/配置回归及完整 224 步同精度模型验收，top-1 全部一致，A→B→A 的 32 步 logits 逐位相同。参考采用官方 HF 计算及显式 FP32 recurrent cache；此前默认 BF16 cache 的失败结果保留。四组性能矩阵、128-token 连续生成 HBM 和短/长 Prefill、Decode 追踪均已实测。

## 1. 范围、环境与可复现入口

实测范围是 **文本输入、BF16 权重/激活、FP32 GDN recurrent state、单 chip、TP=1、batch=1、greedy、eager**。模型主要计算、KV cache、卷积历史与 recurrent state 均位于 NPU；CPU 负责加载、分词、调度和少量长度元数据。视觉塔、MTP/推测解码、量化、图捕获、多 chip 和内存压力下抢占恢复未纳入验收。

| 项目 | 固定配置/实测版本 |
| --- | --- |
| 设备 | Ascend 910C，框架报告 Ascend910_9362；仅暴露逻辑 chip 5，即物理 NPU 2 / chip 1，worker 内编号 0 |
| 内存容量 | 设备物理 HBM 65536 MiB；框架报告可用总容量 62748 MiB，二者口径不同 |
| 系统 | openEuler 24.03 LTS-SP3，aarch64，Linux 5.10.0 |
| 驱动 / CANN | 25.5.2 / 9.0.0 |
| Python / torch / torch_npu | 3.11.16 / 2.9.0+cpu / 2.9.0.post2 |
| Triton-Ascend / Transformers / scheduler | 3.2.1 / 5.12.0 / 0.1.40 |
| 请求配置 | max_model_len=2048，max_total_tokens=16384，chunked_prefill_size=512，prefix granularity=128 |
| 缓存/执行 | 禁用 prefix cache、Host KVStore、图捕获和推测解码；retraction_snapshot_ratio=0.0 |
| 测量 | seed=20261022；性能每组预热 2 次、正式 5 次；正常测量无观测钩子、无 profiler |

`torch` 包的 `+cpu` 后缀不表示模型在 CPU 执行，NPU 后端由 torch_npu 提供。私人 venv 继承既有 aics 环境，未修改共享 nanoGPT。完整 freeze、约束、CANN 加载和安装步骤见 [复现说明](../examples/task1_qwen35/README.md)，该环境清单不等同于任意裸机器的一键安装保证。

使用官方 Qwen ModelScope 权重；主权重 revision 为 `05f056712f8c582c8708395156518d963b46fb29`，SHA256 为 `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`。tokenizer_config/chat_template 的独立 revision 为 `c62d4afcf135a7464a3e6743ee9454a6997291a1`。tokenizer.json、merges 和 vocab 随主权重 revision 固定。全部 12 个文件的长度、revision、SHA256 固定在 [model_manifest.json](../examples/task1_qwen35/model_manifest.json)，下载器校验后才使用。

在已配置环境的仓库根目录启动：

```bash
export TASK1_CHIP=5
source examples/task1_qwen35/env.sh
python -m examples.task1_qwen35.run_engine \
  --model "$HOME/task1/models/Qwen3.5-0.8B" \
  --mode chat --decode-steps 128 \
  --output "$HOME/task1/results/chat-reproduction"
```

复现者应根据实际分配选择 chip；每次使用新结果目录。运行清单记录启动参数、输入/输出 IDs、权重校验和实际 runtime/kernel/scheduler/实验代码逐文件 SHA256。源代码归档部署读取 `BASE_COMMIT` 并记录源码清单，不把无 `.git` 的目录误称为干净 checkout。

## 2. 计算路径与算子支持表

文本骨干有 24 层（18 个 GDN、6 个 full attention），hidden=1024、FFN=3584、vocab=248320，embedding 与 LM head 共享权重。下表中 `T` 是 packed token 数，`B` 是请求数，`P` 是状态池行数；除明确标注外，激活和权重为 BF16。

| 功能 | 主要输入 → 输出形状 | 精度/布局 | 来源与本次工作 | Ascend 验证 |
| --- | --- | --- | --- | --- |
| Embedding / LM head | `[T]` → `[T,1024]` → 必需位置的 `[N,248320]` | 整数 IDs，BF16 参数，共享 storage | 复用模型、loader、logits processor | 文本参数全覆盖、共享权重审计；整模型对照 |
| Linear/GEMM | `[T,1024]` → 各投影维度；MLP 中间 `[T,3584]` | BF16 矩阵乘法 | 复用 merged/column/row parallel linear 与 Ascend GEMM | 真实权重同输入逐层比较，merged gate/up 和 down projection 一致 |
| Gemma RMSNorm / residual | `[T,1024]` → 同形状 | residual 先 BF16 相加；RMS 与 `1+w` 为 FP32 | 公共 norm dispatch 新增/修正 NPU leaf；移除 NPU 路径的 CUDA stream 假设 | 空/单行/65 行、非零权重、残差和 loader |
| SwiGLU | 两个 `[T,3584]` → `[T,3584]` | SiLU 结果 BF16，再与 up 相乘 | 新增 Qwen NPU 专用公共接口；其他模型/平台保留原语义 | 25 项，含 CPU FP64 与 HF 同设备边界对照 |
| Full-attention Q/K norm | Q `[T,8,256]`，K `[T,2,256]` | FP32 norm/offset，BF16 输出 | 复用 prologue，修正 offset 中间精度 | packed stride、空输入、非零 offset |
| Partial RoPE | Q/K 上述形状，仅前 64 维旋转 | BF16 乘法与加减边界；余 192 维不变 | NPU leaf 针对 BF16 head256/rotary64 NeoX 保留 HF 边界 | 跨 64/128/512 位置、独立输出、不连续视图 |
| Causal GQA / KV | Q8/KV2/D256；prefill L 个 query，decode 1 个 | BF16 KV；score/scale/prob/value 分阶段 BF16，softmax FP32 | 几何匹配时用 NPU torch 完整 attention；其他几何保留 native FIA | varlen、分页随机映射、causal extend、污染尾页和 null-page padding |
| Attention gate | `[T,2048]` 与 `[T,8,256]` → `[T,2048]` | BF16 sigmoid 后相乘 | Qwen 专用公共 sigmoid gate | 非连续 head view、空输入；纳入 norm/gate 114 项 |
| GDN projection | `[T,1024]` → q/k/v/z 各 2048，a/b 各 16 | BF16；A_log/dt_bias 计算 FP32 | 复用 fused projection 与 checkpoint 映射 | loader、同输入真实权重、映射测试 |
| Depthwise causal conv | `[6144,T]` 或 `[B,6144]`，权重 `[6144,4]` | FP32 乘加→BF16→SiLU→BF16；历史 `[P,6144,3]` | 修改原 Triton kernel 的 NPU tile 和舍入边界 | 21 项，长度 1/2/3/4/5/63/64/65/128/129，20 个窗口用例各接 5 步 decode，另有抵消回归 |
| GDN Q/K normalize | `[1,T,16,128]` → 同形状 | 输入 dtype 下 L2 normalize，再转 FP32，eps=1e-6 | 新增 NPU leaf | 独立 CPU FP64，另记录跨设备 rsqrt 的 BF16 ULP 差异 |
| GDN gate | a/b `[T,16]` → g/beta | g 使用 FP32 exp/softplus，beta 先 BF16 sigmoid | 公共 gate dispatch；NPU 用 staged softplus | 同设备 HF 逐位比较、参数传递、形状边界 |
| GDN prefill / decode | q/k/v `[1,T,16,128]`；池 `[P,16,128,128]` | FP32 state，池 K-last `[P,H,V,K]`；内部 K-first `[B,H,K,V]` | 新增 64-token 分块 prefill、单步 decode；保留原 cache recipe | 37 项，CPU FP64、HF 同序、非方形、分段、多步、交换读写槽及 padding |
| Gated RMSNorm | `[T,2048]` 按 128 维 head 归一化 | FP32 norm→BF16→BF16 weight multiply→FP32 SiLU gate→BF16 | Qwen NPU 专用公共接口，通用融合接口保留原契约 | 纳入 norm/gate 114 项，HF 同设备逐位对照 |
| 请求生命周期 | 请求映射到 KV/conv/SSM pages | cache 统一分配、归还和复用 | 复用 QwenGDNRecipe、HybridMHATokenToKVPool 与 scheduler | 算子隔离 + 同一 Engine 的 A→B→A 全 logits 比较 |

对 Qwen 几何的 staged attention 使用完整因果矩阵，计算和临时空间随序列长度平方增长；它是本次 eager 正确性实现，不是 FlashAttention 性能实现。读取 host 长度和分页 gather 也有调度/复制成本。不能将该分支外推为所有 NPU 模型或长上下文的性能结论。

## 3. 关键模块：Gated DeltaNet 的完整执行路径

### 3.1 从模型到公共 kernel

`Qwen3_5ForCausalLM` → `Qwen3_5LinearDecoderLayer` → `Qwen3_5GatedDeltaNet` → `HybridLinearAttnBackend` → `GdnAttnBackend/MambaAttnBackend` → 公共 `tokenspeed_kernel.ops.attention.gdn` → Ascend 注册 → NPU leaf。runtime 不直接导入 NPU 包；状态由原 cache recipe 管理，模型没有另建跨请求全局状态。

输入归一化后，fused projection 产生 q/k/v/z/a/b。q/k/v 共 6144 通道，经 `[T,6144]`→`[6144,T]` 的布局转换做宽度 4 的 causal depthwise conv，再 SiLU。每个请求只读取自己的 3 个历史投影；保存的是卷积前值。结果拆成 `[1,T,16,128]`，a/b 不经过卷积。packed 切片可能只有末维连续，不能假定整个张量连续。

定义 `beta_t=sigmoid(b_t)`、`g_t=-exp(A_log)*softplus(a_t+dt_bias)`；q/k 在 BF16 下做 L2 normalize，再转 FP32。以下用 K-first 状态 `S∈R^(K×V)` 表示完整数学递推：

```text
d_t = exp(g_t)
S̄_t = d_t S_(t−1)
p_t = k_tᵀ S̄_t
Δv_t = beta_t (v_t − p_t)
S_t = S̄_t + k_t Δv_tᵀ
o_t = (q_t / sqrt(K))ᵀ S_t
```

这包含 decay、旧状态预测、delta correction 和完整 rank-one 更新。不能删除旧状态项、缩小维度或只算最后一个 token。

### 3.2 分块 Prefill 与布局

每个请求独立划分为 64-token blocks；尾块零 padding，返回时裁掉 padding。块内构造下三角 decay-weighted key interaction，通过 63 次行更新形成等价三角变换，再用矩阵乘法计算整块输出和最终状态。每个块都接收上一块最终状态；没有跳过 token 或依赖 CPU 计算主干。

该 helper 基于 Transformers 5.12.0 的 `torch_chunk_gated_delta_rule` 运算顺序适配，源文件置于 NPU 包的 `thirdparty/`，保留 Apache 2.0 原作者、来源、修改说明及许可；生产运行不依赖导入 Transformers helper。另以不同写法的 CPU FP64 完整转移矩阵递推验证数学，避免只有同源对照。

状态池契约仍是 `[P,H,V,K]`。进入算子时取本请求状态并转置为连续 `[B,H,K,V]`，按 K 维归约；返回时转回池布局。初态只读，空请求保留初态，packed 尾部 padding 输出为零且不推进状态。采用与参考相同的内部归约方向，减少 FP32 极小差异落在 BF16 舍入边界后逐层放大的现象。

### 3.3 Decode、隔离与同步

Decode 先 gather 所有 source rows，再进行任何目标写入，因此两个请求交换源/目标槽也不会提前污染读值。负 read index 表示零初态（即使 null row 有 stale NaN 也清除），负 write index 表示 padding、不写池；有效目标槽由 scheduler 保证独占。结果经过 per-head gated RMSNorm 和 output projection 返回模型层。

所有张量计算留在 NPU。Prefill 的 `cu_seqlens` 被复制到 CPU 驱动请求循环；Decode 的布尔索引及 K-first 连续化也属于 eager 成本。这些操作不能声称支持静态图捕获。配置阶段对非 eager、非 FP32 SSM 或推测解码明确报错。

## 4. 适配障碍及本组修改

直接复用：模型层结构、权重分片映射、embedding/GEMM、混合 attention 路由、统一 cache recipe、scheduler、Engine、logits 和 greedy sampling。新增/修改集中在公共 kernel 边界、NPU leaf、必要的模型接入及实验工具。

1. **缺少 Ascend GDN 路径。** 新增完整 prefill/decode、gate 与状态索引契约；最初逐 token eager 实现能运行，最终改用同数学的 64-token block，以改善运算顺序一致性并实测其性能。
2. **Triton 卷积 unified buffer 超限。** 原 `64×256` prefill tile 对此 Ascend 编译失败；NPU 通道 tile 改为 64，仍计算全部 6144 通道，其他设备不变。
3. **“FP32 accumulator”掩盖 BF16 乘积先舍入。** 将 NPU conv 的输入和权重显式升 FP32 后乘加，在 SiLU 前形成 BF16 输出。回归用例 `(129/128)^2−130/128=1/16384` 能捕捉旧实现错误输出 0。
4. **CANN uncached TND 不支持 head_dim=256。** 先用合法完整 BSH 路径打通；进一步数值定位发现 native FIA/RoPE 与 HF staged BF16 的边界不同，最终对匹配几何使用完整 staged NPU attention/RoPE，其他几何继续 native。
5. **中间 BF16 舍入影响 24 层结果。** 修正 Gemma offset、gated RMSNorm、sigmoid gate、SwiGLU 和 GDN gate。旧 NPU MLP 实际调用 `SiluAndMul.forward_native`，其 FP32 融合 SiLU×up 缺少 BF16 中间结果；根据真实调用栈定位，而非根据可用 kernel 名猜测。
6. **参考缓存精度不一致。** 官方递推内部 FP32 不等于缓存保存 FP32，详见下一节。早期失败保留，明确修订参考缓存策略，阈值不变。

最大障碍是混合 attention 下计算边界、卷积历史、recurrent layout 和缓存生命周期同时成立。单独 GEMM 或 attention 能执行不足以证明模型适配完成。`max_total_tokens=4096` 曾无法满足 hybrid cache 的 LCM 状态块准入；本次固定 16384 和零抢占快照比例，不能把该值解释为纯 Transformer 的可用 KV token 数。

## 5. 分层正确性、精度口径与误差调查

### 5.1 固定判据与独立参考

完整判据见 [数值协议](task1-numerical-protocol.md)，在读取端到端结果前固定，后续未放宽：每步全词表 logits finite，max absolute ≤0.5，relative L2 ≤0.02，cosine ≥0.999；全步骤 top-1 agreement ≥95%。BF16 舍入和 24 层积累是工程阈值的依据，但不是理论误差上界；必须保留失败与逐步指标。

算子层以 CPU FP64/FP32 数学表达式和同设备官方边界作互补对照。GDN 对相同量化 q/k，FP32 state 使用 `atol=2e-6, rtol=2e-5`，FP32 output 使用 `2e-6/1e-4`，BF16 output 使用 `1e-3/8e-3`。块状与任意分段的浮点归约顺序不同，因此用既定数值阈值比较；只读初态、无关槽、padding、复制的卷积历史以及相同请求复跑仍要求精确一致。

BF16 CPU/NPU rsqrt 的个别值可能相差 1 ULP；归一化单独对照并保留完整输出验证，不能放宽 FP32 state 阈值掩盖该差异。Attention/RoPE 对独立 CPU FP32 使用 BF16 `atol=rtol=0.02`，同时增加 staged HF 同设备逐位测试。norm/gate BF16 对 CPU double 为 `0.01/0.01`，FP32 为 `2e-5/2e-5`，另有严格的 offset 和中间舍入回归。

### 5.2 正式参考缓存策略

参考固定为 [Transformers 5.12.0 Qwen3.5](https://github.com/huggingface/transformers/blob/v5.12.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)，eager attention 和 portable torch conv/GDN，不调用 TokenSpeed kernel。两个进程顺序使用同一 NPU、同权重与 BF16 主干；HF 对 Engine 实际生成的 token 轨迹进行 teacher forcing，自己维护 cache。

实测默认 HF `LinearAttentionLayer` 根据 BF16 卷积状态分配 recurrent buffer，每次 `copy_` 将 FP32 返回值舍入 BF16。真实模型 18 层×2 步的 36 次更新均确认此行为，最大状态舍入 abs=0.0195388794、relative L2=0.00181724。本实验 TokenSpeed 状态池从始至终为 FP32，早期默认 HF 对照不属于同状态精度比较。

最终主要验收采用 **“官方 HF 计算 + 显式 FP32 recurrent-cache policy”**：通过独立 cache 实例，在首次 Prefill 写入前就分配 FP32 recurrent buffer，conv/KV 仍 BF16；不改模型公式，不全局替换 HF 算法。CLI 的 `--reference-state-dtype` 没有默认值，每步核查 24 层 cache dtype。另可显式选择 `model` 保存默认官方缓存兼容性结果。两者不能混称。

### 5.3 已通过的算子与工具回归

| 套件 | 通过数 | 覆盖 |
| --- | ---: | --- |
| GDN | 37 | FP64 状态/输出、HF 同序、64 边界、非对称维度、分段、多步、交换槽、空/padding、gate 参数 |
| Norm/gate | 114 | Gemma、Q/K offset、residual、gated norm、sigmoid gate、真实 runtime loader |
| Causal conv | 21 | 真实 6144 通道、新/续算历史、20 个窗口用例各 5 步 decode，抵消回归另测 1 步 |
| SwiGLU | 25 | BF16/FP16/FP32、3584 通道、不连续/空输入、输入只读、HF 边界 |
| Qwen attention/RoPE + 原通用 MHA | 28+4 | causal GQA、随机页布局、无效尾页、null page、partial RoPE、通用 native/graph 回归 |
| Config/mapping | 9+13 | 不支持配置提前失败、真实参数与 state recipe 映射 |
| 合计 | **251** | 合并运行 16.22 s；该时间包含测试开销，不是推理性能 |

实验工具另有 23 项 CPU 测试、8 项离线下载测试和 2 项真实 tokenizer/msgpack 对话输入测试（共 33 项），覆盖 G−1 分母、首帧合并、完整轨迹、非有限值失败、共享权重审计、实际 HF cache 策略、profiler 事件统计、下载中断与 SHA256/长度校验失败时不覆盖旧文件。所有用例实际通过；最终格式化与 chat 修复后又统一执行 33 项 CPU 测试，6.558 s 全通过。不将未执行的其他平台或全仓库测试算入覆盖。

### 5.4 权重、状态与逐层证据

权重审计覆盖 488 个 checkpoint tensors、230 个实际加载名称、含共享别名共 231 个文本参数，全部在 NPU，无缺失文本参数。视觉/MTP/可重建 buffer 单独记录；共享 embedding/LM head storage 一致。

原始 Engine 固定输入长度 1/16/128/512，各生成 32 tokens；同进程再执行 A1(32)→B(65)→A2(同 A1)，共 224 步（每请求 1 次 Prefill+31 次 Decode）。逐步比较全词表 248320 个 logits；确认实际 IDs、位置、阶段与 greedy argmax，prefix cache 关闭。A1/A2 的全部 logits 要求逐位相同，不能只看生成文本。

第五版同输入逐层诊断覆盖 24 层的 Prefill/首个 Decode：48 组 norm/MLP、36 组 GDN projection/conv/gated norm/out projection、12 组 full attention/RoPE 均逐位一致。18 个 GDN 层的 Prefill final state 到 Decode initial state 也逐位相同。剩余逐 token scan 与 HF chunk 的 FP32 final state 相对差约 1–2×10⁻⁷，Prefill 每层至多 10/32768 个 BF16 core 元素不同，但可逐层放大。由此改用同序 block 和 K-first reduction；并继续保留独立 FP64 数学验证。

### 5.5 历史结果与最终验收

前五轮使用默认 BF16 recurrent cache 的官方参考：

| 轮次 | 主要变更 | 最大 abs | 最大 relative L2 | top-1 | 原阈值结论 |
| --- | --- | ---: | ---: | ---: | --- |
| 01 | 初始完整实现 | 0.453125 | 0.0293379 | 224/224 | 未通过 |
| 02 | norm/gate/conv 边界 | 0.484375 | 0.0301218 | 224/224 | 未通过 |
| 03 | SwiGLU 边界 | 0.4375 | 0.0299732 | 224/224 | 未通过 |
| 04 | staged RoPE | 0.3837891 | 0.0285627 | 224/224 | 未通过 |
| 05 | staged full attention | 0.421875 | 0.0288390 | 224/224 | 未通过 |

改为显式 FP32 recurrent cache 后，第 03/05 版仍分别出现最大 relative L2 0.0307558/0.0356438，仍未通过。这说明缓存精度勘误确实必要，但不是所有偏差的唯一原因；不能用它解释掉剩余失败。全部历史结果、逐步指标和来源哈希保留在 [实验数据目录](../examples/task1_qwen35/results/)。

最终第 06 版分块实现通过完整协议，全部 logits 有限，224/224 top-1 相同，A1/A2 的 32 步全词表 logits 最大差均为 0：

| 输入 case | 输出 tokens | 最大 abs | 最大 relative L2 | 最小 cosine |
| --- | ---: | ---: | ---: | ---: |
| length_1 | 32 | 0 | 0 | 0.99997491 |
| length_16 | 32 | 0 | 0 | 0.99997354 |
| length_128 | 32 | 0.328125 | 0.01957169 | 0.99980897 |
| length_512 | 32 | 0.28125 | 0.01789368 | 0.99984729 |
| A1 / A2（各 32 输入） | 各 32 | 0 | 0 | 0.99997497 |
| B（65 输入） | 32 | 0.1875 | 0.01658129 | 0.99984896 |

所有 case 均满足原门槛；长输入最大 relative L2 距 0.02 较近，结论仅覆盖这些固定输入/配置，不宣称所有上下文逐位一致。cosine 由 NPU FP32 归约计算，即使 logits 完全相同，数值也可能略低于 1；abs/relative L2 和隔离检查同时保留。

最终实现另跑官方默认 BF16 recurrent cache 兼容性对照：仍为 224/224 top-1 一致，但最大 abs=0.481330872、relative L2=0.03434195、最小 cosine=0.99945563，未通过同一 2% L2 门槛。该轮完整跑了 224 步，但不属于 FP32-state 主验收范围；它的失败不会被主验收通过覆盖。适配保留 FP32 状态的选择和参考差异必须一并说明。

四个自然对话样例（中文、英文、Python 加法、整数算术）分别生成 101/40/13/19 tokens，均在 EOS 正常结束。Python 输出 `return a + b`，算术结果为 110；中文对 Prefill 的解释存在事实错误，英文只回答一句，算术也未遵循“只回答整数”。这些原始文本仅作为 Engine 连续生成 smoke，不能当作语义或指令遵循评测通过。`chat02` 修复了 Transformers 5.12 模板默认返回 BatchEncoding 的接口变化：显式返回并校验 `list[int]`，两项真实 tokenizer/MessagePack 回归通过；固定 token 的精度和性能路径不受此变更影响。样例未预热，其耗时不进入性能表。

## 6. 可比较的性能与 HBM 基线

输入 32/512 × 输出 32/128，构成四组；每组预热 2 次、正式 5 次。保存精确 token IDs 和 SHA256，`ignore_eos=True` 固定工作量。输出 G tokens 包括 Prefill 首 token，因此 Decode 分母为 G−1。加载、预先分词、首次编译及预热均不进入稳态表。

- TTFT 是 frontend 调用到第一个 token，包含 IPC、调度、H2D、模型和采样/输出；首帧必须恰好 1 token。
- TPOT 是首末 token 交付时间差除以 G−1。前端同步不能同步另一个 worker，因此不使用前端伪同步。
- 模型时间在另一个 diagnostic 运行中，由真实 worker 对每次 `ModelRunner.forward` 前后 NPU synchronize 得到；它会改变 overlap，不替代正常性能。
- HBM 的 allocated/reserved/peak 来自实际 worker；每次 forward reset peak，包括驻留权重/cache，但不覆盖此前调度/分配临时峰值。设备 `npu-smi` 快照另外记录，不能称为连续峰值。

最终正常性能数据如下，单位 ms，表示中位数 [最小值, 最大值]，每格 n=5：

| 输入 tokens | 输出 tokens | TTFT | Decode TPOT | 总耗时中位数 |
| ---: | ---: | --- | --- | ---: |
| 32 | 32 | 275.762 [273.747,279.110] | 72.005 [71.365,72.309] | 2507.936 |
| 32 | 128 | 276.334 [257.893,288.799] | 71.714 [65.873,73.704] | 9365.621 |
| 512 | 32 | 540.531 [520.105,554.320] | 73.610 [66.897,74.317] | 2836.252 |
| 512 | 128 | 529.659 [495.111,551.120] | 72.327 [65.588,74.966] | 9718.655 |

Engine 启动为 12.060 s，单独记录，不进入表格。测量部署的实际源码清单 SHA256 为 `30e25f78dd735593d8dde8cae0df67bae0e258ae529e4c17a87ea6e55ca7ab77`。最终仓库的格式化改动及后续仅 chat 输入接口修复另有源码对应记录，不重写实测清单。

无 profiler、每步前后同步的独立诊断使用输出 **128 tokens**，包括 1 次 Prefill 和 127 次 Decode：

| 输入 | 阶段 | 次数 | 模型 forward 中位 ms | forward 前 allocated GiB | forward peak allocated GiB | peak reserved GiB |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 32 | Prefill | 1 | 178.926 | 1.915 | 1.939 | 2.266 |
| 32 | Decode | 127 | 62.752 | 1.915 | 1.919 | 2.266 |
| 512 | Prefill | 1 | 430.388 | 1.915 | 1.998 | 2.266 |
| 512 | Decode | 127 | 61.025 | 1.915 | 1.922 | 2.266 |

GiB=2³⁰ bytes。当前观测范围最大 allocated 为 1.998 GiB，reserved 为 2.266 GiB；这是逐 forward 范围，**不是完整进程生命周期或设备物理 HBM 峰值**。框架之外的上下文/驱动占用不在 allocated 中。同步诊断与正常交付 TPOT 口径不同，不能把两者之差精确归因给某一个组件。

选定 chip 的 `npu-smi` HBM 快照在启动前/加载后/关闭前分别为 2933/5138/5383，工具原单位为 MB，总容量 65536 MB。它们包含设备既有占用与框架外开销，非逐进程归属、非连续峰值；不将 5383 MB 与 allocator 的 1.998 GiB 混为同一指标。

## 7. 瓶颈证据、优化结果与限制

性能追踪各覆盖短/长输入的第一个 Prefill 与 Decode，额外开销显著，不能填入性能主表。解析实际 `kernel_details.csv`、`operator_details.csv` 和 host→device flow；报告事件数、设备时间段并集及导出 operator self/total，后两者不能相加冒充总耗时。

Profiler Level 0 不提供 AIC 硬件利用率；采集结束时出现 RECORD stop 和少量 incomplete memory record 提示，因此 trace 只用于已导出算子事件分析，不作为 HBM 或硬件占用证据。无 profiler 的 worker allocator 诊断承担内存测量。设备事件时长并集/跨度反映事件空隙，不等同于硬件利用率，也不能单凭间隙断言访存瓶颈。

前后实现使用相同四组配置、相同硬件和无 profiler 的正常计时：

| 输入/输出 | 逐 token scan TTFT ms | 最终 block TTFT ms | 逐 token scan TPOT ms | 最终 TPOT ms |
| --- | ---: | ---: | ---: | ---: |
| 32/32 | 225.213 | 275.762 | 63.322 | 72.005 |
| 32/128 | 223.674 | 276.334 | 63.530 | 71.714 |
| 512/32 | 1682.278 | 540.531 | 60.325 | 73.610 |
| 512/128 | 1688.992 | 529.659 | 60.186 | 72.327 |

长输入 TTFT 降低约 68%（约 3.1–3.2 倍加速），短输入 TTFT 与 Decode 变慢。分块并非所有形状都更快：短请求也要处理一个 64-token 块及其三角行更新，K-first 连续化增加布局操作。新版本优先确保完整数值验收；不能把旧版未通过主精度协议的性能当作可替代最终正确实现。两轮位于共享服务器的不同时段，host 调度有抖动；仅据五次测量不能将全部回退精确归因给单一算子。

以下统一使用 host→device flow 关联的设备事件数，包含相应拷贝事件，不与 kernel CSV 行数混用：

| 单个 forward | 旧设备事件数 | 新设备事件数 | 旧设备事件时间并集 ms | 新设备事件时间并集 ms |
| --- | ---: | ---: | ---: | ---: |
| 32 Prefill | 8746 | 10402 | 64.225 | 72.360 |
| 512 Prefill | 103780 | 14140 | 785.461 | 334.861 |
| 32 首次 Decode | 2295 | 2331 | 12.791 | 13.246 |
| 512 首次 Decode | 2295 | 2331 | 13.243 | 13.375 |

旧版长 Prefill 的 `aclnnMul/ReduceSum/Add` 次数为 46376/18468/9282；与短 Prefill 的差恰为 18 个 GDN 层×480 个新增 token×每步 5/2/1 次操作，直接表明逐 token eager 递推产生大量小算子。新版本长 Prefill 的 Mul/ReduceSum 降为 2096/1170，关联事件总数减少 86.4%，与正常 TTFT 改善方向一致。

**当前长 Prefill 最明显的设备算子热点是 causal convolution。** 新 trace 的 18 次 `_causal_conv1d_fwd_kernel` 合计 232.023 ms，约占选中设备事件并集 69.3%；随后 `aclnnInplaceCopy` 4263 次/33.406 ms，Mul 20.453 ms、ReduceSum 11.553 ms、Matmul 11.038 ms。下一步最值得尝试保持全部通道、FP32 乘加与 BF16 边界的卷积专用分块/融合，再减少 GDN 三角更新的 clone/copy。该结论是事件耗时排序，未取得带宽或 AIC 计数，不能进一步断言卷积受 HBM 带宽限制。

**Decode 的重点是小算子调度、状态/分页索引和布局转换，而非仅优化大 GEMM。** 长输入首个 Decode 有 2331 个关联事件，卷积 update 2.948 ms、索引 2.347 ms、Matmul 1.731 ms；trace 的选中设备事件并集 13.375 ms，而该 profiled forward 为 133.887 ms。大量事件间隙与当前 Python eager/同步路径一致，但 profiler 明显放大主机开销，不能用两者差值量化正常 TPOT 的主机占比。结合无 profiler 的同步模型时间 61.025 ms 与 frontend TPOT 72.327 ms，后续应优先减少每步小 kernel 启动和状态布局往返，并继续独立测量调度/传输成本。

上述归因使用已经导出的实际事件；四窗未匹配 flow 为 0，仅说明选中关联自洽，不证明 profiler 捕获了全部硬件活动。原始计数、operator 表与采集警告见公开 results。

## 8. 交付与复现索引

- [运行、安装、测试和实验命令](../examples/task1_qwen35/README.md)
- [原始数值阈值、失败历史和缓存精度勘误](task1-numerical-protocol.md)
- [脱敏机器可读结果](../examples/task1_qwen35/results/)
- [GDN 算子契约与算法](../tokenspeed-kernel/python/tokenspeed_kernel/ops/attention/gdn/README.md)
- [NPU attention 的精度与支持边界](../tokenspeed-kernel-npu/python/tokenspeed_kernel_npu/ops/README.md)

原始日志、JUnit、运行清单和诊断 JSON 保存在实验工作区；大型全词表 logits/逐层张量保留于远端实验目录，通过逐文件 SHA256 对应公开摘要。公共仓库不包含私人服务器账号、主机地址、绝对私有路径或其他用户进程快照。最终执行仓库要求的 `pre-commit run --all-files`，代码和结果保存在任务分支；具体提交号可通过该分支 Git 历史查看。
