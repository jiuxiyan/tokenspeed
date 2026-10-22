# Task 1 数值验收协议（端到端测量前固定）

2026-10-22 制定，适用于 BF16 文本主干、FP32 GDN recurrent state、greedy、固定官方权重。

## 分层判据

1. 卷积输出：相同 BF16 输入和权重，CPU FP32 累加再量化为参考；`atol=0.01, rtol=0.01`。滚动卷积状态是原始投影的拷贝，要求逐元素精确一致。
2. GDN 递推：将相同已归一化、已量化 Q/K 送入独立 CPU FP64 完整转移矩阵参考，与 NPU FP32 完整递推（初版逐 token、最终版分块）比较；状态使用测试中预先声明的 `atol=2e-6, rtol=2e-5`（具体形状以测试文件为准）。归一化单独对照，不能把前置 BF16 舍入差异误判成状态更新错误。
3. BF16 CPU/NPU rsqrt 的可见结果可能相差一个 ULP。必须独立记录该差异、验证归一化误差界限，再进行递推对照；不能通过放宽 FP32 状态阈值掩盖它。
4. Norm/gate：按原 TokenSpeed 融合计算约定对照独立 CPU 高精度表达式；采用各算子测试中的 BF16/FP32 阈值。HF gated norm 的中间舍入差异另由整模型对照量化。
5. 请求隔离：相同进程执行 A→B→A，要求两次 A 的输出 token ID 和 teacher-forced logits 完全相同；单独覆盖算子池读写索引交换、padding 和分段续算。

## 端到端主要判据

参考为固定 Transformers 5.12.0 的官方 eager Qwen3.5，实现与 TokenSpeed 相互独立；使用同一份已校验权重、同样 BF16 精度和同样输入/后续 token IDs。主要对照可在单 NPU 上顺序运行，CPU FP64 算子参考承担跨设备数学验证。

- 每个 forward 比较全词表 logits，要求 finite。
- 最大绝对误差 ≤ **0.5**。
- 相对 L2 误差 ≤ **0.02**。
- Cosine similarity ≥ **0.999**。
- 全部 teacher-forced 步骤 top-1 一致率 ≥ **95%**；记录每个不一致位置及参考 top-1/top-2 margin。

依据：BF16 单次相对舍入约 0.39%，24 层不同融合/归约顺序会积累误差；0.5 的绝对界限约对应较大幅值 logits 的少量 BF16 ULP，相对 L2 与 cosine 同时约束整体分布。该工程验收并非严格浮点误差上界，因此保留全部原始指标，并逐项调查失败；不得看见结果后默默扩大阈值。自由生成文本相同仅为补充证据。

多长度主测试为 1/16/128/512 token 输入与每个请求 32 个输出（Prefill 产生第一个输出，其余为 Decode）；初始冒烟允许更短输出，但不计作最终验收。

## 实现修复说明（不改变上述端到端门槛）

首轮完整对照保存为 `hf-reference01-complete.json`，224 步 top-1 全部一致，
但相对 L2 最大 0.0293379，按本协议判定失败。随后修复 NPU 的计算边界：

- 卷积显式 FP32 乘法及累加，输出先量化 BF16，再进行 SiLU 并存回 BF16。
  原编译结果的乘积先以 BF16 舍入；增加精确抵消用例，不能仅靠宽松随机误差检测。
- Qwen GDN 的 gated norm 保留官方 normalized activation、weight multiply 的
  BF16 中间边界；full attention gate 使用 BF16 sigmoid 后再乘输出。
- 通用 fused norm/gate 与其他平台保留既有行为，Qwen NPU 显式选择上述接口。

独立 Transformers 的模型算法、权重加载和上面四项 logits 门槛保持不变。
checkpoint 中 A_log 虽以 FP32 保存，全部 288 个值均可被 BF16 精确表示；
HF 加载为 BF16 与 TokenSpeed 保留 FP32 不改变这些权重数值。


第二轮在上述修复后仍为 224/224 top-1 一致，但 relative L2 最大 0.0301218，判定失败。真实权重逐层诊断确认实际 NPU `SiluAndMul.forward_native` 缺少 SiLU 的 BF16 中间结果，新增 Qwen 专用公共接口修复并通过 25 项测试；第三轮最大 relative L2 为 0.0299732，仍判失败。

原生 RoPE 的 FP32 旋转末尾舍入与官方 BF16 两次乘法后加减不同。仅对 BF16 head256/rotary64 NeoX 几何保留官方边界，4 项官方同设备逐位对照与原 11 项 attention/RoPE 数学测试均通过。第四轮最大 relative L2 为 0.0285627，仍判失败。四轮均保留原阈值，未改独立参考模型或权重；后续继续对齐 full attention 的中间舍入。

## 状态精度勘误与第二版对照口径（读取新对照结果之前声明）

第五轮进一步对齐 attention 的 BF16 中间边界后仍未通过，最大 relative L2 为 0.0288390。排查发现，第一版将“官方递推函数内部使用 FP32”误当成了“官方 cache 保存 FP32”，两者并不等价。

Transformers 5.12.0 的 `LinearAttentionLayer.lazy_initialization` 先从卷积状态取得 BF16 `self.dtype`，随后以这个 dtype 分配 recurrent buffer；`update_recurrent_state.copy_` 因而在每次 forward 后将 FP32 递推结果舍入为 BF16。TokenSpeed 本实验的状态池始终为 FP32。这会让后续 Decode 比较不同精度的历史状态；前五轮原始官方参考及其失败结论全部保留，不能改称通过。

第二版增加显式、无默认值的参考状态精度选择：

- `model`：未经改变的官方缓存策略，实际 recurrent cache 为 BF16，继续作为 stock HF 兼容性对照。
- `float32`：官方 HF 模型计算与递推函数保持原样，仅使用独立 cache 实例将 recurrent buffer 从第一次 Prefill 写入前就分配为 FP32；卷积状态与普通 KV 保持 BF16。不先丢失精度再升回 FP32，不调用任何 TokenSpeed kernel，不全局替换 HF 算法。

第二版主要数值验收采用后者，名称必须写成 **“官方 HF 计算 + 显式 FP32 recurrent-cache policy”**。所有输入、权重、生成配置和原有数值阈值保持不变；另记录实际 cache dtype，分别报告两种策略的结果。第二版通过不能用于覆盖或宣称第一版默认官方缓存对照通过。

## 最终分块实现与同精度验收结果

使用显式 FP32 recurrent-cache policy 重新比较旧第 03/05 版，最大 relative L2 仍为 0.0307558/0.0356438，均未通过。缓存精度勘误并未自动解决模型误差；两份失败证据与默认缓存五轮结果一并保留。

同输入逐层诊断把剩余早期差异定位到 FP32 GDN scan/chunk 的归约顺序及 prefill gate 的 softplus 运算边界；Prefill→Decode 的全部 18 层 FP32 state 交接逐位相同。最终实现使用完整 64-token chunk 和内部 K-first reduction，池布局与 dtype 不变。独立 CPU FP64 状态/output 阈值保持原值。由于分块与任意分段的归约顺序不同，二者按上述数值阈值比较，不要求浮点逐位相同；初态只读、无关状态槽、padding 与相同请求复跑仍要求精确一致。

第 06 版完成 1/16/128/512 输入各 32 输出及 A1→B→A2，共 224 步：max absolute=0.328125、max relative L2=0.0195716899、min cosine=0.9998089671、top-1=224/224，所有值有限，完整协议通过。A1/A2 每步全词表 logits 均精确相同。输入长度 1/16 和两次 A 的全部 logits 还与同精度 HF 参考逐位相同；65/128/512 输入仍有可测差异，未宣称所有形状 exact。最终结果名为 `hf-reference06-fp32-state.json`；对应公开摘要和逐步 CSV 见实验 results 目录。

最终第 06 版另做 stock BF16-cache 兼容性比较：224 步完整执行、top-1=224/224、max absolute=0.481330872、max relative L2=0.03434195、min cosine=0.99945563，按原数值门槛未通过。`protocol_scope_complete=false` 表示不属于要求 FP32 state 的主协议，不表示缺少步骤；该兼容性失败单独公开。
