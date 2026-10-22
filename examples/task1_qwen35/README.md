# Qwen3.5-0.8B 单 chip 实验

这些入口调用 TokenSpeed 原有 `Engine`、权重加载、调度和 cache 生命周期。
不提供另一个模型推理实现。文本路径、BF16 主干、FP32 SSM 状态、TP=1、
batch=1、greedy、eager、禁用 prefix cache/Host KVStore/推测解码。
单请求基线显式设置 `retraction_snapshot_ratio=0.0`：不预留请求抢占恢复快照。
这个小型 cache 池默认按 10% 划分时可能不足一个 LCM block，导致启动被拒绝。
本实验不测试内存压力下的请求抢占/恢复；需要该能力时应增加 cache 容量并另测。
默认 `max_total_tokens=16384`。Qwen3.5 hybrid cache 同时容纳普通 KV 和两个
recurrent-state group，每个活动请求的每组状态需要输入/输出两个 block。
因此这个容量参数不能直接解释成普通 Transformer 的“可用 KV token 数”。
曾用 4096 得到仅 4 个 LCM 父块，连一个请求所需的状态加 KV 都无法同时容纳，
引擎报告 `max_single_request_tokens=0`；16384 用于当前单请求基线。

本次实测在服务器已有 `/opt/miniconda3/envs/aics` 的 Python 3.11 上创建私人
venv，使用 `--system-site-packages` 继承基础环境。以下步骤在仓库根目录的
bash 中执行，用于同一基础环境上的首次复现：

```bash
/opt/miniconda3/envs/aics/bin/python3.11 -m venv --system-site-packages "$HOME/task1/.venv"
export TASK1_CHIP=5
export TASK1_VENV="$HOME/task1/.venv"
export TASK1_CANN_ROOT=/usr/local/Ascend/cann
source examples/task1_qwen35/env.sh
python -m pip install \
  -c examples/task1_qwen35/constraints.txt \
  -r examples/task1_qwen35/requirements-verified.txt
mkdir -p "$HOME/task1/logs" "$HOME/task1/results"
```

`env.sh` 会先 `source "$TASK1_CANN_ROOT/set_env.sh"`，再激活私人 venv，
并设置 runtime/kernel 源码搜索路径及单 chip 可见性。`TASK1_CHIP=5` 是本次
实测选择；其他运行须按实际分配指定一个 chip。已有 venv 的后续 shell 只需
设置这些变量并 source `env.sh`，无需重复创建环境或安装依赖。

[`requirements-verified.txt`](requirements-verified.txt) 是实测环境完整的
`pip freeze`，包括从 aics 继承的包；它记录已验证组合，**不保证在任意机器的
裸 Python 环境直接安装即可复现**。本次 `torch==2.9.0+cpu` 由已有 aics 环境
继承，配合 torch_npu 在 NPU 执行。不同机器需要先准备兼容的驱动、CANN、
Python 与该 PyTorch 构建，再按清单核对；CANN 和驱动不由 pip 清单安装。
安装时始终同时使用 [`constraints.txt`](constraints.txt)，避免解析依赖时
升级已选 PyTorch。上述命令是复现说明，不会修改共享 aics 环境。

推理脚本不下载权重；`--model` 必须是已固定 revision 的本地 checkpoint。
首次准备可使用固定清单下载器：

```bash
python -m examples.task1_qwen35.download_model \
  --destination "$HOME/task1/models/Qwen3.5-0.8B"
```

下载器使用同目录 `model_manifest.json` 中记录的官方 Qwen ModelScope 文件路径、
revision、字节数与 SHA256。已有文件先校验，下载后再次校验并原子替换；失败时
保留 `.partial`，不使用未经校验的文件。可用 `--manifest` 显式选择其他清单。
每次运行使用新输出目录，避免覆盖实验。
离线入口不启动 SMG gateway；仍需要当前仓库的 Python runtime、
`tokenspeed-scheduler`、`torch`/`torch_npu`、Transformers 和 kernel 依赖。
先通过环境/算子测试，再运行模型。

`env.sh` 供 bash source，要求显式指定单个 `TASK1_CHIP`，可配置
`TASK1_VENV`、`TASK1_CANN_ROOT`、`TASK1_REPO_ROOT`。它不安装或修改共享软件。
当前 fork 的 scheduler 版本为 0.1.40；不要直接安装上游较新版本。
后续安装任何依赖都应附加兼容性约束，防止依赖解析升级 torch：

```bash
python -m pip install -c examples/task1_qwen35/constraints.txt <所需依赖>
```

约束固定 `torch==2.9.0+cpu`、`torch-npu==2.9.0.post2`、
`transformers==5.12.0`、`triton-ascend==3.2.1`、
`tokenspeed-scheduler==0.1.40` 和 `apache-tvm-ffi==0.1.13.post3`。
`torch +cpu` 是配合 torch_npu 的已选 PyTorch 构建，并不代表模型在 CPU 执行。
模型 cache recipe 的 prefix granularity 固定为 128，实验配置显式匹配。

环境记录优先读取 Git HEAD；通过源代码归档部署、没有 `.git` 的目录读取根目录
`BASE_COMMIT`（一个完整 40 字符提交号）。同时记录实际 runtime、kernel、scheduler
和实验源码逐文件 SHA256 及整体 manifest SHA256，因此新增或修改代码也可追溯。
Git checkout 另记录相对 HEAD 的 binary diff SHA256；归档环境该字段为 null，
不把缺少 Git 元数据误称为干净工作树。

## 1. 冒烟与请求隔离

```bash
python -m examples.task1_qwen35.run_engine \
  --model "$HOME/task1/models/Qwen3.5-0.8B" \
  --output "$HOME/task1/results/smoke" --mode smoke \
  --input-lengths 32 --decode-steps 4 2>&1 | tee "$HOME/task1/logs/smoke.log"
```

保存实际 input/output IDs、解码文本和引擎元数据。A→B→A 使用同一 Engine、
`max_num_seqs=1`，每次请求正常结束，禁用 prefix caching，验证槽位复用后 A 的
token 序列是否一致。它是端到端隔离检查，不能替代算子层对最终状态的对照。

另外可运行四个真实对话样例（中文、英文、Python 代码、整数运算）：

```bash
python -m examples.task1_qwen35.run_engine \
  --model "$HOME/task1/models/Qwen3.5-0.8B" \
  --output "$HOME/task1/results/chat" --mode chat --decode-steps 128 \
  2>&1 | tee "$HOME/task1/logs/chat.log"
```

chat 模式使用 checkpoint 自带 chat template，`enable_thinking=False`，
greedy 并正常在 EOS 停止，`--decode-steps` 在此是最大生成长度。
完整提示、实际 token 数、token IDs 和可读回答均保存；主性能矩阵的固定长度
fixture 完全不变。自然语言样例只检查推理可用性，不替代数值正确性对照。

## 2. 四组正常性能测量

```bash
python -m examples.task1_qwen35.run_engine \
  --model "$HOME/task1/models/Qwen3.5-0.8B" \
  --output "$HOME/task1/results/benchmark" --mode benchmark \
  --input-lengths 32 512 --output-lengths 32 128 \
  --warmups 2 --repeats 5 2>&1 | tee "$HOME/task1/logs/benchmark.log"
```

固定 token fixture 由 tokenizer 构建后精确裁到指定长度，保存完整 IDs 与 SHA256；
`ignore_eos=True` 保证测量的生成长度。首个 token 来自 Prefill，生成 G 个 token
仅有 G−1 个 Decode 步。报告所有原始采样及中位数/min/max。

主结果是**前端观测**：TTFT 包含 IPC、调度、H2D、模型、采样及输出处理，
不含此前完成的分词和模型加载；TPOT 使用首末 token 交付时间除以 G−1。
首帧若合并了多个 token，精确 TTFT 和 TPOT 标记为空，不能用首帧到达时间冒充。
收到输出意味着对应 token 的设备计算已完成；前端 `torch.npu.synchronize()`
不能同步另一个 worker 进程，因此此处不使用伪同步。

`enable_log_request_stats=True` 在运行日志中输出引擎自身 Prefill/TTFT/Decode
统计，属于 host 生命周期口径。模型计算阶段的同步时间由下一节单独获取。
`manifest.json` 记录设备级 `npu-smi info` 原文与快照时机；整设备占用含背景进程，
这些快照不能称为连续采样峰值。

## 3. 同步模型计时、真实 worker HBM、NPU profile

```bash
python -m examples.task1_qwen35.run_engine \
  --model "$HOME/task1/models/Qwen3.5-0.8B" \
  --output "$HOME/task1/results/diagnostic" --mode diagnostic \
  --input-lengths 32 512 --decode-steps 8 \
  2>&1 | tee "$HOME/task1/logs/diagnostic.log"
python -m examples.task1_qwen35.summarize_diagnostics \
  --run "$HOME/task1/results/diagnostic" \
  --output "$HOME/task1/results/diagnostic/summary.json"
```

另用新目录重复运行并加 `--profile`，各输入的第一个 Prefill/Decode forward
会生成 `torch_npu.profiler` trace。分析 operator/CPU 调度空洞/转换时间；带 profiler
的 forward 标记为 `profiled=true`，汇总排除其耗时。同步诊断也会改变 overlap，
不得替代正常性能主矩阵。
每种诊断输入先执行 `--warmups` 次完整请求（默认 2 次），预热期间关闭 worker
观测和 profiler，避免把首次形状编译当作稳态计算瓶颈。

诊断通过 spawn 入口安装 `ModelRunner.forward` 观测包装，在真实 worker 数据线程
内计时，返回原始结果，不修改模型计算。每次 forward 前后同步 NPU，记录
allocated/reserved 及该 forward 的 allocator peak。reset 范围仅覆盖模型 forward，
不包含此前 cache 分配及调度前处理的临时峰值；驻留权重/cache 仍计入 allocated。
包装仅在 correctness/diagnostic 模式启用，主基准没有它。

这两个模式还记录 `weight_audit.json`：实际消费的 checkpoint 键及形状、原 loader 返回的 loaded set、全部文本参数的 device/dtype、参数覆盖和 tied-weight storage 同一性。只调用原 loader 一次，视觉/MTP 与可重建 rotary buffer 的忽略键单独列出；缺少文本参数会失败。参数级覆盖不能独自证明 fused projection 每个 shard 的数值，必须结合原始键清单与整模型对照。

## 4. 完整模型独立数值对照

```bash
python -m examples.task1_qwen35.run_engine \
  --model "$HOME/task1/models/Qwen3.5-0.8B" \
  --output "$HOME/task1/results/correctness" --mode correctness \
  --correctness-lengths 1 16 128 512 --decode-steps 32 \
  2>&1 | tee "$HOME/task1/logs/correctness.log"
```

32 个输出对应 1 次 Prefill 和 31 次 Decode。真实 worker 为每一步保存全词表 logits、模型输入和位置。比较时逐步核对实际输入 IDs、文本位置和阶段，并检查 greedy 输出等于该步 logits 的 argmax。另一个进程通过官方
Transformers 模型，对**同一 Engine 生成轨迹做 teacher forcing**，保留 HF 自己的
cache；每个 case 都建立新 cache。不会因自由生成分叉而比较不同输入下的 logits。
HF 显式使用 eager attention 和官方 PyTorch conv/GDN/norm，无 TokenSpeed 算子。

比较入口要求预先显式指定阈值，不内置“足够宽松”阈值：

```bash
python -m examples.task1_qwen35.compare_hf \
  --model "$HOME/task1/models/Qwen3.5-0.8B" \
  --engine-results "$HOME/task1/results/correctness" \
  --output "$HOME/task1/results/hf_comparison.json" \
  --device npu --dtype bfloat16 --reference-state-dtype float32 \
  --max-abs 0.5 --max-relative-l2 0.02 \
  --min-cosine 0.999 --min-top1-agreement 0.95
```

上述数值与预先固定的[数值验收协议](../../docs/task1-numerical-protocol.md)
一致，须结合模块误差分析解释，不是所有 BF16
实现的通用保证；失败时定位误差来源，不事后改阈值把失败改为通过。
本次整模型主验收使用 `--device npu --dtype bfloat16 --reference-state-dtype float32`：先结束 TokenSpeed
进程，再在同一个可见 NPU 上运行官方 Transformers 5.12.0 eager 参考，二者
使用相同权重和 token IDs。参考保留官方数学计算，使用独立 cache 实例从首次
Prefill 写入起保存 FP32 recurrent state；conv/KV 仍为 BF16。这称为“官方 HF
计算 + 显式 FP32 recurrent-cache policy”，不称为未经修改的 stock HF。
Transformers 5.12.0 默认 cache 会把 FP32 递推结果存为 BF16，与本实验的 FP32
状态池精度不同。可明确传 `--reference-state-dtype model` 重现原始官方缓存；
该选项无默认值，两个口径的结果分开保存。每步核验全部 24 层的实际 cache dtype。
五轮默认缓存对照的失败与后续精度勘误均保留，详见数值协议。
跨设备数学验证由算子/模块测试中的独立 CPU FP64
表达式承担。脚本也支持 CPU 或 FP32 整模型参考，但这些是额外实验选项，
不能据脚本支持就声称已经完成该设备或精度下的整模型实测。

全词表不做过滤或非有限值掩码，任意 NaN/Inf 均失败；两端各自实现 causal attention mask。主验收要求完整运行与全部长度、32 个输出和 A→B→A；显式 `--labels` 或 `--allow-partial` 可用于诊断，报告会区分数值比较通过与完整协议通过。

输出每步 max absolute error、RMSE、relative L2、cosine、top-1 一致性、参考
top-1/2 margin 和 token logprob。整体阈值失败或 A1/A2 logits 不完全相同时
返回非零状态。HF eager Gated RMSNorm 有中间 BF16 舍入；Qwen NPU 路径现已
显式对齐 normalized activation 和 weight multiply 的 BF16 边界。通用融合
算子仍保留原契约。具体修复与首轮失败记录见上述协议，验收阈值保持不变。

## 算子与实验工具回归

先按本文开头设置单 chip 环境，在仓库根目录运行以下完整 NPU/配置回归命令。
当前组合共 251 项：norm/gate 114、卷积 21、GDN 37、SwiGLU 25、
Qwen attention/RoPE 28、其他 MHA 4、配置 9、线性注意力映射 13。

```bash
python -m pytest -q \
  tokenspeed-kernel-npu/test/test_gdn.py \
  tokenspeed-kernel-npu/test/test_qwen35_norm_gate.py \
  tokenspeed-kernel-npu/test/test_causal_conv_task1.py \
  tokenspeed-kernel-npu/test/test_qwen35_activation.py \
  tokenspeed-kernel-npu/test/test_qwen35_attention.py \
  tokenspeed-kernel-npu/test/test_mha.py \
  test/runtime/test_ascend_gdn_config.py \
  test/runtime/test_linear_attn_mapping.py \
  --junitxml="$HOME/task1/logs/final-npu-tests.xml"
```

实验工具与下载器共 33 项 CPU 测试；其中 23 项检查实验工具，2 项使用本地
真实 tokenizer 检查 chat token IDs 与 IPC 序列化，8 项检查离线下载行为。
完整执行需要 CPU PyTorch、Transformers、tokenizers 与 msgspec，无需运行 NPU 推理：

```bash
python -m unittest -v \
  examples.task1_qwen35.test_bookkeeping \
  examples.task1_qwen35.test_numerical_bookkeeping \
  examples.task1_qwen35.test_reference_cache \
  examples.task1_qwen35.test_profile_analysis \
  examples.task1_qwen35.test_chat_input \
  examples.task1_qwen35.test_download_model
```

这些测试检查 token 数、G−1 分母、首帧合并、最后空帧、数值统计、轨迹校验、
overlap 排除条件、tied-weight 覆盖、参考 cache 精度隔离及 profiler 汇总口径。
缺少张量依赖时部分用例会跳过，应核查测试输出，不能把 skip 计为通过。

下载器的 8 项测试仅依赖 Python 标准库，可在未安装 PyTorch 的本地机器单独运行：

```bash
python -m unittest -v examples.task1_qwen35.test_download_model
```

这些离线用例使用临时目录和模拟网络响应，覆盖已有文件校验、成功后的原子替换、
校验失败或传输中断时保留原文件及 partial、partial symlink 拒绝、合法临时 manifest
读入后下载往返，以及不安全/未固定 revision/重复清单项的拒绝。它们不访问网络或
依赖本机私有文件，也不替代真实模型下载、NPU 算子与整模型验证。

## 5. 汇总实测结果

完成主基准和独立诊断后生成报告表格（仅需 Python 标准库）：

```bash
python -m examples.task1_qwen35.summarize_results \
  --benchmark "$HOME/task1/results/benchmark" \
  --diagnostic "$HOME/task1/results/diagnostic" \
  --reference "$HOME/task1/results/hf_comparison.json" \
  --output "$HOME/task1/results/summary"
```

输出 `summary.md`、`benchmark.csv`、`summary.json`，从逐次原始记录重算
median/min/max，排除 warmup，验证实际 token 数、G−1 分母、输入 hash 和 cache
命中。独立列出 worker 模型计时、allocated/reserved/forward peak 与 profiler
状态；保留设备原始快照及所有输入文件 SHA256。诊断行明确其输出长度，不能把
8 个输出的 HBM 峰值当作 128 个输出的峰值。`--diagnostic` 可接多个目录；可省略
`--reference`；默认拒绝未完成运行，`--allow-partial` 只用于查看进度。

Engine overlap 可能在最后一个输出后仍执行一次未交付 decode。新 observer 只在
真实 extend 开始时绑定新请求标签，并按预期 forward 数收集；多余 forward
另存 `worker_excluded` 的输入/位置/原因。旧记录中的唯一前缀多余 decode 只有在
其输入等于上一请求最后输出、位置等于上一请求 L+G−1 时才可排除，并在数值报告
逐项保留证据；其他多余或缺失记录仍失败，不能任意截取前/后若干步使测试通过。

## 6. 只读分析 profiler 导出

`analyze_profile.py` 仅使用 Python 标准库读取已经生成的导出文件，不导入
PyTorch/torch_npu，不启动推理。字段已按实测环境安装的 torch_npu 2.9.0.post2
源码核实，来源及 SHA256 记录在 [`profiler_schema.json`](profiler_schema.json)。
每个输入长度、Prefill/Decode 的 trace 分别分析，不把不同采样窗口拼成一次运行。

```bash
python -m examples.task1_qwen35.analyze_profile \
  --directory /path/to/one/ASCEND_PROFILER_OUTPUT \
  --trace-selection host-to-device \
  --output "$HOME/task1/results/profile-summary-one-window"
python -m unittest examples.task1_qwen35.test_profile_analysis -v
```

输出 `profile_summary.json`、`kernel_types.csv`、`operator_totals.csv` 和
`trace_categories.csv`（仅生成对应输入存在的表）。具体统计口径：

- `kernel_details.csv` 使用已确认的 `Device_id`、`Name`、`Type`、
  `Accelerator Core`、`Start Time(us)` 和 `Duration(us)`，按类型/core/device
  汇总次数及累积时间，同时对每个设备的时间区间求并集。
- `operator_details.csv` 保留 Host/Device 的 Self 和 Total 区别。Total 含子算子，
  不能跨层累加成总体时间；Self 的设备时间也是关联 kernel 之和，可能重叠。
- `trace_view.json` 的 `ts`/`dur` 单位均为微秒。自动选择遵循已安装解析器的
  `HostToDevice` 配对 flow，将终点按 `(pid, tid, ts)` 匹配唯一的 `ph=X` 事件。
  无法匹配的 flow 单独保留，未关联的设备事件不臆断为 kernel；无匹配时明确
  标记 `selected_device_events_found=false`，不能解释成设备完全空闲。
- trace 跨度取全部正时长 X 事件的最早开始到最晚结束，并另报选中设备事件的
  跨度、busy 区间并集、间隙及最大间隙。并集去除了同时执行的重复时间；累积
  duration 可能超过跨度，不能直接相除后称为设备利用率。这里的 busy fraction
  仅是所选时间线的覆盖率，也不等于 AI Core 占用率。

`host-to-device` 得到的是已关联事件子集，busy union 可能低估全部设备活动。
如需包含未关联事件，先核查输出的 `processes` 元数据，再显式使用
`--trace-selection explicit-pid --device-pids <已核实的设备时间线PID>`。
工具不根据未确认的进程名称猜测设备身份，也不假设 CSV 与 trace 的时钟原点
相同。原文件不被修改，输出记录每个输入文件的 SHA256；缺失文件会列出，
不兼容字段会报错。上述单测使用合成 fixture，不能作为实际性能结果。

## 7. 已完成实验的公开证据

[`results/`](results/) 保存从原始记录按字段白名单导出的 JSON/CSV，每个文件小于
500 KB。公开数据保留输入 token IDs/hash、逐步数值、逐次计时、源码指纹及原文件
SHA256；省略主机、用户名、绝对路径、PID 和其他进程信息。原始文件到本地路径的
映射留在私有实验归档中。`baseline_commit` 是部署归档的起始提交，实际运行版本
由各次 `source_manifest_sha256` 标识，不能把起始提交当作最终实现提交。

| 证据 | 入口与口径 |
| --- | --- |
| 最终数值验收 | [`final_correctness.json`](results/final_correctness.json)、[`224 步明细`](results/final_correctness_steps.csv)：显式 FP32 recurrent-cache 参考，阈值不变，224 步通过；A–B–A 输出及 logits 完全一致。 |
| 原版 HF 兼容性 | [`final_stock_correctness.json`](results/final_stock_correctness.json)、[`逐步明细`](results/final_stock_correctness_steps.csv)：原版模型默认 BF16 recurrent cache，224 步 top-1 一致，但最大 relative L2 为 0.03434，数值阈值未通过。 |
| 历史失败 | [`stock_hf_rounds.json`](results/stock_hf_rounds.json) 保留五轮默认缓存结果；[`fp32_reference_history.json`](results/fp32_reference_history.json) 保留两轮 FP32-cache 失败，说明仅改变参考缓存精度并未消除全部偏差。各 JSON 指向对应逐步 CSV。 |
| 最终性能 | [`final_benchmark.json`](results/final_benchmark.json)、[`20 次正式样本`](results/final_benchmark_samples.csv)：32/512 输入 × 32/128 输出，每组 2 次预热、5 次测量。各组 `final_samples_*.json` 保留包括预热在内的完整 token 到达时间。 |
| 基线性能 | [`baseline_benchmark.json`](results/baseline_benchmark.json)、[`20 次正式样本`](results/baseline_benchmark_samples.csv)：顺序 GDN 版本，采用相同输入和统计口径；两轮在共享服务器不同时间运行，不能将所有差值单因归于某个算子。 |
| 显存与 worker 诊断 | [`final_diagnostics.json`](results/final_diagnostics.json)、[`256 次 forward`](results/final_diagnostic_steps.csv)：独立关闭 profiler，32/512 输入各生成 128 token。框架 allocator 峰值与设备整体 HBM 快照分别列出，快照不代表峰值。 |
| 算子跟踪 | [`final_profile_summary.json`](results/final_profile_summary.json)、[`baseline_profile_summary.json`](results/baseline_profile_summary.json)：每轮四个独立窗口，只采首个 Prefill/Decode，统一以 HostToDevice 关联设备事件统计。JSON 同时链接算子、kernel 名称、Task Type 与窗口 CSV。 |
| 自然语言 smoke | [`final_chat_smoke.json`](results/final_chat_smoke.json)：中文、英文、代码、算术四例的原始提示词/输出、token IDs/数量和停止原因。无预热，不作性能基准；保留输出中的语义和格式问题，不宣称指令遵循全部通过。 |
| 权重覆盖 | [`final_weight_audit.json`](results/final_weight_audit.json)：原 loader 调用一次，230 个 loaded 名称及一个共享 lm_head 覆盖全部 231 个文本参数；保留逐参数 dtype、shape 和覆盖方式。 |
| 测试与源码对应 | [`tests_summary.json`](results/tests_summary.json) 保留 251 项 NPU 模块测试和 33 项 CPU 工具测试的用例/状态/耗时；[`final_source_equivalence.json`](results/final_source_equivalence.json) 记录测量版本之后的格式化 hash 映射。 |

参考精度的定义和更正过程见
[`数值验收协议`](../../docs/task1-numerical-protocol.md)。FP32-cache 主验收通过与原版
BF16-cache 对照失败是不同结论，不能合并成“原版 HF 全部数值通过”。

Profiler Level0 不提供 AIC metrics；实际日志还报告停止于 RECORD 状态和不完整
memory records。公开跟踪结果仅支持已观察事件的数量、耗时与区间并集分析，
不证明捕获完整，也不构成硬件利用率、内存带宽或 HBM 峰值测量。Profiled 时间
含采集开销，不能替代无 profiler 的 TTFT/TPOT。
