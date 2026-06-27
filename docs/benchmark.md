# Mimir2 Benchmark 文档

## 1. 评测目标

衡量 Mimir2 的 KV 管理优化在**多智能体并发、资源受限**场景下相对
native vLLM 的端到端收益。判定标准严格遵循"端到端是唯一真理"原则：
以 job completion time（JCT）和 TTFT 为主指标，prefix cache 命中率、
显存占用等只作辅助诊断，绝不替代端到端。

## 2. 负载：BFCL web search agent trace

### 2.1 trace 采集

使用 BFCL v4 web search 数据集（100 个多跳研究问题）作为任务来源，
以 DeepSeek-v4-pro 扮演智能体"大脑"运行 ReAct 循环（web_search /
retrieve_page 工具），记录每一步的真实决策：

- 每轮 prompt token 数（跨轮累积增长）
- 每轮 decode token 数（reasoning + 回复合计，从 API usage 取真实值）
- 每轮是否调用工具

工具**结果**为合成文本（长度可控），工具**耗时**按 BFCL 论文报告的
分布（mean ~1.9s, std ~2.1s，对数正态）在回放时采样。即：智能体的
多轮决策结构来自真实运行，token 维度与耗时维度可控可复现。

采集到 21 条 trace，每条约 8 轮，最终 prompt ~2.8K token。

### 2.2 回放 harness（`replay_benchmark.py`）

每个智能体程序 = 一条 trace 的回放器，按 trace 脚本走完多轮：

- **异步推进**：N 个智能体各自独立推进，互不等齐。
- **Poisson 到达**：智能体按泊松过程注入，非同时涌入。
- **绝不 drain**：智能体间持续争抢显存，不在轮间清空 cache。
- **上下文累积**：每轮 prompt 按 trace 记录的 token 数（×scale 放大）
  增长，模拟真实智能体的前缀复用。
- **真实 decode**：每轮生成 trace 记录的真实 token 数，而非 1 个。

每个智能体用**不同的 filler 文本**构造 prompt，避免不同智能体的
prompt 共享前缀导致 prefix cache 跨智能体假命中。

### 2.3 压力证据

每次运行后从服务日志提取"压力证据"：GPU KV cache usage 峰值/均值、
prefix cache 命中率、preemption 计数。若 GPU usage 峰值低于 50%，
harness 会警告"压力不足，Mimir2 无发挥空间"——这是防止在无压力场景
下得出假结论的护栏。

## 3. 微实验：reload vs prefill

在接入 connector 前，先验证"reload 代替重算"这条路径的物理可行性
（`reload_vs_prefill.py`）。Qwen3-4B / RTX 3090，关闭 GPU prefix
caching，单请求测单步 TTFT：

| 前缀长度 | prefill | reload (CPU→GPU) | 加速 |
|----------|---------|------------------|------|
| 4K  (4080 tok)  | 0.538s | 0.175s | 3.1× |
| 16K (16368 tok) | 3.041s | 0.646s | 4.7× |
| 32K (32752 tok) | 8.871s | 1.191s | 7.4× |

**结论**：reload 端到端在所有长度都赢 prefill，且优势随长度增长
（3.1× → 7.4×）。这验证了阶段 B/C（落 CPU + reload）的物理基础。

但 reload 时间中约 80% 是 connector 的同步搬运开销（逐层 `.to(cpu)/
.to(cuda)`，无流水线），远高于纯 PCIe 传输的理论值。这说明 connector
需要 layer-wise 异步流水线优化才能逼近物理上限——这是阶段 B/C 并入
端到端使用前必须完成的工作。

## 4. 端到端结果：pin-only vs native

### 4.1 主结果（12 智能体，scale=8，真实压力）

配置：Qwen3-4B / RTX 3090（卡 3）/ 12 智能体 / Poisson 到达率 2.0/s /
trace 放大 8 倍（最终上下文 ~26K token）。压力证据：native 与 Mimir2
的 GPU KV usage 峰值均约 85%，prefix 命中率约 20–24%——真实驱逐压力。

| 指标 | native vLLM | Mimir2 (pin-only) | 改善 |
|------|-------------|-------------------|------|
| JCT 中位 | 136.9s | 123.1s | **-10%** |
| JCT p90 | 166.2s | 155.1s | -7% |
| TTFT 中位 | 10.37s | 8.56s | **-17%** |
| TTFT p90 | 36.2s | 33.4s | -8% |
| prefix 命中率 | 19.8% | 24.4% | +4.6pp |

**结论**：pin/TTL 在真实驱逐压力下端到端成立。pin 主动保住短工具
间隙的跨轮 KV，避免 native LRU 回收导致的重算。技术指标（prefix
命中率 +4.6pp）与端到端（JCT -10%、TTFT -17%）一致改善。

### 4.2 压力区间的关键性

- **4 智能体**：pin 与 native 无差异。原因：压力不足（GPU usage ~38%），
  native prefix cache 本就留住跨轮 KV，pin 无用武之地。
- **12 智能体**：pin 赢。原因：压力真实（usage ~85%），native LRU
  回收跨轮 KV，pin 救住。

Mimir2 的收益**仅在真实驱逐压力下显现**，这正对应赛题"资源受限 +
多任务并发"场景。低压力下无收益是符合预期的，而非缺陷。

## 5. 负面结果（诚实记录）

以下实验未带来端到端收益或导致问题，如实记录，作为后续优化的依据。

### 5.1 connector 单独（无 pin）高并发下慢 10×

12 智能体下，仅开 connector（无 pin）比 native 慢约 10×（JCT
~1451s vs 137s），3/12 智能体超时失败，GPU 撑至 99.9%。

根因：connector 只在请求结束时 store，但工具间隙期 KV 还在 GPU
prefix cache 里、未被驱逐，connector 无机会 reload——只付出同步搬运
开销却无收益。这证明**阶段 B/C 必须配合阶段 A 的 pin（触发驱逐→
落 CPU→reload）才有意义**，connector 单独是纯开销。

### 5.2 pin + connector 叠加导致系统停滞

12 智能体下，同时开 pin 和 connector，系统 prompt throughput 从
352 tok/s 衰减到 0，接近死锁。

根因：connector 同步 store 阻塞 prefill + pin 占显存加剧争抢 + TTL=2s
太短 pin 未复用。三重叠加，均无收益。

教训：pin 与 connector 不能简单叠加。connector 必须先做 layer-wise
异步流水线优化（消除 store 阻塞），再并入；且落 CPU 应只在 pin TTL
到期时触发（pin 保不住的场景），而非每请求都 store。

## 6. 复现

```bash
# 环境：conda activate mimir2（python 3.12, vLLM 0.10.2 editable）

# pin-only 服务（创新点 #1）
CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 MIMIR_PIN_TTL=2.0 \
vllm serve /data/models/Qwen3-4B --port 8199 --scheduling-policy mimir \
  --max-model-len 40960 --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.85 --dtype bfloat16

# native 对照服务
CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
vllm serve /data/models/Qwen3-4B --port 8199 \
  --max-model-len 40960 --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.85 --dtype bfloat16

# 跑 benchmark（两者同样配置，仅服务不同）
python benchmarks/mimir/replay_benchmark.py \
  --traces benchmarks/mimir/traces/batch30 \
  --num-agents 12 --arrival-rate 2.0 --scale 8 --port 8199 \
  --server-log <server_log_path> --label <label> \
  --out benchmarks/mimir/results/<label>.json
```

trace 采集（需要 DeepSeek API）：
```bash
export DEEPSEEK_API_KEY=...
python benchmarks/mimir/collect_trace.py --num-tasks 30 \
  --out benchmarks/mimir/traces/batch30
```
