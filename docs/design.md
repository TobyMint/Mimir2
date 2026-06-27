# Mimir2 设计文档

## 1. 问题背景

智能体（Agent）推理与传统单轮 LLM 推理在内存使用上有本质差异。现代
ReAct 风格的智能体在"推理 → 工具调用 → 推理"的循环中持续运行，呈现
以下特征：

- **KV Cache 跨轮累积**：每一轮把上一轮的工具结果拼回上下文，prompt
  单调增长。轮与轮之间存在大量共享前缀，理论上可复用。
- **工具调用引入间隙（gap）**：智能体在推理步之间等待工具返回，期间
  其 KV Cache 在 GPU 中处于空闲状态。
- **长生命周期**：一个智能体程序可能持续数十轮。

传统推理引擎（如 vLLM）默认的 KV 管理面向单轮生成：请求结束即释放
KV，靠 prefix cache 的 LRU 在显存压力下回收。这在智能体场景下会带来
一个问题——**当多个智能体并发争抢显存时，某个智能体在工具间隙的
KV 会被 LRU 回收，等它工具返回、进入下一轮时，前缀 KV 已丢失，必须
重新 prefill**，造成显著延迟。

资源受限场景（单卡 24GB）下，这个问题尤为突出：显存容量小，并发
智能体更容易触发回收，重算开销被放大。

## 2. 核心设计：KV Cache 三阶段生命周期

Mimir2 将智能体一次跨工具间隙的 KV Cache 生命周期划分为三个阶段，
分别由不同机制管理：

| 阶段 | 触发条件 | KV 所在 | 机制 | 作用 |
|------|----------|---------|------|------|
| A. 保活 | 工具即将返回（短工具） | GPU | **pin + TTL** | 命中前缀，免重算 |
| B. 落盘 | TTL 到期、工具未归、显存有压 | CPU 内存 | **offload**（规划中） | 不丢弃，留待 reload |
| C. 重载 | 工具返回、KV 已被驱逐出 GPU | CPU → GPU | **reload**（规划中） | 快于重算 |

### 2.1 阶段 A：pin + TTL（已实现）

当一个智能体某轮请求完成、且即将进入工具调用间隙时，若该工具历史
耗时较短（下一轮大概率在 TTL 窗口内返回），则将其 KV Cache **pin
在 GPU 中**，在 TTL 期间不被 LRU 回收。下一轮请求到来时，前缀命中
prefix cache，跳过重算。

**TTL 决策（pin 与否）：** 固定 TTL = 2 秒。仅当本轮调用了工具、且该
智能体（job）历史上该工具的平均耗时 ≤ TTL 时才 pin；长工具不 pin
（KV 会闲占显存，得不偿失）。这一策略基于一个经验观察：智能体工具
调用耗时呈长尾分布，多数较短，少数极长——对短工具 pin 能命中，对
长工具 pin 是浪费。

> 注：固定 TTL 是当前实现。更精细的方案是基于工具耗时分布的 cost
> model 自适应计算 TTL，作为后续优化方向。

**实现要点：**
- 在 vLLM v1 scheduler 中新增 `mimir` 调度策略。
- 请求携带 `job_id`（标识同一智能体的多轮）、`is_last_step`（最后一
  轮不 pin）、`has_tool_call`、`tool_duration` 等元数据，经 OpenAI
  API 的 `vllm_xargs` 传入。
- 请求完成时，按上述规则决定是否 pin；每个调度步开头检查 TTL 到期
  的 pin 并释放其 KV。

### 2.2 阶段 B/C：落 CPU 与 reload（规划中）

pin 的 TTL 到期时，当前实现直接释放 KV（丢弃）。这意味着对长工具，
pin 保不住的 KV 仍会丢失，下一轮仍需重算。

阶段 B/C 的目标是：**TTL 到期时不丢弃，而是 offload 到 CPU 内存；
等工具返回、KV 已被驱逐出 GPU 时，从 CPU reload 回 GPU**。微实验
表明，在本地 PCIe 带宽下，reload 比重新 prefill 快 3–7 倍（随上下文
长度增长），因此这条路径能进一步降低长工具场景的延迟。

这部分通过一个可替换的 **KV connector**（`MimirConnector`）实现，负责
KV 在 GPU 与 CPU 之间的搬运。connector 已有最小可用实现并验证了
reload 路径正确性，但在高并发下其同步搬运开销过大，需要 layer-wise
异步流水线优化后才能并入端到端使用（见 benchmark 文档中的相关实验）。

## 3. 与 native vLLM 的关系

Mimir2 构建在 vLLM 0.10.2 之上，不替换其 PagedAttention、continuous
batching、prefix caching 等基础机制，而是**在其 KV 管理之上叠加智能体
感知的保活/落盘/重载策略**。native vLLM 的 prefix cache 是被动的
LRU 回收，Mimir2 的 pin 是主动的、按工具语义保活。两者关系：

- 低压力下（并发少、显存充裕），native prefix cache 本身就能留住跨
  轮 KV，pin 无额外收益。
- 高压力下（并发多、显存紧张），native 的 LRU 会回收跨轮 KV，pin
  主动保活，带来收益。

因此 Mimir2 的价值在**资源受限 + 多智能体并发**场景，正是赛题所指。

## 4. 仓库结构

```
Mimir2/
├── vllm/                      # vLLM 0.10.2 源码（editable 安装）
│   ├── v1/core/sched/         # 调度器，pin/TTL 改动集中于此
│   │   ├── scheduler.py       # mimir policy + pin/unpin
│   │   └── request_queue.py   # MIMIR policy 注册
│   ├── v1/request.py          # Request 加 job_id 等元数据
│   └── distributed/.../v1/
│       └── mimir_connector.py # CPU-memory KV connector（阶段 B/C）
├── benchmarks/mimir/          # benchmark harness 与 trace
│   ├── collect_trace.py       # BFCL + DeepSeek 采集 agent trace
│   ├── replay_benchmark.py    # trace 回放压测
│   └── reload_vs_prefill.py   # reload vs prefill 微实验
└── docs/                      # 文档
```

## 5. 当前状态

- ✅ 阶段 A（pin + TTL）：已实现，端到端验证在 12 智能体真实压力下
  较 native vLLM 提升：JCT 中位 -10%，TTFT 中位 -17%（见 benchmark
  文档）。
- ⏳ 阶段 B/C（落 CPU + reload）：connector 已验证 reload 正确性与单
  请求收益，高并发优化（layer-wise 流水线）待做。
