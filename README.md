# Mimir2

面向智能体的内存管理系统 —— 第三届研究生操作系统大赛参赛项目。

Mimir2 在 vLLM 0.10.2 之上，针对智能体多轮推理的 KV Cache 管理进行
优化。核心是 **KV Cache 三阶段生命周期**：工具间隙短则 pin 保活、
TTL 到期则落 CPU、工具返回则 reload，以减少资源受限多并发场景下的
重算开销。

## 背景

随着基于大语言模型的智能体系统发展，其推理范式已从单轮生成扩展为
涵盖规划、执行与反思的长生命周期复杂过程，带来 **KV Cache 持续累积、
上下文高度冗余、工具调用引入间隙** 等问题。传统推理框架（如 vLLM）
默认的 KV 管理面向单轮生成，在多智能体并发争抢显存时，会把某个智能体
工具间隙的 KV 回收，导致下一轮重算。资源受限（单卡 24GB）下该问题
被放大。

## 当前进展

| 机制 | 状态 | 端到端收益（12 智能体真实压力 vs native vLLM） |
|------|------|------|
| pin + TTL（阶段 A） | ✅ 已实现验证 | JCT 中位 -10%，TTFT 中位 -17% |
| 落 CPU + reload（阶段 B/C） | ⏳ connector 已验证 reload 正确性，高并发优化待做 | 微实验：reload 比 prefill 快 3.1–7.4× |

详见 [设计文档](docs/design.md) 与 [Benchmark 文档](docs/benchmark.md)。

## 赛题信息

- **赛道：** 研究创新赛道
- **题目：** 面向智能体的内存管理系统设计与实现（高校赛题）
- **基础框架：** vLLM 0.10.2
- **评测模型：** Qwen3（4B / 8B）
- **硬件：** 单卡 RTX 3090（24GB）

## 快速开始

### 环境

```bash
conda create -n mimir2 python=3.12 -y
conda activate mimir2
# 安装 build 依赖后
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu126
pip install -e . --no-build-isolation          # vLLM editable
pip install transformers==4.55.2                # 版本兼容
```

### 启动服务

```bash
# Mimir2（pin/TTL，创新点 #1）
CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 MIMIR_PIN_TTL=2.0 \
vllm serve /data/models/Qwen3-4B --port 8199 --scheduling-policy mimir \
  --max-model-len 40960 --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.85 --dtype bfloat16

# native vLLM（对照）
CUDA_VISIBLE_DEVICES=3 VLLM_ENABLE_V1_MULTIPROCESSING=0 \
vllm serve /data/models/Qwen3-4B --port 8199 \
  --max-model-len 40960 --max-num-batched-tokens 65536 \
  --gpu-memory-utilization 0.85 --dtype bfloat16
```

### 跑 benchmark

```bash
python benchmarks/mimir/replay_benchmark.py \
  --traces benchmarks/mimir/traces/batch30 \
  --num-agents 12 --arrival-rate 2.0 --scale 8 --port 8199 \
  --server-log <server_log> --label <label> \
  --out benchmarks/mimir/results/<label>.json
```

## 项目结构

```
Mimir2/
├── vllm/                              # vLLM 0.10.2 源码（editable）
│   ├── v1/core/sched/scheduler.py     # mimir policy + pin/TTL
│   ├── v1/core/sched/request_queue.py # MIMIR policy 注册
│   ├── v1/request.py                  # Request 加 job_id 等元数据
│   └── distributed/.../v1/mimir_connector.py  # CPU-memory KV connector
├── benchmarks/mimir/
│   ├── collect_trace.py               # BFCL + DeepSeek 采集 agent trace
│   ├── replay_benchmark.py            # trace 回放压测
│   ├── reload_vs_prefill.py           # reload vs prefill 微实验
│   └── results/                       # 实验结果 JSON
├── docs/
│   ├── design.md                      # 设计文档
│   ├── benchmark.md                   # Benchmark 方法论与结果
│   └── 赛题要求.md
└── README.md
```

## 文档

- [设计文档](docs/design.md) — 三阶段生命周期、pin/TTL 机制
- [Benchmark 文档](docs/benchmark.md) — 负载、harness、结果（含负面结果）
- [reload vs prefill 微实验](docs/reload-vs-prefill-microbench.md)

## 评分构成

| 维度 | 占比 |
|------|------|
| 功能完整性 | 30% |
| 应用效果 | 40% |
| 代码规范性 | 20% |
| 文档质量 | 10% |

## 参考资料

- [vLLM 文档](https://docs.vllm.ai/en/stable/index.html) / [vLLM 仓库](https://github.com/vllm-project/vllm)
- [BFCL v4 web search](https://gorilla.cs.berkeley.edu/blogs/15_bfcl_v4_web_search.html)
- [Qwen3](https://github.com/QwenLM/Qwen3)
