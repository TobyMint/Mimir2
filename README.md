# Mimir2

面向智能体的内存管理系统 —— 第三届研究生操作系统大赛参赛项目。

## 背景

随着基于大语言模型的智能体系统发展，其推理范式已从单轮生成扩展为涵盖规划、执行与反思的长生命周期复杂过程，带来 **KV Cache 持续累积、上下文高度冗余、推理路径分支、工具调用产生大规模中间数据** 等问题。传统推理框架（如 vLLM、llama.cpp）主要面向线性序列生成，难以高效应对智能体推理的内存特征。

本赛题要求基于开源技术栈，实现一个面向智能体推理过程的内存管理系统，在保证推理效果的前提下，通过对 KV Cache、上下文结构及显存分配机制的系统性优化，降低内存占用并提升整体推理效率。

## 赛题信息

- **赛道：** 研究创新赛道
- **题目：** 面向智能体的内存管理系统设计与实现（高校赛题）
- **基础操作系统：** openEuler / openKylin / OpenHarmony 等（至少一个国内主流开源 OS）
- **基础框架：** vLLM / llama.cpp（择一扩展）
- **评测模型：** Qwen、MiniCPM 等开源大模型

## 优化方向

- KV Cache 生命周期管理（复用 / 淘汰 / 分层存储）
- 分支推理内存共享（Copy-on-Write 机制）
- Prompt 与上下文压缩（去重 / 精简）
- 工具调用数据优化（结构化存储 / 按需加载）
- 分层内存与异构存储优化（GPU 显存 / 主存 / 外存的冷热分离与动态迁移）
- 异构 AI 加速硬件支持（CUDA / DTK / CANN 等）

## 评分构成

| 维度 | 占比 |
|------|------|
| 功能完整性 | 30% |
| 应用效果 | 40% |
| 代码规范性 | 20% |
| 文档质量 | 10% |

## 项目结构

```
Mimir2/
├── docs/        # 赛题要求及设计文档
└── README.md
```

## 参考资料

- [vLLM 文档](https://docs.vllm.ai/en/stable/index.html) / [vLLM 仓库](https://github.com/vllm-project/vllm)
- [llama.cpp 仓库](https://github.com/ggerganov/llama.cpp)
- [MiniCPM3-4B](https://www.modelscope.cn/models/OpenBMB/MiniCPM3-4B)
- [LangChain](https://github.com/langchain-ai/langchain) / [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)

---

*项目处于初始阶段，具体设计与实现待后续展开。*
