---
title: "[论文笔记] Mem-α: Learning Memory Construction via Reinforcement Learning"
date: 2026/01/12 17:49
tags:
    - Agent
    - Agent Memory
    - 多模态
    - 多模态Agent
    - 强化学习
categories:
    - 论文笔记
---

目前状态：九月份挂了arxiv，ICLR2026 评分4664

## 动机

目的：构建一个能有效通过交互与反馈，构建复杂记忆的RL Agent

方法上：以往基于RL的memory agent受限于较为简单的记忆结构（全部重写/维护一个分点列表），无法应对复杂数据

数据上：通常聚焦于LoCoMo的设定，上下文长度较短（max: ~26k tokens）；训练和测试在统一分布上，任务较为简单（？）

<!--more-->

## 贡献

1. 把记忆构建转化为智能体决策问题，根据信息chunk决定执行怎样的记忆操作，并基于下游任务的准确性来设定多种奖励

2. 构造了一个涵盖多种多轮对话记忆的数据集

3. 采用包含核心记忆、情景记忆、语义记忆的综合记忆架构，并与训练流程解耦

    4. 发现提出的Mem-alpha方法具有强泛化性：仅在30k数据上训练，却也能在400k的测试集上泛化 → 说明提取出了根本性的记忆管理原则，而不仅仅是记录了特定模式；（也能说明512 token的core memory是完全充足的，足以涵盖400k的上下文）

## 相关研究

### Latent memory

把记忆融入到模型的内部结构中，如hidden states, KV cache, soft prompt, model parameter

优点：无需外部存储

缺点：记忆容量较小，而且需要直接访问模型内部结构，无法用于闭源模型

## 方法

### 三种记忆模式

![image.png](image.webp)

Core memory：一段固定长度的段落（<512 tokens），一般是关于用户的基本信息，模型每次推理都能访问到，只允许完全重写（理由：保证整体质量）

Semantic memory：事实知识的分点列表，允许增量更新（insert, update, delete）；类似long-term memory的语义相关性

Episodic memory：按照时间顺序排列的情景记忆，xxx在xxx时间做了xxx事；类似short-term memory；不过mem-alpha实现中和语义记忆的检索方式一样

检索方式：core直接放进每次的prompt中，semantic和episodic每次通过embedding或bm-25检索到top-k的memory放入prompt

优点：记忆具有多样性，不局限于只允许完全重写（传统基于RAG的记忆），或是只允许增量更新（ACE）

### 训练方案

![image.png](image%201.webp)

奖励函数：r1 (accuracy) + r2 (tool-call format) + beta * r3 (compression rate) + gamma * r4 (memory quality)

其中 beta 和 gamma 是超参数

r1 和 r2 都是 [0, 1] 之间的指标

r3 = 1 -  len_new / len_old

r4 是用 qwen3-32b 评判的记忆有效指标

训练方法：GRPO，为了鼓励模型探索，移除了 KL 散度惩罚项

训练目标：一个单独的记忆决策模型（Qwen3-4B，没用8B的原因是大模型在指令遵循上出现问题）；QA则采用另外一个 fixed-param LLM。

## 评测

### Benchmark

MemoryAgentBench，一个测试agent记忆能力的benchmark，包含三类数据集：

- Accurate retrieval：文档检索、长对话检索

    - SQuAD：单文档 QA

    - HotpotQA：多文档 QA

    - PerLTQA：长上下文 QA

- Test-time learning: 测试时文本分类

    - TREC Coarse：问题类型粗分类

    - NLU：多领域意图分类

    - PubMed：医学文献主题分类

- Long-Range Understanding：超长文本摘要

    - BookSum：长书籍摘要



