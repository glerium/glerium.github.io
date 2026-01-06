---
layout: blog
title: "Agent Memory 相关文献整理"
date: 2026/01/05 19:26:42
tags:
    - LLM
    - Agent
    - Agent Memory
categories:
    - 论文笔记
    - 论文总结
---

## MemAgent (2025.06)

任务：长文档的QA

思路：把文档视作一个“证据流”，通过流式读取文档的方式，动态更新文档对应的记忆；最后通过最终的记忆回答问题；这个流程可以通过强化学习进行优化：当LLM答对问题时，就给予奖励。

<!--more-->

优势：流式处理文档，拥有无限长度；RL可以引导模型保留正确的信息；性能复杂度为O(n)，避免了传统transformer的二次方复杂度

## MEM1 (2025.06)

解决的问题：智能体在长上下文中的诸多问题（n^2复杂度、lost in the middle、超长上下文的泛化问题）

思路：不再保留全部的历史上下文，而是维护一个internal state，根据每轮调用来更新IS；使用强化学习优化系统，并以任务成功为训练目标

内部状态结构：xml风格，<IS>内部状态，<query>查询内容，<answer>LLM响应结果，<info>外部工具的输出；每次只保留最近一次的<IS>

## Agentic Context Engineering (2025.10)

解决的问题：简洁偏差、上下文坍塌

思路：维护一个只允许增量更新的playbook，构建multi-agent系统来维护它（generator, reflector, curator）

## MemSearcher (2025.11)

解决的问题：上下文过长导致的①噪声过多 ②成本上升

思路：维护一个固定长度上限的memory，每次只允许模型访问这个memory

![image.png](image.webp)

## Mem0 (2023.07)

短期记忆：直接存储在 context window 中

长期记忆的形式：embedding + metadata

查询：通过 vector search 匹配 top-k 的记忆输入LLM

## LightMem (2025.10)

思路：借鉴人类的记忆模型，把记忆分为感官记忆、短期记忆、长期记忆

实现：

- 感官记忆 → 短期记忆：用 LLMLingua-2 对上下文进行压缩，然后按主题切段

    - 当主题内部达到上下文边界时，触发压缩

- 短期记忆 → 长期记忆：在系统离线或空闲时触发长期记忆的整理、去重和巩固

## Mem-α (2025.09)

记忆架构：三层记忆，核心记忆、情景记忆、语义记忆；允许insert delete update三种操作

![image.png](image%201.webp)

思路：采用强化学习训练一个记忆决策智能体

奖励：准确率奖励、tool call格式奖励、记忆压缩比奖励、记忆内容有效性奖励（使用LLM评判）

## Memory-R1 (2025.08)

思路：把流程分为两个阶段，记忆构建（更新记忆）、问答生成（根据用户询问筛选记忆）；分别训练两个智能体，一个操作记忆，另一个负责筛选候选记忆

记忆蒸馏：采用RAG检索出top-k条候选记忆，然后要求记忆蒸馏智能体从中筛选出有效的记忆

## Generic Agentic Memory via Deep Research (2025.11)

思路：不提前压缩上下文，而是在需要的时候即时生成有意义的上下问题；

实现：两个Agent，memorizer 和 researcher

- memorizer：把历史轨迹按照session分解切片后，生成摘要，并把完整信息存入page store中

- researcher：当有在线请求时，根据摘要进行deep research，不断规划检索动作、获取页面、整合信息，直到生成满意的答案

![image.png](image%202.webp)

