---
layout: blog
title: "[论文笔记] Improving Language Understanding by Generative Pre-Training (GPT-1)"
date: 2025.07.18 00:00:00
tags:
    - LLM
categories:
    - 论文笔记
---

### 研究背景

- 现实生活中无标签的文本数据远多于带标签的数据

  - 现有的许多模型依赖于大量任务特定的、带标签的数据进行训练，十分费时费力

  - 无监督的方式需要考虑两大问题：

    - 在pretrain时采用什么任务对提取知识最有效果（解决方法：生成式任务）

    - 如何将无监督中学习的知识应用到下游任务上（解决方法：minimal change to structure during finetuning）

### 亮点

- 对于不同任务而言，GPT-1的模型架构是统一的，微调时对架构的更改很少

- 在9 of 12个任务上取得了SOTA

- 证明了在多个任务上，基于广泛无标签语料的生成式预训练可以提升模型效果

### 方法

- 模型架构：12层的Transformer decoder-only架构

  - 预训练阶段：h = transformer(x), $y = W_e(h)$

  - 微调阶段：h = transformer(x), $y = W_y(h)$

  - 这样在微调阶段，只需要训练 $W_y$ 矩阵即可

- 预训练阶段

  - 任务：生成式目标，优化next token prediction的似然函数L1

- 微调阶段

  - 任务：定义目标似然函数L2 = P(y | x)；L1为输入token的似然函数

    - $$loss = L2 + \alpha \times L1$$