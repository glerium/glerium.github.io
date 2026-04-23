---
layout: blog
title: "[论文笔记] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (2024.07)"
date: 2026/04/22 10:54
tags:
  - Infra
  - FlashAttention
categories:
  - 论文笔记
---

# [论文笔记] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision (2024.07)

在 FA2 中，注意力的计算速度得以大幅提升。但其在新一代GPU上的利用率依然偏低，比如在 Hopper 架构的 H100 上利用率仅有35%，而GEMM内核则可达80-90%。这种现象主要源自以下几个原因：

1. 实现差异：没有采用Hopper架构的专有指令集进行加速

2. 先前实现遵循简化的同步模型，没有利用异步性和低精度计算的特性

因此 FA3 中提出了以下三点改进：

<!-- more -->

1. 生产者-消费者异步化，类似于流水线并行；将线程拆分为两部分，一部分为生产者，另一部分为消费者。

2. 将 softmax 与分块 GEMM 运算相并行；重构 FA2 算法，解耦了 softmax 与 GEMM 操作的依赖性

3. 针对硬件特性，将部分 GEMM 操作替换为 FP8 低精度版本，实现大约一倍的 TFLOPs 加速

## GPU内存结构

线程层次结构：从低到高分别为，threads, warps (32 threads), warpgroups (4 warps), threadblocks (CTA), threadblock clusters, grids
