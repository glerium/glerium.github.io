---
layout: blog
title: "[论文笔记] FlashAttention-4: Algorithm and Kernel Pipelining Co-Design  for Asymmetric Hardware Scaling (2026.03)"
date: 2026/04/22 15:44
tags:
  - Infra
  - FlashAttention
categories:
  - 论文笔记
---

背景：FA3虽然取得了很大的性能突破，但其主要针对 H100 架构的硬件，目前业内主要采用的是 Blackwell 架构，tensor core吞吐量翻倍，而SRAM等其他单元增长不大。因此需要针对这一特性进行新的算法优化。

FA4主要进行了以下几点改进：

1. 重新设计流水线以实现全异步的矩阵乘法以及更大的矩阵尺寸
2. 通过软件模拟指数运算与softmax以减少 non-matmul op
3. 使用 tensor memory 与 2-CTA MMA 模式减少反向传播中的 IO 开销
4. 实现上的改进：使用 CuTe-DSL 实现，编译速度提升 20-30x

