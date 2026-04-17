---
layout: blog
title: "[论文笔记] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (2023.07)"
date: 2026/04/13 15:11
tags:
  - Infra
  - FlashAttention
categories:
  - 论文笔记
---

# [论文笔记] FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning (2023.07)

FlashAttention-2 是基于 FlashAttention 的改进版，运算效率得到了约 2x 提升。下面是本文的阅读笔记。

## 动机

虽然 FlashAttention 通过减少 IO 搬运，大幅提升了运算速度以及减少显存占用，但并没有完全发挥GEMM运算的理论速度，只达到了理论FLOPs的25-40\%。其原因在于GPU上不同 thread blocks 和 warps 分配不够优秀，核心占用率偏低且存在不必要的内存读写。因此 FlashAttn-2 进行了三点优化：

1. 调整算法，**减少非矩阵乘法的运算量**。在 GPU 上有专门的 GEMM 计算单元，吞吐量可高达其他算子的 16 倍。
2. 在序列长度维度上，将**注意力计算并行分配**到不同 thread block 上；
3. 在线程块内实现 warps 并行，减少共享内存通信。

## 背景

### 硬件简介

执行模型：GPU 内部有许多线程（即内核 kernel）。线程被组织成线程块（thread blocks）在 SM (streaming multiprocessors) 上运行。每个 block 内的线程被分组成 warps（线程束），即32个线程。同一个 warps 内的线程可以快速交换数据进行通信。同一个 block 的不同 warps 可以读写 shared memory 进行通信。

## 效果

\~2x 加速，达到理论峰值的 50-73\%。

