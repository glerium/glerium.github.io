---
layout: blog
title: "[论文笔记] FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness (2022.06)"
date: 2026/04/08 14:09
tags:
  - Infra
  - FlashAttention
categories:
  - 论文笔记
---

## 摘要

Transformer -> 时空复杂度为n^2，长序列的时间和内存占用过大

Appr. Attn -> 牺牲精度，但无法提升 wall-clock time -> 侧重于降低FLOPs，忽视了降低IO开销

FlashAttn -> IO aware，通过分块减少 HBM（显存）和 SRAM（寄存器）之间的读写次数 -> 节省IO成本

扩展：block-sparse attn，速度优于所有近似的注意力方法

<!--more-->

效果：

1. 速度提升： BERT-Large (nseq=512): ~15%；GPT-2 (nseq=1K): 3x

2. 质量提升：GPT-2 困惑度降低 0.7，长文本分类 ACC 提升 6.4 pt

> PPL：模型困惑度 $PPL=2^H=\exp\left( -\dfrac1N \displaystyle\sum_{i=1}^N \ln q(x_i|x_{<i}) \right)=\left(\displaystyle\prod_{i=1}^N \dfrac1{q(x_i|x_{<i})}\right)^{1/N}$，这里的H代表交叉熵

> PPL 可以理解为几何平均预测概率的倒数；PPL=1 代表完美预测

## 动机

Appr. attn：只考虑FLOPs，但忽视了IO操作的开销

分析发现，Transformer中的大部分操作是 memory-bound 而非 compute-bound

FlashAttn：减少HBM访问次数

两个关键问题：

1. 在没有完整输入的情况下完成 softmax 计算 -> 分块计算、多轮遍历，以增量方式规约（tiling）

2. 避免存储完整的 attn 矩阵 -> 在前向传播中存储 softmax 归一化因子，反向传播时快速重新计算注意力

效果：

1. 细粒度控制内存访问，把注意力计算融合到单个kernel
2. 重计算增加了FLOPs，但HBM访问次数更少，性能更加优秀
3. 此外内存占用随序列长度**线性增长**

理论分析：

1. FlashAttn 需要访问 $O(N^2d^2/M)$ 次内存，传统 attn 需要访问 $O(Nd + N^2)$ 次，这里M是SRAM容量
2. 下界证明：任意精确注意力算法的 HBM 访问次数都不优于该结果

延伸：block-sparse FlashAttn，IO复杂度更加优秀，且提升幅度与稀疏化比例成正比

基准测试：序列长度<512时，FlashAttn速度快，且内存效率高；block-sparse flashattn是速度最快的近似注意力方法

## 背景

### GPU的硬件架构

硬件层次从上到下分别是：SRAM (20MB，19TB/s), HBM (40GB, 1.5TB/s), DRAM (>1TB, 12.8GB/s)

compute-bound 和 memory-bound 的衡量方式：arithmetic intensity，每byte字节访问对应的算术运算次数（计算量除以数据量）

compute-bound：大矩阵乘法、多通道的卷积

memory-bound：逐元素运算、规约运算

在训练时，融合内核的中间结果依然需要写入HBM以供反向传播使用，因此性能会降低

### 对标准attn算法的初步分析

标准attn分为三步：

1. $S = QK^T$
2. $P = \text{softmax}(S)$
3. $O = PV$

其中，大部分操作是memory-bound的（如softmax），会显著拖慢速度；其他逐元素操作，如掩码、对P矩阵的随机失活等，会进一步加剧问题。

此外，从内存占用来看，标准attn会实例化一个 N\*N 的矩阵；在长序列情况下，这会带来很大的内存开销。

## FlashAttention算法

### 基本思路

将QKV分割为多个数据块，针对每个块计算attn输出；在将每个块的结果相加前乘以归一化因子，即可得到正确结果。

Tiling：将QKV分块计算

数值稳定的softmax计算步骤如下：

$$\begin{align} m(x) &= \max_i(x_i) \\ f(x)&=[e^{x_1-m(x)},\cdots,e^{x_n-m(x)}] \\ l(x)&= \sum f(x)_i \\ \text{softmax}(x)&=\dfrac{f(x)}{l(x)}\end{align}$$

对于一个长向量 x 的 softmax 操作，我们可以将其拆成两个向量 $x = [x^{(1)},~x^{(2)}]$ 分别计算，最后进行合并，即

$$\begin{align} m(x)&=m([x^{(1)},~x^{(2)}]=\max(m_1,m_2)) \\ f(x)&=[e^{m_1-m(x)}f_1,~e^{m_2-m(x)}f_2] \\ l(x)&=e^{m_1-m(x)}f_1+e^{m_2-m(x)}f_2 \\ \text{softmax}(x)&=\dfrac{f(x)}{l(x)} \end{align}$$

这意味着，我们只需要多维护 $m(x),l(x)$ 两个变量，就可以做到分块计算整个序列的 softmax

Recomputation（重计算）：在前向传播的过程中不保存整个注意力矩阵，只存储 $m(x),l(x)$ 两个归一化权重；反向传播的过程中基于存储的值按照 tiling 的方式再算一遍。

实现方式基于内核融合，把整个算法实现为单个内核，避免重复读写。

### 算法流程

![](Pasted%20image%2020260410151847.webp)

attn 的计算过程：S = QK^T, P = softmax(S), O = PV，即O=softmax(QK^T)V

SRAM上存储四部分：Q K V O

计算过程是一个双重循环：外层为K V，内层为Q O

外层循环每次从HBM中读取K V矩阵分块的内容存入SRAM用于计算；随后内层循环逐步分块完成对Q矩阵的读取，使用当前的QKV对之前的O进行更新；同时在计算时维护两个中间变量m和l，作为后续计算所需的权重。

![](Pasted%20image%2020260410154818.webp)

### 性能分析

IO复杂度从原本的 $O(N^2+Nd)$ 降低到了 $O(N^2d^2M^{-1})$。

原始attention的瓶颈在把整个注意力矩阵存入HBM中，这是 $O(N^2)$ 的复杂度。flashattn算法中，外层循环共有 $O(NdM^{-1})$ 次，每次都要完整读取一遍Q矩阵，因此每次的访存量为 $O(Nd)$，总体复杂度为两者相乘，即 $O(N^2d^2M^{-1})$。

GPT-2下，虽然GFLOPs更高（66.6->75.2），但访存量大幅减少（40.3GB->4.4GB），因此速度明显提升（41.7ms->7.3ms）

## 稀疏块FlashAttn

采用和上述方法类似的算法，计算如下内容：

$$\begin{align} S&=QK^T, \\ P&=\text{softmax}(S\odot\mathbf{1}_M) \\ O&=PV \end{align}$$

这样可以将IO复杂度进一步压缩至 $O(Nd+N^2d^2M^{-1}s)$，其中 s 是稀疏度。

在长序列情况下，将 s 设置为 $N^{-1/2}$ 或 $N^{-1}\log N$，即可将整体复杂度降低至 $N\sqrt N$ 甚至 $N \log N$ 的级别。（注意：这里只是降低了IO复杂度，实际上的FLOPs复杂度依然是N^2）

## 实验

### 速度比较

以 BERT-large 达到72.0%准确度的时间作为比较对象，训练速度提升了15%（20.0min->17.4min）。

在GPT-2上，速度相比Megatron-LM提升了1.7倍，且能够得到完全相同的困惑度与loss曲线。

使用 vanilla Transformer 在 Long-range Arena 基准上进行测试，在效果几乎不变的情况下，FlashAttn得到了2.4x的加速比。

![](Pasted%20image%2020260410162833.webp)

### 性能比较

计算速度的加快，使得模型可以在长下文更长的数据集上进行训练，从而达到更好的效果。

因此，FlashAttn 使用较基线更长的序列作为训练数据集，发现可以在训练速度更快的情况下，达到更优秀的困惑度。
