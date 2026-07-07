---
layout: blog
title: "[论文笔记] Query-Focused Retrieval Heads Improve Long-Context Reasoning and Re-ranking (2025.09)"
date: 2026/06/15 17:00
tags:
  - 检索头
  - 可解释性
categories:
  - 论文笔记
---

QRHead是对retrieval head的改进。

## 动机

先前的retrieval head解释了LLM的检索机制，但仍存在两个问题：

1. copy-paste目标过于简单，鲁棒性不足

2. 采用的大海捞针任务是合成数据，与真实语言场景下数据分布不一致

<!--more-->

因此QRHead针对性地提出了两点改进：

1. 识别qrhead时基于query-context评分函数，通过相关性来衡量每个head的注意力质量

2. 采用真实世界的数据进行运行，同时发现方法仅需较少的数据即可运行

![image.png](image.webp)

## 对QRHead的定义

首先定义head $h$对doc $i$的QRScore：

$\text{QRScore}_h(q,d_i)=\frac{1}{|q|}\sum_{t_q\in q}\sum_{t_d\in d_i}A_h^{t_q \to t_d}$

这表示head为doc i分配的注意力比例。

然后定义head h本身的QRScore，即：

$\text{QRScore}_h(q)=\sum_{d_i\in D^*}\text{QRScore}_{h}(q,d_i)$

这里的 $D^*$ 表示所有 groundtruth 证据集。head的QRScore表示它对所有正确证据分配的注意力权重。

为了检测QRHead，我们需要一个真实数据集 

$\mathcal{T}=\{(q,D,D^*_{q})\}$

在这个数据集上计算每个head对应的QRScore，然后取top-k分数的head即为正确答案。

## QRRetriever的构建

在识别出模型对应的QRHead之后，我们可以构建一个reranker。

具体来说，对于一个query q和文章集合D，我们可以计算每篇文章和q的QRscore平均值：

$\mathcal{R}(q,d_i)=\frac{1}{|\mathcal H_{\text{select}}|}\sum_{h\in \mathcal H_{\text{select}}} \text{QRScore}(q,d_i)$

这表示了所有QRHead在query q的条件下，对这篇文章的注意力平均值，然后按照R降序排列文档即可。

## 实验

作者基于QRHead构建了QRRetriever，可以进行段落的reranking任务。

基于QRRetriever进行了两种类型的实验：长文本多跳推理。

- 长文本多跳推理：把长上下文拆分成若干个小的chunk，通过QRRetriever重排序后取top-k输入模型

    - 数据集：LongMemEval、CLIPPER

    - 指标：Recall、End-to-end Accuracy

- 段落重排序

    - 指标：nDCG（推荐系统中衡量排序质量的指标）

- 在这两种任务上，QRHead方法都优于full attention head和retrieval head方法。

## 复现实验

发现：在Llama-3.1-70B-Instruct模型中，大部分QRHead都集中在35层的位置（总共80层）

