---
layout: blog
title: "[论文笔记] Language Models are Unsupervised Multitask Learners (GPT-2)"
date: 2025.07.17 00:00:00
tags:
    - LLM
categories:
    - 论文笔记
---

### 背景

- 现有的模型大多基于较小的语料，而且是针对特定任务的

- 从meta-learning的角度来看，multi-task learning需要相当多的 (task, dataset) 组合作为train sample，这是不现实的

### 创新点

- 提出了一个更大的LM架构，包含三种大小：GPT-1、BERT、GPT-2

- 提出了一个LM语料库，从Reddit的优质贴外链汇总得到

  - 为了避免train set和test set重叠，删除了Wikipedia的网页

- 仅在大型语料库采用pretrain的情况下，发现模型能天然地适配于各种任务，而无需微调

  - 一个无监督的模型在许多任务上取得了SOTA效果

- 证明了基于极大似然估计的语言模型能够拟合现实任务

### 缺点

- 在部分任务上，GPT-2的定量性能依然较差，不具备现实应用的能力

### 架构

- 大致与GPT-1相同，但有几点改进

  - Layer normalization从post-norm改为了pre-norm，这样可以提高训练的稳定性

  - 每一层的initialization参数都进行了缩放，变为先前的 1/sqrt(n)，这里n是层数

    - 目的：为每层的初始化方差会累加，这样可以让方差稳定，防止梯度出现问题

  - 词库和context size都有提升

- 编码器：BPE, byte pair encoder

  - 这是一种位于字符级和单词级语言建模之间的编码方式，对于频繁出现的语素进行word-level的编码，反之则采用byte-level编码
