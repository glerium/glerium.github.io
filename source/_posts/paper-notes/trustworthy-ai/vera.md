---
layout: blog
title: "[论文笔记] VERA: Identifying and Leveraging Visual Evidence Retrieval Heads in Long-Context Understanding (2026.02)"
date: 2026/06/08 15:21
tags:
  - 检索头
  - 可信生成
categories:
  - 论文笔记
---

## 动机

1. MLLM的在长上下文的检索能力难以解释

2. 在长上下文下，相比纯文本场景，多模态场景下模型的检索能力变弱

  1. 图片分辨率越低，性能降低越多（DeepSeek-OCR）

<!--more-->

## 贡献

### 发现并识别了VER Heads

在模型的 retrieval-moment（输出token熵值最高的时刻）导出每个attention-head在的注意力权重。考虑证据块被每个head覆盖的注意力评分之和（每个patch中证据块的面积占比 * 注意力权重）作为视觉检索分数。取检索分数最高的 top-k 个 head，认为他们是VER heads。

  在模型检索证据的时刻，假如某个head对于证据部分的注意力强度很高，则认为它是负责做检索的head，这确实是一个直观的理解。但存在两个问题：第一，为什么说输出token熵值最高的时刻就是 retrieval-moment？代码中实际也没有实现熵值这个东西。第二，相关性不代表因果性，也许他们只是刚好在检索过程中被激活，并负责对长上下文的语义理解等事情。感觉这里单纯取 top-k 的做法还是比较简单。

### VER Heads的性质

- 在整个模型中比较稀疏，只占<1.65%；检索分数呈现出长尾分布。

- 在不同数据集之间较为保守；不同数据集之间分数的相关系数较高，而且取出的VER Heads很相似。

- 从效果看，遮蔽掉VER Heads后，模型效果大幅下降。而遮蔽OCR Heads或随机头则不会产生类似效果。

- VER Heads的对应分数在微调后，会随着模型性能的提升而更加集中

### 将检索头用于辅助模型性能提升

3. 离线阶段：对某个特定的模型，遍历数据集，识别出VER Heads

4. 在线阶段

  1. 将image+query输入模型，根据retrieval-moment的attention分布找出evidence

  2. 将evidence输入OCR模型转成文字，将image+text+query输入模型得出答案

优化点：提出了一个可以自动识别evidence位置的方法；将evidence转成文字的手段可以有效弥补MLLM在多模态场景下进行检索的不足
