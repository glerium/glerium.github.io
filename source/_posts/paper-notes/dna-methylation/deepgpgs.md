---
title: "[论文笔记] DeepGpgs: a novel deep learning framework for predicting arginine methylation sites combined with Gaussian prior and gated self-attention mechanism "
date: 2023/12/11 20:14
tags:
    - DNA甲基化
categories:
    - 论文笔记
---

**任务：蛋白质甲基化预测**

**期刊：Briefing in Bioinformatics（中科院2区）**

<!--more-->

## 特征提取

BLOSUM矩阵：表示蛋白质序列中氨基酸的可替代性，文章用这个来对每个氨基酸进行embedding



## 模型的Attention部分

这里给multi-head attention做了一个shortcut，不过不是简单的x+Attention(x)，而是对原始输入做了个线性变换成为 $\sigma(f(x))+\mathrm{Attention}(x)$

![image.png](image.webp)

## 损失函数

这篇文章损失函数的设置很有意思，一方面考虑到了少部分负样本没有被标注出来的情况（这部分感觉用处不大），另一方面针对正负样本不平衡的问题做了优化（让数量较少的正样本在计算loss时获得更高权重）

![image.png](image%201.webp)



