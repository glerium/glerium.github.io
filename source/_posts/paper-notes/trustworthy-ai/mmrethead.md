---
layout: blog
title: "[论文笔记] Can Retrieval Heads See Images? Multimodal Retrieval Heads in Long-Context Vision-Language Models (2026.05)"
date: 2026/06/29 18:12
tags:
  - 检索头
  - 可解释性
  - 多模态
categories:
  - 论文笔记
---

## 动机

Retrieval head采用大海捞针任务，成功识别出了检索头。但在MM场景下，证据常常以编码方式存在，而不是copy-paste，因此要看问题token对证据区域token的注意力质量。

<!--more-->

## 方法

文中对retrieval score的定义：

$S_h(x)=\frac{1}{|q|}\sum_{t_q\in q}\sum_{g_i\in G_x}\sum_{t=s_i}^{e_i}A_{t_q\to t}^h$

表示平均每个query token对所有groundtruth token的注意力总量。

作者还定义了null-question calibration（空问题校准），即：

$S_h^{cal}(x)=S_h(x)-S_h^{null}(x)$

这便是校准后的retrieval score，可以消除掉模型内部的bias导致部分token天然容易被注意到的问题

## 实验

作者设计了一个多模态版本的大海捞针任务（MM-NIAH），包含以下四种类型：

|任务|needle 类型|测试什么|
|-|-|-|
|Text retrieval|普通文本|模型是否能检索文本证据|
|Image retrieval|图像 needle|模型是否能检索视觉证据|
|Rendered-text retrieval|被渲染成图片的文字|图片里的文字会调用文本头还是图像头|
|Identical image retrieval|是否出现同一张图|检索头可以识别内容，还是单纯做模式匹配|

基准模型包括：

- Qwen2.5-VL 7B / 32B

- Qwen3-VL 8B / 32B

- Gemma3 12B / 27B

把每个注意力头按照校准后的retrieval score排序，选出top-5%的head作为retrieval heads

## MMRetHead的特性

### 稀疏性

计算每个head在所有数据上校准后的retrieval score均值 $\bar s_h$，以及 $s_i=\max (\bar s_i,0)$

然后定义稀疏性指标如下：找出其中校准后retrieval score最高的若干头，其可以解释至少50%的校准后retrieval score，即

$\rho_{0.5}=\frac{1}{|\mathcal{H}|}\min\left\{k:\sum_{i=1}^{k}s_{(i)}\geq0.5\sum_{h\in\mathcal{H}}s_{h}\right\}$

结果发现，大概4.4-10.2%的head贡献了50%以上的正得分，这说明检索头是稀疏的

> 这里与原始retrieval head不同的一点是，retrieval head是通过比较感性的方式来证明的（45%的head得分为0，只有0.3-0.6%的得分大于0.1）；而这篇文章量化定义了稀疏度指标来进行衡量

### 内在性

本文与retrieval head中的结论类似，即：多模态中的检索头也是在预训练阶段形成的，微调阶段几乎不会形成新的检索头

在微调后，Qwen3-VL-8B的46/58个，以及Gemma3-12B-IT中34/39个检索头与微调前重合。

### 动态激活

先前研究发现，不同任务会激活不同检索头。

本文发现，上下文长度也是影响因素之一

![image.png](image.webp)

上下文长度相差较大的设定下，激活的检索头差距较大。

此外，不同上下文设定下，检索头的层级分布也呈现出明显差异。

![image.png](image%201.webp)

较短的上下文倾向于激活浅层head作为检索头，而较长的上下文则涉及较为复杂的语义理解，会激活较深层的head作为检索头。

> 第一次提出这个特性，准备在QRHead的数据集验证一下
待办

## 因果属性

### 遮蔽头的因果性

在将MMRetHead遮蔽后，检索效果大幅下降（相比于random mask）

而且还进行了消融，表明这个结论与上下文长度、证据所在位置、模态均无关。

![image.png](image%202.webp)

### 跨长度的遮蔽实验

在一个长度下检测出的MMRetHead，将其应用于另一个长度的数据下进行测试时进行遮蔽

![image.png](image%203.webp)

结论：虽然不同上下文的head集合会有变化，但是一个长度下的head在其他长度下依然具有很高的重要性。

> 似乎是第一次提出这一结论（待考证）

### 跨数据集的遮蔽实验

用MM-NIAH数据集下检测出的头，放在MMLongBench下进行遮蔽，发现效果48.2%→5.7%；说明具有跨任务的因果重要性，具有良好的泛化性。

## MMRetHead的模态特征

### 文本与图像模态之间的重合性

既存在模态无关的检索头，也存在模态相关的。

![image.png](image%204.webp)

不同上下文长度的交集范围从0.18-0.64；平均值为0.51

### 渲染文本检索的模态倾向

将文本渲染为图像时，检索头的分布更倾向于文本检索

与文本的交集比例为0.79，远高于与图像的0.51的交集比例

说明检索头不仅对模态敏感，还对图像内容敏感。

> projector 图像映射为文本模态

### 检索头分布与图文比例的关系

检索头对图文比例敏感；随着图像比例上升，检索头与文本的重叠逐渐减少。

## 效果

### 多模态文档重排序实验

定义retrieval head score如下：

$R=\frac{1}{|\mathcal{H}_{\mathrm{sel}}|}\sum_{h\in\mathcal{H}_{\mathrm{sel}}}\frac{1}{|q|}\sum_{t_{q}\in q}\sum_{t_{d}\in d_{i}}A_{t_{q}\rightarrow t_{d}}^{h}.$

![image.png](image%205.webp)

page-level：对页面进行排序

layout-level：对布局（表格、图片、段落等）进行排序

## 不足

- 没有考虑不同语言对retrieval head的影响



