---
layout: blog
title: "[论文笔记] ReDeEP: Detecting Hallucination in Retrieval-Augmented Generation via Mechanistic Interpretability (2025.01)"
date: 2026/06/29 18:12
tags:
  - 检索头
  - 可解释性
categories:
  - 论文笔记
---

要解决的问题：RAG模型有时在检索到**正确且相关**的知识时，依然会产生错误回答

→ 分析：内部知识与外部知识的冲突，FFN强调内部知识，而copying head强调外部知识

→ 贡献：提出了redeep，可以解耦内外部知识来进行幻觉检测；以及AARF，通过调节内外部知识的贡献来进行幻觉缓解。

<!--more-->

---

对幻觉发生的归因：

1. 参数化知识过度叠加，且复制头没有有效保留信息

2. LLM在生成过程中丢失了复制头关注的信息

本文的复制头与retrieval head中不同。

---

## 如何衡量内部知识和外部知识

#### 外部上下文

关注两个点 (1) 注意力头是否聚焦于正确的上下文 (2) 在生成过程中能否有效保留这些信息

考虑最后一个 token $t_n$ 的注意力权重，选出权重最大的 top-k% 的token $\mathcal{I}_n^{l,h}$；然后计算 $t_n$ 与这些token之间隐藏状态向量的余弦相似度平均值，即

$\mathcal{E}_n^{l,h} = \frac{e \cdot x_n^L}{\|e\| \|x_n^L\|}, \qquad e = \frac{1}{|\mathcal{I}_n^{l,h}|} \sum_{j \in \mathcal{I}_n^{l,h}} x_j^L.$

该公式即为ECS (External Context Score)，需要注意的是，每个retrieval head都存在对应的ECS

这是token-level的定义，response-level的定义只需要将每个token的ECS取平均值即可

### 参数化知识

参数知识存储于FFN中。为了衡量这部分值，可以把残差流在FFN前后的状态通过LogitLens映射到词汇分布上，然后计算他们的Jensen-Shannon Divergence（JSD），即

$\mathcal{P}_n^l = \operatorname{JSD} \left( q(x_n^{\text{mid},l}) \parallel q(x_n^l) \right),
$

这就是token-level的Parametric Knowledge Score，可以作为模型利用内部知识的衡量。

response-level的PKS定义为每个token的PKS均值。

## 实验

### RQ1：内外部知识的利用与幻觉之间的关系

#### 外部知识

作者测量了同一问题在幻觉情况与非幻觉情况下平均ECS的差值，发现1006/1024个head在生成幻觉时ECS较低；说明幻觉生成时对外部知识的利用较少。

对其进行Pearson相关性检验，发现大部分头上都是正相关。

计算每个head的copying head score，发现与幻觉相关的头通常是复制头；这便导致要么无法集中注意力于有效信息，要么无法有效保留这些信息

![image.png](image.webp)

#### 内部知识

与上节类似，作者计算了幻觉与非幻觉情况下每层（注意这里不是每个head）PCS的均值差值，发现后几层的PCS在幻觉情况下明显偏高，皮尔逊相关系数也正相关。

![image.png](image%201.webp)

将后部分的FFN定义为知识FFN，可以发现过度在知识FFN中添加参数知识易引发幻觉。

### RQ2：从因果角度研究RQ1的相关性

作者采用因果干预方法：在注意力分数中施加噪声，放大残差流中FFN的贡献，对比实验组和对照组中负对数似然损失。发现实验组中loss明显偏大

> 存疑：这不是和训练分布明显不同了吗，感觉损失偏大是正常的

### RQ3：综合分析

![image.png](image%202.webp)

在已知信息且不存在幻觉的情况下，内部知识分数相对更低，模型更加有效利用外部知识来回答问题

## ReDeEP和AARF

### token-level的ReDeEP

幻觉来源于Copying head对外部知识利用不足，以及FFN过度依赖参数知识，这两种情况可以使用PCS, ECS来进行衡量

因此作者定义了一个幻觉分数：

$\mathcal{H}_t(\mathbf{r}) = \frac{1}{|\mathbf{r}|} \sum_{t \in \mathbf{r}} \mathcal{H}_t(t), \qquad \mathcal{H}_t(t) = \sum_{l \in \mathcal{F}} \alpha \cdot \mathcal{P}_t^l - \sum_{l,h \in \mathcal{A}} \beta \cdot \mathcal{E}_t^{l,h},
$



通过在幻觉数据集上进行回归，来拟合出最优参数；当幻觉分数过高时认为出现了幻觉。

这里ECS只考虑Copying head上的分数。

> 感觉这里做的有点草率了，直接对所有layer和head的PCS, ECS进行平均，回归分析的鲁棒性不足

### chunk-level的ReDeEP

token-level对计算资源的要求过大，且准确性存在问题，因此引入了chunk-level的幻觉检测方法。

为了计算指定response chunk的幻觉分数，计算出将其与context chunk所有token之间的平均注意力分数 $W_{i,j}^{l,h}$，然后找出平均注意力最高的context chunk；将其输入到另一个embedding模型中，将两者之间的相似度作为chunk-level的ECS

chunk-level的PCS则可以通过对所有token的PCS取平均得到。

> 为什么chunk-level的ECS不是直接取平均：(1) 耗时太久，Appendix K显示，chunk-level可以实现大概1.7倍的加速 (2) 幻觉常常不是token-level的，例如New York作为一个整体参与计算；从chunk-level进行幻觉的评估更加符合语义 (3) 幻觉token往往只存在于少量token中，这种错误虽然占比较小，但对语义的影响较大，直接取平均会稀释掉幻觉带来的影响

幻觉分数的定义与token-level一致：

$\mathcal{H}_c(\mathbf{r}) = \sum_{l \in \mathcal{F}} \alpha \cdot \tilde{\mathcal{P}}_\mathbf{r}^l - \sum_{l,h \in \mathcal{A}} \beta \cdot \tilde{\mathcal{E}}_\mathbf{r}^{l,h}.
$

### AARF方法

首先计算token-level的幻觉分数，如果其大于 $\tau$，则增强Attn模块的贡献，同时减弱FFN的贡献：

$f(\mathbf{x}) = \sum_{l=1}^L \sum_{h=1}^H \widehat{\mathrm{Attn}}^{l,h} \left( \mathbf{X}_{\leq n}^{l-1} \right) \mathbf{W}_U + \sum_{l=1}^L \widehat{\mathrm{FFN}}^l \left( \mathbf{x}_n^{\mathrm{mid},l} \right) \mathbf{W}_U + \mathbf{x}_n \mathbf{W}_U$

$\begin{align*}
\widehat{\mathrm{Attn}}^{l,h}(\cdot) &= \begin{cases} 
\alpha_2 \cdot \mathrm{Attn}^{l,h} \left( \mathbf{X}_{\leq n}^{l-1} \right), & \text{if } (l,h) \in \mathcal{A}, \\ 
\mathrm{Attn}^{l,h} \left( \mathbf{X}_{\leq n}^{l-1} \right), & \text{otherwise} 
\end{cases} , \\
\widehat{\mathrm{FFN}}^l (\cdot) &= \begin{cases} 
\beta_2 \cdot \mathrm{FFN}^l \left( \mathbf{x}_n^{\mathrm{mid},l} \right), & \text{if } l \in \mathcal{F}, \\ 
\mathrm{FFN}^l \left( \mathbf{x}_n^{\mathrm{mid},l} \right), & \text{otherwise.} 
\end{cases}
\end{align*}$

这里 $\alpha_2$ 大于1，且 $\beta_2$ 小于1，为两个超参数

## 实验

作者分别对ReDeEP和AARF进行了实验；

前者在RAGTruth和Dolly两个数据集上进行实验，任务为幻觉输出判断；

后者则在这两个数据集上采用AARF进行生成实验，使用GPT-4o进行自动评估；

![image.png](image%203.webp)

## 实验复现

Dolly数据集 Llama3-8b

|原文结果|AUC|PCC|Rec|F1|
|-|-|-|-|-|
|ReDeEP(token)|0.6701 |0.2421 |0.8293 |0.6901|
|ReDeEP(chunk)|0.7354  |0.3652 |0.8392|0.7100|

|复现结果|AUC|PCC|Rec|F1|
|-|-|-|-|-|
|ReDeEP(token)|0.648|0.236|0.714|0.633|
|ReDeEP(chunk)|0.585|0.089|0.800|0.566|

发现：代码实现中，dolly数据集在幻觉检测中只使用了top-1的copy head；将其改为top-2的copy head会让性能劣化

|Top-2 head复现|AUC|PCC|Rec|F1|
|-|-|-|-|-|
|ReDeEP(token)|0.584|0.089|0.800|0.560|

尝试对这两个头进行按照与幻觉的相关性进行加权（相关性越高权重越大）

|Top-2加权|AUC|PCC|Rec|F1|
|-|-|-|-|-|
|ReDeEP(token)|0.582|0.089|0.800|0.566|

对原因进行分析，发现不同头之间的权重相差较小（两个头的相关性仅相差不到10%）

此外还对RAGTruth数据集进行了复现，结果也与文章有差距

|原文结果|AUC|PCC|Rec|F1|
|-|-|-|-|-|
|ReDeEP(token)|0.7522 |0.4493 |0.7984 |0.7132|
|ReDeEP(chunk)|0.7285 |0.3964 |0.7819 |0.6947|

|复现结果|AUC|PCC|Rec|F1|
|-|-|-|-|-|
|ReDeEP(token)|0.6623|0.2965|0.5315|0.5673|
|ReDeEP(chunk)|0.7028|0.3471|0.6255 |0.6751|

> 找一下follow redeep的文章



