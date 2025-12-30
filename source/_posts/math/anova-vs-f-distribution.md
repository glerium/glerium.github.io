---
title: 方差分析与F分布
date: 2024/04/27 22:16:48
tags:
    - 数理统计
categories:
    - 课程笔记
---
# 方差分析简介

方差分析（Analysis of Variance，ANOVA）是一种检验方法，可以用来比较两个及以上个样本之间的平均值是否存在显著差异。 由于各种因素的影响，研究所得的数据呈现波动状。造成波动的原因可分成两类，一种是不可控的随机因素（比如实验误差等，服从于某种正态分布），另一种是研究中施加的对结果形成影响的可控因素。

<!--more-->

方差分析的目的就在于检验某种可控因素的不同水平是否会对实验结果造成影响。

如果方差分析只针对一个因素进行，称为单因素方差分析；如果同时针对多个因素进行，则称为多因素方差分析。

# 单因素方差分析

假设我们要检验的因素为A，它有 $k$种水平，记作 $A_1,~A_2,~\ldots,~A_k$，且在每种水平下样本满足正态分布 $\mathcal{N}(\mu_i,\sigma_i^2)$。在此基础上我们做方差齐性假设 $\forall i,~\sigma_i^2=\sigma^2$。我们现在要假设因素A的不同水平是否会影响到样本均值，即

$$\begin{aligned}
H_0:&~\mu_1=\mu_2=\cdots=\mu_k \\
H_1:&~\exists i,j,~ ~\text{s.t.} ~\mu_i \neq \mu_j
\end{aligned}$$

我们构造如下的统计量，记总误差平方和为

$$SST  = \sum_{i=1}^{k}\sum_{j=1}^{n_i}\left(x_{ij}-\bar{x}\right)^2$$

组间误差平方和为

$$SSA = \sum_{i=1}^{k}n_i(\bar{x}_i-\bar{x})^2$$

组内误差平方和为

$$SSE  = \sum_{i=1}^{k}\sum_{j=1}^{n_i}\left(x_{ij}-\bar{x}_i\right)^2$$

显然有 

$$SST=SSA+SSE$$

$SSA$ 与 $SSE$ 相独立，且

$$\dfrac{SSE}{\sigma^2}\sim\mathcal{X}^2(n-k)$$

若原假设 $H_0$成立，则还有

$$\dfrac{SSA}{\sigma^2}\sim\mathcal{X}^2(k-1)$$

因此我们构造 F 统计量：

$$F=\dfrac{SSA/(k-1)\sigma^2}{SSE/(n-k)\sigma^2}=\dfrac{SSA/(k-1)}{SSE/(n-k)}\sim F(k-1,n-k)$$

我们可以通过查 F 分布表，当 $F \le F_{1-\alpha}(k-1,n-k)$接受 $H_0$；否则拒绝 $H_0$。

# 多因素方差分析

多因素方差分析的定义可以基于单因素方差分析来理解：

单因素方差分析是在只考虑一个自变量的情况下，当自变量取不同值时，样本的各分类组之间是否有差异；

多因素方差分析则是同时考虑多个自变量：

$$\left({factor}_1,~ {factor}_2,~\ldots,~{factor}_n\right)$$

$$factor_i=\left\{v_{i,1},~v_{i,2},~\ldots,~ v_{i,n_i} \right\}$$

则此时样本可被划分为互不相交的 $\prod_{i=1}^{n}n_{i}$个分类组；基于这些分类组我们仍可仿照单因素方差分析对每个因素进行单因素方差分析，更重要的是我们可以对这些分类组取适当的交并补集，从而构造合适的新分类组来研究不同因素间的相互作用是否显著（注意这是一种定性的研究，只能给出 是/否 的结论，而无法量化因素A对因素B的影响）。

接下来对这个问题进行建模：

$$\text{sample:}~~ y_{(v_1,~v_2,~\ldots,~v_n)} = μ+ \sum f_i+ \varepsilon$$

$\sum f_i$ represents all related factors' prime influence containing interactive and non text-interactive; $\varepsilon$ represent observation error.

上面的式子是基于如下假设：

样本取值的变动范围可以分解为变量单独影响、变量相互影响（注意相互是一种二元关系，事实上应当还要考虑多元的情况）和随机误差。

所以上式简化后从样本的角度可写成如下形式：

$$	SST=SSA+SSB+SSAB+SSE$$

$$SST=\sum_{q=1}^{n_j}\sum_{p=1}^{n_i}\sum_k^{n_{pq}}\left( 
y_{(v_{i,p},v_{j,q})}-\bar{y} \right)^2$$

其中 $n_{pq}$为该水平组合下的样本数量。

$$SSA=\sum_{p=1}^{n_i}\sum_{q=1}^{n_j}n_{pq}\cdot\left( \bar{y}_{(v_{i,p})} - \bar{y} \right)^2$$

其中 $\bar{y}_{(v_{i,p})}$为该水平组合下的样本均值。

$SSB$同理。

$$SSE = \sum_{q=1}^{n_j}\sum_{p=1}^{n_i}\sum_{k}^{n_{pq}}\left( 
y_{(v_{i,p},v_{j,q}),k} - \bar{y}_{(v_{i,p},v_{j,q})} \right)^2$$

四个方差也有对应的自由度：

$$\begin{aligned}
d_t&=\ n_i\cdot n_j\cdot n_{pq}-1 \\
d_a&=\ n_i-1 \\
d_b&=\ n_j-1 \\
d_e&=\ n_i\cdot n_j\cdot \left(n_{pq}-1\right) \\
\end{aligned}$$

从而可以构造F检验量：

$$\begin{aligned}
F_A &= \dfrac{SSA / (k-1)}{SSE / kr(l-1)} = \dfrac{MSA}{MSE} \\
F_B &= \dfrac{SSB/(r-1)}{SSE / kr(l-1)} = \dfrac{MSB}{MSE} \\ 
F_{AB} &= \dfrac{SSAB/(k-1)(r-1)}{SSE/kr(l-1)} = \dfrac{MSAB}{MSE}
\end{aligned}$$

然后查F分布表得出P值，与显著性水平进行比较：

  $P>\alpha$接受；$P \le \alpha$拒绝。
