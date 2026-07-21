---
title: "[论文笔记] iDNA-ABF: multi-scale deep biological language learning model for the interpretable prediction of DNA methylations"
date: 2023/12/01 20:04
tags:
    - DNA甲基化
categories:
    - 论文笔记
---

期刊：Bioinformatics（IF=5.8，中科院三区）

目标：人类体内的蛋白质磷酸化预测

主要贡献：采用蛋白质相互作用作为特征（PPI, Protein-protein interaction），向神经网络引入了新信息，我们在DNA/RNA甲基化预测时可以考虑借鉴

<!--more-->

## PPI特征提取

首先从STRING Database下载人类体内所有蛋白质种类和它们之间的相互作用信息，每条数据可视为 $(protein_x, ~protein_y, ~score)$，代表两种蛋白质之间的相互作用强度。把蛋白质视为点，相互作用视为边，可以构建出一张无向图出来。随后采用graph embedding方法，将每个蛋白质embed为一个128维向量，作为onehot以外另一维度上的特征。

## 网络架构

![image.png](image.webp)

如图所示。值得关注的一点是网络采用bilinear features的思想进行特征融合，后续可以考虑使用BAN（bilinear attention）生成联合特征，此外这里在fusion之前没有做self-attention，添加以后效果可能会更好。还有一点可以尝试改进的是在网络中引入self-attention/transformer-encoder做特征提取

