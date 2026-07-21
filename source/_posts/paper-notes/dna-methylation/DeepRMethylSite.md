---
title: "[论文笔记] DeepRMethylSite: A Deep Learning based approach for Prediction of Arginine Methylation sites in Proteins"
date: 2023/12/01 18:04
tags:
    - DNA甲基化
categories:
    - 论文笔记
---

论文PDF：[https://pubs.rsc.org/en/content/articlepdf/2020/mo/d0mo00025f](https://pubs.rsc.org/en/content/articlepdf/2020/mo/d0mo00025f)

源代码：[https://github.com/dukkakc/DeepRMethylSite](https://github.com/dukkakc/DeepRMethylSite)

<!--more-->

## Related works

方法：SVM、随机森林、聚类

特征提取手段：PseAAC、香农熵等（人工提取特征）

## 主要内容

蛋白质甲基化预测（有或无），不涉及位点预测

数据集是51bp的蛋白质，正例为中央有甲基化，负例中央无甲基化

初始数据正负样本不平衡，采用sklearn的下采样对负样本进行裁剪

模型架构：

CNN模型+LSTM模型，将两个模型的预测结果加权平均

权重由人工筛选，在(0, 1)的范围内逐个测试，最终选取0.16lstm+0.83cnn

编码方式：onehot和embedding，后者效果更好，映射到33~39位编码

主要贡献：在蛋白质预测领域引入了CNN，以替代先前常用的SVM方法



