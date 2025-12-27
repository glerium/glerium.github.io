---
title: "[论文笔记] iDNA-ABF: multi-scale deep biological language learning model for the interpretable prediction of DNA methylations"
date: 2023/11/22 11:00:50
tags:
    - DNA甲基化
categories:
    - 论文笔记
---
### Method

给定一组固定长度的DNA序列，进行有无甲基化的预测，输出一个binary概率

![image.webp](image.webp)

网络架构：

- 从3mers和6mers两个角度对序列进行tokenize，分别进两个dnabert预训练模型，输出两个features：F1和F2

- 对F1和F2进行特征融合（加权平均，权值是可学习的参数）

- 融合后的features进全连接层做二分类

### 可解释性

- 从多个数据集的ACC Recall Precicion AUC MCC角度，对不同模型进行对比

- **采用Uniform Manifold Approximation and Projection (UMAP)对不同模型对DNA序列的聚类结果进行可视化，和其他模型进行对比**

![image.webp](image%201.webp)

- **跨物种对比Accuracy，反映进化路径相似的物种之间甲基化结构的相似性**

![image.webp](image%202.webp)

- **cross-species validation，用一个物种训练，另一个物种预测，反映出的结论和上条类似**

![image.webp](image%203.webp)

- **用attention map反映出不同k-mers在特征提取中的差异性，以此证明采用3mers+6mers构建模型能提取到更多信息。**

    - 3mers特征“learns more local discriminative information as compared to that before training”，而6mers特征“more focused on global information after training”

![image.webp](image%204.webp)

- 验证不同尺度的数据集对结果的影响

![image.webp](image%205.webp)

