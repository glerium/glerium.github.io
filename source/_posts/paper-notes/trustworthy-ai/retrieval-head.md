---
layout: blog
title: "[论文笔记] Retrieval Head Mechanistically Explains Long-Context Factuality (2024.04)"
date: 2026/06/08 15:20
tags:
  - 检索头
  - 可解释性
categories:
  - 论文笔记
---

检索头较早的一篇文章。发现了检索头的一些性质：

- 普遍性：所有模型都有检索头

- 稀疏性

- 固有性：不随continual pretrain而改变

- 动态激活：一部分检索头始终激活，另一部分随机激活

- 因果性：敲除会导致无法检索上下文信息，但对于只需要内部知识的场景影响较小。
