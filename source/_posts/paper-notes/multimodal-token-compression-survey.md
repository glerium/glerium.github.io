---
title: "[论文笔记] When Tokens Talk Too Much: A Survey of Multimodal Long-Context Token Compression  across Images, Videos, and Audios"
date: 2025/12/14 22:08:30
tags:
    - LLM
    - Token压缩
categories:
    - 论文笔记
---
## Token压缩的动机

- self-attention二次方复杂度带来的计算挑战 → 加速推理和减少内存消耗

    - 在MLLM中更加明显，visual / audio数据的token数量比纯文本高几个数量级

    ![image.webp](image.webp)

## 为什么有效

多模态数据中有大量信息冗余（图片：相邻像素；视频：周围的重复帧；音频：相近时间与相近频谱之间的冗余）

例如，高分辨率图像包含很强的局部相关性，而视频流在帧之间具有广泛的时空冗余，音频信号通常包含扩展的静默段或平稳噪声。

## 优势

- 加速推理，减少内存消耗

- 作为后训练方法，无需重新训练模型

## 现有方法分类

- 基于模态分类：image-centric, video-centric, audio-centric

- 基于机制分类：

    - transformation-based：快速，但压缩效果通常一般（~25%）

        - pixel-shuffle (training-free)

        - pooling / interpolation (training-free)

        - convolution

    - similarity-based：较为灵活，但可能丢失信息

    - attention-based：灵活且运算量较小，但是和FlashAttention / KV Cache加速库冲突，反而导致速度降低

    - query-based

## 流程

- importance identification / 重要性识别

- redundancy quantification / 冗余量化

- token merging or pruning / token合并或修剪

