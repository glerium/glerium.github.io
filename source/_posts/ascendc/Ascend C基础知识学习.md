---
title: "Ascend C基础知识学习"
date: 2026-07-07 15:12:00
categories:
    - 学习笔记
tags:
    - AI Infra
    - Ascend C
    - 算子开发
---

开发模型：Host-Device

编程模型：
* SIMD：单指令多数据（matmul等）
* SIMT：单指令多线程（复杂分支控制等，仅A5支持）

存储单元：
* AI Core内部存储：LocalTensor
* 外部存储：GlobalTensor

<!--more-->

DMA可以实现GM与LM之间，以及不同层级LM之间的搬运

AI Core的异步指令序列是通过Scalar计算单元下发的。

## 来源

资料整理自公开文档：
* [异构并行编程模型-CANN社区版9.0.X-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_00028.html)
* [编程模型概述-CANN社区版9.0.X-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/programug/Ascendcopdevg/atlas_ascendc_10_10062.html)