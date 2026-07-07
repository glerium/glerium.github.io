---
title: "Ascend C核内同步控制"
date: 2026-07-07 15:27:00
categories:
    - 学习笔记
tags:
    - AI Infra
    - Ascend C
	- 算子开发
---

## 指令流水分类

Ascend的指令流水类型分为以下几种：
* PIPE_S：scalar流水（如Tensor.GetValue）
* PIPE_V：vec计算
* PIPE_M：cube计算
* PIPE_MTE1：LM Tensor内部搬运（L1->L0A等）
* PIPE_MTE2：GM到LM的搬运（GM->L1等）
* PIPE_MTE3：LM到GM的搬运
* PIPE_FIX：FixPipe，从L0C向外的搬运

<!--more-->

## 同步控制分类

* 多流水同步：通过SetFlag和WaitFlag实现
	* SetFlag：前序指令的数据读写全部执行完毕后执行，设置硬件标志位为1
	* WaitFlag：等待直到对应标志位变为1，然后重置为0
* 单流水同步：PipeBarrier，阻塞指令流直到前序指令读写全部完成

## 数据流举例

以纯vec操作为例，其数据流包括三个阶段：
1. DMA(VECIN) 将数据从GM搬运到LM
2. 在LM上调用vec单元进行计算
3. DMA(VECOUT) 将数据从LM搬运到GM

## SetFlag & WaitFlag

同一个核内部不同流水之间的同步指令，用于具有数据依赖的不同流水指令。

```cpp
template <HardEvent event>
__aicore__ inline void SetFlag(int32_t eventID)
template <HardEvent event>
__aicore__ inline void WaitFlag(int32_t eventID)
```

调用示例：

```cpp
AscendC::GlobalTensor<half> dstGlobal;
AscendC::LocalTensor<half> dstLocal;
dstLocal.SetValue(0, 0);
uint32_t dataSize = 512; 
// 静态Tensor编程场景中，eventID由开发者自行管理
AscendC::SetFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
AscendC::WaitFlag<AscendC::HardEvent::S_MTE3>(EVENT_ID0);
AscendC::DataCopy(dstGlobal, dstLocal, dataSize);
```

这份代码里展示了一个MTE3对Scalar的依赖；DataCopy指令（PIPE_MTE3）需要等待SetValue（PIPE_S）结束之后才能运行；所以对应的HardEvent是S_MTE3。

## PipeBarrier

这个指令控制的是具有数据依赖的相同流水（如PIPE_V与PIPE_V之间的依赖）

```cpp
template <pipe_t pipe>
__aicore__ inline void PipeBarrier()
```

调用示例：

```cpp
AscendC::LocalTensor<half> src0Local;
AscendC::LocalTensor<half> src1Local;
AscendC::LocalTensor<half> src2Local;
AscendC::LocalTensor<half> dst0Local;
AscendC::LocalTensor<half> dst1Local;

AscendC::Add(dst0Local, src0Local, src1Local, 512);
AscendC::PipeBarrier<PIPE_V>();
AscendC::Mul(dst1Local, dst0Local, src2Local, 512);
```

这里计算了 src2 * (src0 + src1)，乘法依赖于加法运算执行。

不过上述代码只是一个示例，实际上编译器会自动控制VEC上的信号量，无需手动插入PIPE_V同步。

## 来源

资料整理自公开文档：
* [同步控制简介-CANN社区版9.0.X-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0179.html)
* [SetFlag/WaitFlag(ISASI)-CANN社区版9.0.X-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0270.html)
* [PipeBarrier(ISASI)-CANN社区版9.0.X-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0271.html)
