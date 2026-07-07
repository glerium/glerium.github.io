---
title: "Ascend C核间同步控制"
date: 2026-07-07 15:55:00
categories:
    - 学习笔记
tags:
    - AI Infra
    - Ascend C
	- 算子开发
---

## CrossCoreSetFlag & CrossCoreWaitFlag

用于支撑核间的同步控制。

```cpp
template <uint8_t modeId, pipe_t pipe>
__aicore__ inline void CrossCoreSetFlag(uint16_t flagId)
```

与SetFlag & WaitFlag 类似，pipe用于指定数据流之间的依赖，flagId表示事件类型；此外还新增了modeId用于指定模式。

<!--more-->

### 模式分类

| 模式ID | 介绍                                           |
| ---- | -------------------------------------------- |
| 0    | AI Core核间的同步控制                               |
| 1    | AI Core内部，AIV之间的同步控制                         |
| 2    | AI Core内部，AIC与AIV之间的同步控制。两个AIV是一个整体。         |
| 4    | AI Core内部，AIC与AIV之间的同步控制。AIV0与AIV1可单独触发AIC等待 |

- 模式0：AI Core核间的同步控制。对于AIC场景，同步所有的AIC核，直到所有的AIC核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行；对于AIV场景，同步所有的AIV核，直到所有的AIV核都执行到CrossCoreSetFlag时，CrossCoreWaitFlag后续的指令才会执行。
- 模式1：AI Core内部，AIV核之间的同步控制。如果两个AIV核都运行了CrossCoreSetFlag，CrossCoreWaitFlag后续的指令才会执行。
- 模式2：AI Core内部，AIC与AIV之间的同步控制。在AIC核执行CrossCoreSetFlag之后， 两个AIV上CrossCoreWaitFlag后续的指令才会继续执行；两个AIV都执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。
- 模式4：仅支持A5芯片。AI Core内部，AIC与AIV之间的同步控制。AIV0与AIV1可单独触发AIC等待。比如，在AIC核执行CrossCoreSetFlag之后， AIV0上CrossCoreWaitFlag后续的指令才会继续执行；AIV0执行CrossCoreSetFlag后，AIC上CrossCoreWaitFlag后续的指令才能执行。

### 调用示例

```cpp
if (g_coreType == AscendC::AIC) {
    AscendC::CrossCoreSetFlag<0x2, PIPE_FIX>(0x8);
}

// AIV侧等待AIC Set消息, 进行Vector后处理
if (g_coreType == AscendC::AIV) {
    AscendC::CrossCoreWaitFlag(0x8);
}
```

数据路径：AIC计算后存储在L0C，随后调用FIXP从L0C传输到UB上，传输完成后通过PIPE_FIX激活SetFlag；AIV等待Flag，随后在VEC上进行后处理
## 来源

* [CrossCoreSetFlag(ISASI)-CANN社区版9.0.X-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0273.html)
* [CrossCoreWaitFlag(ISASI)-CANN社区版9.0.X-昇腾社区](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900/API/ascendcopapi/atlasascendc_api_07_0274.html)