---
title: "GroupedMatmulV4 A5 MXFP8 调度源码导读"
date: 2026-07-24 11:18
categories:
    - 学习笔记
tags:
    - AI Infra
    - Ascend C
    - Ascend 950
    - 算子开发
    - 含AI生成
toc:
  number: false
---

> 本文聚焦 `aclnnGroupedMatmulV4` 在 A5/DAV_3510 上的 MXFP8 主路径：
> `x/weight=FLOAT8_E4M3FN`、`scale/perTokenScale=FLOAT8_E8M0`、weight 为 ND，
> `transA=false`、`transB=false`。其他 format、转置方式和量化模式只在必要时用于对比。

<!--more-->

## 1. 先看全局

这条路径可以分为三层：

1. Host 侧识别 A5 MXFP8 场景，计算 `baseM/baseN/baseK`、L1 流水和 tiling key；
2. Kernel 侧遍历 group，将每个 group 的 M/N 输出 tile 分配给多个 AI Core；
3. 每个 AI Core 对自己的 M/N tile 遍历完整 K，通过 L1、L0A/L0B 分块和多次 MMAD 在 L0C 上累加。

主要调用链：

```text
aclnnGroupedMatmulV4GetWorkspaceSize
  └─ aclnnGroupedMatmulGetWorkspaceSizeCommon
      └─ l0op::GroupedMatmul
          ├─ InferShape
          └─ ADD_TO_LAUNCHER_LIST_AICORE
              └─ TilingGMM
                  └─ GroupedQmmBasicApiTiling       Host MX tiling
                      └─ grouped_matmul             A5 kernel 入口
                          └─ GmmCgmctMxKernel       组装模板
                              └─ KernelQGmmMx      group 和 MN 分核
                                  └─ BlockMmadMx   核内 K 流水和 MMAD
```

关键入口：

| 层级 | 代码 | 职责 |
| --- | --- | --- |
| V4 API | [`aclnn_grouped_matmul.cpp`](../op_api/aclnn_grouped_matmul.cpp#L3113) | 参数校验、构造 executor |
| Host 总入口 | [`TilingGMM`](../op_host/op_tiling/grouped_matmul_tiling.cpp#L2240) | 架构和 dtype 分流 |
| MX tiling | [`GroupedQmmBasicApiTiling`](../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp#L21) | 计算 MX Basic API tiling data |
| Kernel 总入口 | [`grouped_matmul`](../op_kernel/grouped_matmul_apt.cpp#L222) | 按编译宏和 tiling key 选择模板 |
| Kernel 调度 | [`KernelQGmmMx`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L55) | group 遍历和 MN tile 分核 |
| Cube 计算 | [`BlockMmadMx`](../../common/cgmct/block/block_mmad_mx.h#L69) | GM/L1/L0 搬运、MX scale、MMAD、写回 |

V4 只影响 ACLNN 入口的校验和 executor 构造。进入 `l0op::GroupedMatmul`
后，V4 与其他版本共用同一套 `GroupedMatmul` tiling 和 kernel。
严格说 ACLNN 是两阶段接口：
[`aclnnGroupedMatmulV4GetWorkspaceSize`](../op_api/aclnn_grouped_matmul.cpp#L3113)
完成校验、建图并生成 executor，
[`aclnnGroupedMatmulV4`](../op_api/aclnn_grouped_matmul.cpp#L3292)
再用 workspace 和 executor 发起执行。Host tiling 属于这套 executor/算子执行链路，
不是 `aclnnGroupedMatmulV4` 函数体里的一次普通 C++ 直接调用。

---

## 2. 文中使用的例子

以典型 MoE split-M 场景为例：

```text
groupNum = 4
groupType = 0                  // SPLIT_M
groupListType = 0              // cumsum
groupList = [10, 30, 30, 45]

K = 4096
N = 512
baseM = 128
baseN = 256
baseK = 128
```

`groupList` 是累积值，因此四组的 M 为：

```text
M0 = 10 - 0  = 10
M1 = 30 - 10 = 20
M2 = 30 - 30 = 0
M3 = 45 - 30 = 15
```

数学上是四个 grouped GEMM problem：

```text
Group 0: X0[10,4096] @ W0[4096,512] -> Y0[10,512]
Group 1: X1[20,4096] @ W1[4096,512] -> Y1[20,512]
Group 2: empty, skip
Group 3: X3[15,4096] @ W3[4096,512] -> Y3[15,512]
```

`groupNum` 可以类比 batch size，但它更准确的含义是“grouped GEMM problem 的数量”。
它不是普通 BMM 中每组 shape 都固定的 batch 维；例如 split-M 时每个
expert 的 `M_i` 可以不同。

---

## 3. Host 侧：如何选中 A5 MXFP8 tiling

### 3.1 `TilingGMM` 进入 A5 量化模板

[`TilingGMM`](../op_host/op_tiling/grouped_matmul_tiling.cpp#L2240) 先取出 `x/weight`
第一个动态输入的 descriptor，得到 dtype。在 `DAV_3510` 上，两个输入都是
1 Byte 类型时归入全量化路径：

`xDesc` 就是 `x[0]` 的 tensor descriptor，保存 dtype、format、shape 等
Host 侧元数据；它不是 `x` 的 GM 数据地址。这里只通过
`xDesc->GetDataType()` 用它做模板分流。

```cpp
if (compileInfoPtr->npuArch == NpuArch::DAV_3510) {
    bool isQuant = ... ||
        (GetSizeByDataType(xDType) == 1 &&
         GetSizeByDataType(weightDtype) == 1);
    if (isQuant) {
        std::vector<int32_t> registerList = {0, 1};
        return TilingRegistry::GetInstance().DoTilingImpl(context, registerList);
    }
}
```

`{0,1}` 对应的注册在
[`grouped_quant_matmul_tiling.cpp`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp#L1413)：

```cpp
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmul, GroupedQmmBasicApiTiling, 0);
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmul, GroupedQmmTiling,         1);
```

注册表按优先级尝试模板。`scale=FLOAT8_E8M0` 时，
[`GroupedQmmBasicApiTiling::IsCapable`](../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp#L27)
识别到 MXFP8，模板 0 返回成功，不再尝试通用模板 1。

### 3.2 tiling 框架的执行顺序

`GroupedQmmBasicApiTiling` 继承自 `TilingBaseClass`。实际流程由
[`TilingBaseClass::DoTiling`](../../../common/include/op_host/tiling_base.h#L99) 组织：

```text
GetShapeAttrsInfo
  ↓
GetPlatformInfo
  ↓
IsCapable
  ↓
DoOpTiling
  ↓
DoLibApiTiling
  ↓
GetWorkspaceSize
  ↓
PostTiling
  ↓
GetTilingKey
```

各阶段的职责：

| 阶段 | 主要工作 |
| --- | --- |
| `GetShapeAttrsInfo` | 解析 dtype、format、transpose、groupType、groupList 和 M/N/K |
| `GetPlatformInfo` | 读取 AIC 核数以及 L1/L0A/L0B/L0C 容量 |
| `IsCapable` | 判断当前模板是否支持这组输入 |
| `DoOpTiling` | 写入 group、量化模式、bias 等算子级参数 |
| `DoLibApiTiling` | 计算 Cube 基础块和 L1 流水，是主要 tiling 逻辑 |
| `PostTiling` | 将结果序列化到 kernel 可读的 raw tiling buffer |
| `GetTilingKey` | 根据 `transB/transA/kernelType` 选择 kernel 编译分支 |

`GetShapeAttrsInfo` 的公共实现依次调用：

- [`AnalyzeDtype`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp#L263)：解析 X/W/Y/scale 的 dtype 和 format；
- [`AnalyzeAttrs`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp#L99)：解析 transpose、groupType 等属性；
- [`AnalyzeInputs`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp#L687)：计算 groupNum、M/N/K，识别量化模式并校验 scale shape。

`scaleDtype == FLOAT8_E8M0` 使
[`SetQuantMode`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp#L768) 设置：

```text
aQuantMode = MX_PERGROUP_MODE
bQuantMode = MX_PERGROUP_MODE
```

### 3.3 基础块：`baseM/baseN/baseK`

MX Basic API 的主 tiling 函数是
[`GroupedQmmBasicApiTiling::DoLibApiTiling`](../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp#L67)：

```cpp
GroupedQmmTiling::CalBasicBlock();
CalL1Tiling();
```

[`CalBasicBlock`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp#L1091) 计算：

```text
A 基础块: [baseM, baseK]
B 基础块: [baseK, baseN]
C 基础块: [baseM, baseN]
```

主路径中 `baseM/baseN` 上限为 256，再根据 dtype、format 和 Cube 要求对齐。
`baseK` 先按 dtype 和 128 Byte 级别的基础块选择，MX 模式再要求对齐到 64：

```cpp
if (bQuantMode == MX_PERGROUP_MODE) {
    baseK = CeilAlign(baseK, MXFP_BASEK_FACTOR); // 64
}
```

### 3.4 L1 流水：`kAL1/kBL1/scaleKAL1`

[`CalL1Tiling`](../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp#L97)
在已知 `baseM/baseN/baseK` 后，根据 L1 容量计算 A/B 和 scale 的流水深度：

```text
CalL1Tiling
  ├─ InitCommonL1TilingFields
  ├─ CalcLeftL1Size
  └─ CalL1Depth
      ├─ CalStepKs
      └─ CalScaleFactors
```

中间参数的关系：

```text
kAL1 = stepKa * baseK
kBL1 = stepKb * baseK
```

- `baseK`：一次 L1 → L0 并执行 MMAD 的 K 基础粒度；
- `kAL1/kBL1`：A/B 一套 L1 buffer 一次覆盖的 K 元素数；
- `scaleKAL1`：一次搬入 L1 的 MX scale 所覆盖的 K 范围。

`dbL0C` 由两个 FP32 C tile 是否能同时容纳在 L0C 中决定：

```cpp
dbL0c = baseM * baseN * sizeof(float) * 2 <= l0cSize ? 2 : 1;
```

对应实现在
[`InitCommonL1TilingFields`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp#L1141)。

### 3.5 Host 与 kernel 的交界

MX Basic API 最终写入的核心字段：

```text
GMMQuantParams:
    groupNum, groupType, groupListType, aQuantMode, bQuantMode, hasBias ...

QuantBasicApiMMTiling:
    m, n, k
    baseM, baseN, baseK
    kAL1, kBL1
    scaleKAL1, scaleKBL1
    dbL0C
```

结构定义在
[`grouped_matmul_tiling_data_apt.h`](../op_kernel/arch35/grouped_matmul_tiling_data_apt.h)。
[`PostTiling`](../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp#L92)
将整个结构拷贝到 `context` 的 raw tiling buffer，kernel 入口再从 `GM_ADDR tiling`
解出它。

---

## 4. Kernel 入口：从 tiling key 到 MX 模板

[`grouped_matmul_apt.cpp`](../op_kernel/grouped_matmul_apt.cpp#L222) 中的
`grouped_matmul` 是 A5 kernel 入口。对双 FP8 输入，
[`grouped_matmul_utils.h`](../op_kernel/grouped_matmul_utils.h#L35) 定义：

```text
V310_GMM_QUANT
V310_GMM_QUANT_MX
```

`V310` 是代码中的历史命名，这个分支实际用于当前 A5/arch35 实现。

ND、A/B 非转置时命中
[`grouped_matmul_apt.cpp`](../op_kernel/grouped_matmul_apt.cpp#L234) 的分支：

```cpp
GMM_QUANT_MX_BASIC_API_IMPL_CLASS(
    Cgmct::Gemm::layout::RowMajor,
    Cgmct::Gemm::layout::RowMajor,
    Cgmct::Gemm::layout::RowMajor);
```

该宏解出 tiling data，调用
[`GmmCgmctMxKernel`](../op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cgmct_mx_kernel.h#L31)。
后者将三个核心部件组装起来：

```cpp
using BlockScheduler = GroupedMatmulAswtWithTailSplitScheduler;
using BlockMmad = Block::BlockMmadMx<...>;
using GmmKernel = Kernel::KernelQGmmMx<..., BlockMmad, ..., BlockScheduler>;
```

职责边界：

- `KernelQGmmMx`：当前是哪个 group，这个 Core 负责哪些 M/N tile；
- `BlockMmadMx`：单个 M/N tile 如何遍历 K，如何搬运并调用 MMAD。

---

## 5. group 调度：`groupNum_` 与 `groupList`

[`KernelQGmmMx::Run`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L180) 遍历所有 group：

```cpp
for (uint32_t loopIdx = 0; loopIdx < groupNum_ - 1; ++loopIdx) {
    SetMNK(loopIdx);
    ...
    bs.UpdateNextProblem(problemShape_);
    ProcessSingleGroup<false>(...);
}

// 最后一组可额外拆分 tail tile
SetMNK(groupNum_ - 1);
...
ProcessSingleGroup<...>(...);
```

[`GetSplitValueFromGroupList`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L434)
将 cumsum 或 count 形式的 groupList 转成当前组的长度。在 split-M 中，
[`SetMNK`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L408) 只更新 M：

```cpp
problemShape_.M = splitValue;
```

K 和 N 对所有 group 保持不变。空 group 会被跳过。

每个 AI Core 都执行同样的 group 循环，但 scheduler 根据本核的
`blockIdx` 判断这个 group 中哪些 tile 归它。这不是把 group 直接静态分给某一个核。

---

## 6. M/N 分核：Scheduler 到底在拆什么

### 6.1 输出平面的 tile 数

[`BlockSchedulerGmmAswtWithTailSplit::UpdateNextProblem`](../../common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h#L76)
为当前 group 计算：

```cpp
mCnt_ = CeilDiv(M, baseM);
nCnt_ = CeilDiv(N, baseN);
totalCnt_ = mCnt_ * nCnt_;
```

这是判断“当前路径没有跨 Core split-K”的最直接证据：调度任务数只有
`mCnt*nCnt`，没有再乘 `kCnt`，调度坐标中也没有 `kTileIdx`。

### 6.2 `blockIdx + roundIdx * blockNum`

Scheduler 中：

```cpp
blockNum_ = AscendC::GetBlockNum();
blockIdx_ = AscendC::GetBlockIdx() / AscendC::GetTaskRation();
```

[`GetTileIdx`](../../common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h#L179)
将本核第 `roundIdx_` 轮的逻辑任务号计算为：

```cpp
index = blockIdx_ + roundIdx_ * blockNum_;
```

再将 `index` 映射为 `(mTileIdx,nTileIdx)`。实际实现还有 window 和 N 方向蛇形遍历，
用于减少同地址冲突并改善 cache 局部性，但任务唯一性来自上面的确定性映射。

`roundIdx_` 是 scheduler 对象的普通成员，而 scheduler 对象在每个 AI Core 上局部创建。
所以每个 Core 有自己的 `roundIdx_`，不共享、不需要 atomic 或跨核同步。

例如 4 个 Core、12 个逻辑 tile，忽略蛇形映射后可简化为：

```text
Core 0: index 0, 4, 8
Core 1: index 1, 5, 9
Core 2: index 2, 6, 10
Core 3: index 3, 7, 11
```

### 6.3 `ProcessSingleGroup` 的 `do-while`

[`ProcessSingleGroup`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L363) 的结构：

```cpp
if (!bs.GetTileIdx(tileIdx)) {
    return;                         // 当前 Core 在本 group 无任务
}

UpdateOffset(groupIdx);             // group 基地址
UpdateMMGlobalAddr();

do {
    BlockShape singleShape = bs.GetBlockShape(tileIdx);
    blockOffset_ = coord.GetQuantOffset(...); // tile 内偏移
    Iterate(singleShape.M, singleShape.N);    // 算一个 MN tile 的完整 K
} while (bs.GetTileIdx(tileIdx));             // 本 Core 的下一个 tile
```

因此这个 `do-while` 既不是 group 循环，也不是 K 循环，而是：

> 当前 AI Core 在当前 group 中被分配的所有 M/N 输出 tile 的循环。

`GetBlockShape` 会返回尾块的实际 M/N 大小。`GetQuantOffset` 结合 tile index
和 tail 子块偏移，得到 A、B、Y、scale 和 bias 的地址偏移。

### 6.4 最后一组的 tail split

如果最后一组 tile 数较少，大量 Core 空闲，
[`UpdateTailTile`](../../common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h#L148)
可将尾部 M/N tile 进一步分成子块。拆分维度仍然是 M/N，不是 K。

---

## 7. 一个 Core 内：为什么整个 K 都在同一个 Cube 上

[`KernelQGmmMx::Iterate`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L397)
将 scheduler 分配的 M/N 大小和完整 K 一起传入 `BlockMmadMx`：

```cpp
BlockShape blockShape{
    singleCoreM,
    singleCoreN,
    problemShape_.K
};
mmadOp_(..., blockShape);
```

所以一次 `mmadOp_` 逻辑上负责：

```text
A[singleCoreM, K] @ B[K, singleCoreN]
```

K 不跨 Core 分，但会在这个 Core 内部分为三个层次：

```text
完整 K
  ↓
kAL1 / kBL1       GM → L1 的覆盖范围
  ↓
baseK             L1 → L0A/L0B，一次 MMAD 的 K 粒度
```

### 7.1 L1 循环

`BlockMmadMx` 根据 `kAL1 >= kBL1` 选择 A 或 B 作为外层复用对象。
例如 [`IterAL1BL1`](../../common/cgmct/block/block_mmad_mx.h#L712)：

```cpp
for (kOuter = 0; kOuter < K; kOuter += kAL1) {
    CopyAInL1(...);
    for (kInner = kOuter;
         kInner < min(kOuter + kAL1, K);
         kInner += kBL1) {
        CopyBInL1(...);
        Iterate(...);               // 继续按 baseK 拆
    }
}
```

反向复用的实现是
[`IterBL1AL1`](../../common/cgmct/block/block_mmad_mx.h#L671)。

### 7.2 L0/MMAD 循环

[`BlockMmadMx::Iterate`](../../common/cgmct/block/block_mmad_mx.h#L639)
将当前 L1 K 范围继续按 `baseK` 拆分：

```cpp
for (kL0Offset = kL1Offset;
     kL0Offset < min(kL1Offset + minKL1, K);
     kL0Offset += baseK) {
    CopyInL0A(...);
    CopyInL0B(...);
    Mmad(..., l0cOffset, ...);
}
```

每次计算：

```text
L0A[baseM, baseK] @ L0B[baseK, baseN]
    ↓
累加到同一个 L0C[baseM, baseN]
```

`cmatrixInitVal` 和 `unitFlag` 控制完整 K 的累加语义：

```cpp
mmadParams.unitFlag =
    kL0Offset + baseK >= K ? FINAL_ACCUMULATION
                           : NON_FINAL_ACCUMULATION;

mmadParams.cmatrixInitVal =
    kL0Offset == 0 && !isBias;
```

- 第一个 K 块初始化 L0C，或在有 bias 时从 bias 初始化；
- 中间 K 块继续累加；
- 最后一个 K 块标记完整 C tile 完成。

代码中 `mmadParams.k = Align64(curKL0)` 的
[`Align64`](../../common/cgmct/utils/common_utils.h#L116)
定义在 CGMCT 公共工具头文件中，作用是将 K 向上对齐到 64；
它是 C++ inline 辅助函数，不是额外的调度层次。

当前 MX kernel 中没有 `kTileIdx`、partial-sum workspace、atomic add 或跨核
reduction，因此不是并行 split-K。

### 7.3 `groupType=2/SPLIT_K` 不等于并行 split-K

`groupType=2` 时，`groupList` 描述分组轴上的 K 长度，
[`SetMNK`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L408) 更新当前 group 的 K。
但每个 group 的结果写入独立的 `groupIdx*M*N` 输出区域，并没有将多个
Core 对同一 C tile 的 K partial sum 合并起来。

---

## 8. 片上存储和 MX scale 数据流

一个 M/N tile 的主要数据路径：

```text
X/W/scale in GM
  │
  ├─ MTE2: X/W       GM -> L1
  └─ MTE2: scale     GM -> L1
          │
          └─ MTE1 MX LoadData
                  data + scale -> L0A/L0B
                          │
                          └─ Cube MMAD
                                  │
                                  └─ L0C FP32
                                          │
                                          └─ FixPipe -> GM
```

### 8.1 `MatmulWithScale` 只是模板策略

[`MatmulWithScale`](../../common/cgmct/policy/dispatch_policy.h#L99) 是本仓库定义的
compile-time policy：

```cpp
template <class SingleCoreShape, uint64_t FULL_LOAD_MODE_ = 0>
struct MatmulWithScale {
    using ScheduleType = KernelMmadWithScale;
    using SingleShape = SingleCoreShape;
    static constexpr uint64_t fullLoadMode = FULL_LOAD_MODE_;
};
```

它用于选中 `BlockMmadMx` 的模板特化，自身不执行计算，也不是硬件指令。

### 8.2 scale 在哪里进入计算

scale 先通过
[`CopyInScaleA`](../../common/cgmct/block/block_mmad_mx.h#L341) 和
[`CopyInScaleB`](../../common/cgmct/block/block_mmad_mx.h#L379) 从 GM 搬到 L1。

真正将 scale 与数据一起送入 MX 计算流水的是：

```cpp
AscendC::LoadData(
    l0aLocal,
    al1Local,
    scaleAl1Local,
    loadDataParams,
    loadData2DMxParams);
```

以及 B 侧对应调用，位置分别是：

- [`CopyInL0A`](../../common/cgmct/block/block_mmad_mx.h#L431)；
- [`CopyInL0B`](../../common/cgmct/block/block_mmad_mx.h#L485)。

`AscendC::LoadData` 是 AscendC 硬件抽象/intrinsic 接口。在这个 overload 中，
L1 中的 FP8 数据、E8M0 scale 和 `LoadData2DMxParams` 同时传入。后续
`Mmad` 不再显式接收 scale。

这条 MXFP8 路径不会先在 GM 生成一份完整 FP16/FP32 反量化 tensor；
scale/dequant 语义融合在 MX LoadData/MMAD 流水中。最后的 `CopyOut` 主要负责
L0C FP32 到 FP16/BF16/FP32 输出的 FixPipe 转换。

---

## 9. Ping-pong 分别发生在哪一层

### 9.1 L0A/L0B：相邻 `baseK` 块之间

在核内 K 循环中：

```cpp
l0Offset = HALF_L0_SIZE * (l0PingPong_ & 1);
CopyInL0A(l0aLocal_[l0Offset], ...);
CopyInL0B(l0bLocal_[l0Offset], ...);
Mmad(...);
l0PingPong_++;
```

所以相邻 `baseK` 块交替使用 L0A/L0B Ping 和 Pong，用来重叠下一个 K 块的
MTE1 LoadData 与当前 K 块的 Cube MMAD。

### 9.2 L0C：相邻 M/N 输出 tile 之间

[`RunMmadLoop`](../../common/cgmct/block/block_mmad_mx.h#L826) 在完整 K 循环开始前选择：

```cpp
l0cOffset = (l0cPingPong_ & 1) * HALF_L0C_SIZE;
```

这个 `l0cOffset` 在当前 M/N tile 的整个 K 累加过程中不变。完成后：

```cpp
CopyOut(cGlobal, c1Local_[l0cOffset], ...);
UpdateL0cPingPong();
```

下一个 M/N tile 才切换 L0C 的另一半，从而可以重叠：

```text
前一个 tile: FixPipe 正在从 L0C Ping 写回
下一个 tile: Cube 正在向 L0C Pong 计算
```

如果某个 Core 在当前 kernel 中只分到一个 M/N tile，它只会使用 L0C Ping，
Pong 没有机会被使用，这与 tiling 允许 `dbL0C=2` 并不矛盾。

---

## 10. 用一个完整例子串起来

将第 2 节的数据简化到 Group 1：

```text
M = 20
N = 512
K = 4096

baseM = 128
baseN = 256
baseK = 128

假设 kAL1 = 512
假设 kBL1 = 256
```

### 10.1 M/N 分核

```text
mCnt = ceil(20 / 128)  = 1
nCnt = ceil(512 / 256) = 2
totalCnt = 2
```

该 group 只有两个输出 tile：

```text
Tile 0: Y[0:20,   0:256]
Tile 1: Y[0:20, 256:512]
```

若两个 Core 参与：

```text
Core A: X[20,4096] @ W[4096,256] -> Tile 0
Core B: X[20,4096] @ W[4096,256] -> Tile 1
```

每个 Core 负责各自 N 区间的完整 K。

### 10.2 一个 Core 内的 K 层次

```text
完整 K = 4096

A L1 一次覆盖 512 K
B L1 一次覆盖 256 K
一次 MMAD 覆盖 128 K
```

从 MMAD 粒度看，当前 Core 需要：

```text
4096 / 128 = 32 次 baseK MMAD
```

简化时间线：

```text
K[0:128]
  LoadData -> L0A/B Ping
  MMAD -> L0C Ping（初始化）

K[128:256]
  LoadData -> L0A/B Pong
  MMAD -> L0C Ping（累加）

K[256:384]
  LoadData -> L0A/B Ping
  MMAD -> L0C Ping（累加）

...

K[3968:4096]
  LoadData -> L0A/B Pong
  MMAD -> L0C Ping（FINAL_ACCUMULATION）

FixPipe: L0C Ping -> GM
```

如果同一 Core 还有下一个 M/N tile，下一个 tile 的 32 次 K MMAD 会全部累加到
L0C Pong，然后再切回 Ping。

---

## 11. 源码阅读地图

建议按下列顺序阅读：

1. [`TilingGMM`](../op_host/op_tiling/grouped_matmul_tiling.cpp#L2240)：先看 A5 量化分流；
2. [`GroupedQmmBasicApiTiling::DoLibApiTiling`](../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp#L67)：看主 tiling 调度；
3. [`CalBasicBlock`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp#L1091)：看 `baseM/N/K`；
4. [`GroupedQmmBasicApiTiling::CalL1Tiling`](../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp#L97)：看 L1 深度和 scale 覆盖；
5. [`grouped_matmul`](../op_kernel/grouped_matmul_apt.cpp#L222)：看 kernel 编译分支；
6. [`GmmCgmctMxKernel`](../op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cgmct_mx_kernel.h#L31)：看组件组装；
7. [`KernelQGmmMx::Run`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L180)：看 group 循环；
8. [`ProcessSingleGroup`](../../common/cgmct/kernel/kernel_qgmm_mx.h#L363)：看每核 MN tile 循环；
9. [`BlockSchedulerGmmAswtWithTailSplit`](../../common/cgmct/block/block_scheduler_gmm_aswt_with_tail_split.h#L71)：看 tile 与 Core 映射；
10. [`BlockMmadMx::RunMmadLoop`](../../common/cgmct/block/block_mmad_mx.h#L826)：看一个 MN tile 的完整 K；
11. [`IterAL1BL1`](../../common/cgmct/block/block_mmad_mx.h#L712) 和 [`Iterate`](../../common/cgmct/block/block_mmad_mx.h#L639)：看 L1/baseK 循环；
12. [`CopyInL0A`](../../common/cgmct/block/block_mmad_mx.h#L431)、[`CopyInL0B`](../../common/cgmct/block/block_mmad_mx.h#L485) 和 [`Mmad`](../../common/cgmct/block/block_mmad_mx.h#L856)：看 MX scale 和真正计算。

---

## 12. 容易混淆的概念

| 容易误解的说法 | 更准确的理解 |
| --- | --- |
| `groupNum` 就是 batch size | 它是 grouped GEMM 数量，可类比 batch，但每组分组维度可不同 |
| 一个 group 只在一个 Cube 上算 | 一个 group 的 M/N 输出平面会分成多个 tile，分给多个 Core/Cube |
| `baseK` 代表跨 Core 拆 K | `baseK` 是同一 Core 内 L1 → L0/MMAD 的 K 粒度 |
| `groupType=SPLIT_K` 就是并行 split-K | 它是 groupList 沿 K 的功能分组；当前 MX 路径没有跨 Core partial-sum reduction |
| `roundIdx_` 是所有 Core 共享的计数器 | 它是每个 Core 的 scheduler 局部状态，无需跨核同步 |
| L0C ping-pong 用于同一 tile 的 K 分块 | 同一 tile 的所有 K 分块必须累加到同一 L0C；L0C 在相邻 MN tile 间切换 |
| `MatmulWithScale` 是一条指令 | 它是仓库定义的模板策略；硬件抽象接口是带 MX 参数的 `AscendC::LoadData` |
| FP8 输入会先整体转成 FP16 | 当前路径不生成完整高精度中间 tensor，scale/dequant 融合在 MX 计算流水中 |

最后可以用一句话概括这条调度路径：

> Host 选定 M/N/K 基础块和片上流水；Kernel 按 group 遍历问题，按 M/N tile 将输出平面分给多个 Cube；每个 Cube 在核内遍历自己 tile 的完整 K，并通过 L1/L0 分块、MX scale LoadData、MMAD 和 L0C 累加完成计算。
