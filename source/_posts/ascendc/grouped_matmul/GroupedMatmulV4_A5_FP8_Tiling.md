---
title: "GroupedMatmulV4 A5 FP8 Tiling 代码解析"
date: 2026-07-20 17:50:34
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

## 1. 文档范围

本文面向以下场景：

- 算子：`GroupedMatmul`；
- 对外接口：`aclnnGroupedMatmulV4`；
- 芯片架构：A5，在代码中对应 `NpuArch::DAV_3510` 和 `arch35`；
- 输入类型：`x` 和 `weight` 均为 `FLOAT8_E4M3FN`；
- 重点量化模式：MXFP8，即 `scale`/`perTokenScale` 使用 `FLOAT8_E8M0`。

<!--more-->

V4 只影响 ACLNN 参数校验和执行器构造。进入 L0 算子后，V4/V5 共用 `GroupedMatmul` 的 infer shape、tiling 和 kernel 实现。因此本文从通用的 `TilingGMM()` 开始分析。

主要代码文件：

| 文件 | 作用 |
| --- | --- |
| [`grouped_matmul_tiling.cpp`](../op_host/op_tiling/grouped_matmul_tiling.cpp) | GroupedMatmul tiling 总入口和架构/数据类型分流 |
| [`grouped_quant_matmul_tiling.cpp`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp) | A5 量化场景公共解析、通用 tiling 和公共辅助函数 |
| [`grouped_quant_basic_api_matmul_tiling.cpp`](../op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp) | A5 MXFP8/MXFP4 Basic API tiling |
| [`grouped_quant_matmul_tiling.h`](../op_host/op_tiling/arch35/grouped_quant_matmul_tiling.h) | 量化模式、输入信息、基础 tiling 中间结构和常量 |
| [`grouped_matmul_tiling_data_apt.h`](../op_kernel/arch35/grouped_matmul_tiling_data_apt.h) | Host 和 kernel 共享的最终 tiling data 定义 |
| [`grouped_matmul_apt.cpp`](../op_kernel/grouped_matmul_apt.cpp) | A5 kernel 总入口和 tiling key 分支 |
| [`gqmm_cgmct_mx_kernel.h`](../op_kernel/arch35/quant_adaptive_sliding_window_templates/gqmm_cgmct_mx_kernel.h) | MXFP8 最终使用的 kernel 包装和 CGMCT 类型组装 |

## 2. 先明确矩阵和单位

GroupedMatmul 的单组计算可以写成：

```text
A[M, K] × B[K, N] = C[M, N]
```

代码中的关键维度和 tiling 字段：

| 字段 | 含义 | 单位 |
| --- | --- | --- |
| `mSize/nSize/kSize` | 整个问题的 M/N/K | 元素个数 |
| `baseM/baseN/baseK` | 一次基础 Cube/MMAD tile 的 M/N/K | 元素个数 |
| `singleCoreM/N/K` | 单核逻辑处理范围 | 元素个数 |
| `depthA1/depthB1` | L1 中 A/B tile 的流水深度 | tile 份数，包含双缓冲语义 |
| `stepKa/stepKb` | 一次 L1 装载覆盖的 `baseK` 数量 | `baseK` 的倍数 |
| `kAL1/kBL1` | 一次 A/B L1 装载覆盖的 K 元素数 | 元素个数 |
| `scaleKAL1/scaleKBL1` | MX scale 一次装载覆盖的 K | 元素个数 |
| `L1_ALIGN_SIZE` | L1 连续搬运对齐粒度 | 字节 |
| `CUBE_BLOCK` | M/N 方向的 Cube 对齐粒度 | 元素个数 |

特别注意：`baseM/baseN/baseK` 始终是元素数量。代码中有些常量以字节为单位，必须先通过 `GetShapeWithDataType()` 换算成元素数量，才能用于维度对齐。

### 2.1 两个字节/元素转换函数

```cpp
uint64_t GetSizeWithDataType(uint64_t shapeSize, ge::DataType dtype);
uint64_t GetShapeWithDataType(uint64_t size, ge::DataType dtype);
```

- `GetSizeWithDataType()`：元素数量转换成字节数；
- `GetShapeWithDataType()`：字节数转换成该 dtype 下的元素数量。

常见换算关系：

| dtype | 单元素大小 | 32 字节对应元素数 | 128 字节对应元素数 |
| --- | ---: | ---: | ---: |
| FP32 | 4 B | 8 | 32 |
| FP16/BF16 | 2 B | 16 | 64 |
| FP8/INT8 | 1 B | 32 | 128 |
| FP4/INT4 | 0.5 B | 64 | 256 |

“连续方向按 32 字节对齐”的准确含义是：

```text
对齐后的元素数 × 单元素字节数，是 32 字节的整数倍。
```

例如 50 个 FP8 元素占 50 字节，需要补到 64 个元素，即 64 字节；50 个 FP16 元素占 100 字节，需要补到 64 个元素，即 128 字节。

### 2.2 先理解这些参数所在的硬件层次

可以把一次矩阵乘的数据流简化为：

```text
GM（容量大，保存完整输入）
  ↓ 分块搬运
L1（片上缓存，保存接下来若干个 K 块）
  ↓ 继续送入矩阵计算单元
L0A/L0B + Cube/MMAD（计算一个基础 M×N×K 块）
  ↓
L0C/输出
```

- `mSize/nSize/kSize` 描述 GM 中需要解决的完整数学问题；
- `baseM/baseN/baseK` 描述底层一次基础计算块；
- `singleCoreM/N/K` 描述通用 tiling 中分给一个核的一份有效任务范围；
- `depthA1/depthB1`、`stepKa/stepKb`、`kAL1/kBL1` 描述 L1 一次准备多少个 K 块；
- `scaleKAL1/scaleKBL1` 描述与这些数据块配套的 MX scale 一次准备多长的 K；
- `L1_ALIGN_SIZE`、`CUBE_BLOCK` 是搬运和计算硬件要求的对齐粒度。

还要注意两套数据结构的区别：

1. `GQmmBasicTiling` 是 host tiling 内部的中间结果，包含 `singleCore*`、`depth*`、`stepK*` 等字段；
2. MX Basic API 最终传给 kernel 的 `QuantBasicApiMMTiling` 不包含 `singleCore*` 和 `depth*`，host 会把有用结果转换成 `base*`、`kAL1/kBL1`、`scaleKAL1/scaleKBL1`。

因此下面对 `singleCore*` 和 `depth*` 的解释既适用于通用 `TCubeTiling`，也解释了 MX Basic API 在 host 内部如何得到最终字段，但不能认为 MX kernel 会原样读取这些中间字段。

### 2.3 `mSize/nSize/kSize`：完整数学问题的大小

单组矩阵乘为：

```text
A[M,K] × B[K,N] = C[M,N]
```

所以：

- `mSize`：A 和 C 的行数；
- `nSize`：B 和 C 的列数；
- `kSize`：A 的列数和 B 的行数，也就是乘加归约轴长度。

它们都是元素个数，不是字节数。例如：

```text
x      shape = [600,4096]
weight shape = [4096,700]
```

在不转置时：

```text
mSize=600, kSize=4096, nSize=700
```

这三个值描述完整问题，但通常太大，不能一次全部放进片上存储并完成计算，因此需要继续切成基础块。

### 2.4 `baseM/baseN/baseK`：一份基础计算块

它们定义：

```text
A 基础块 = [baseM,baseK]
B 基础块 = [baseK,baseN]
C 基础块 = [baseM,baseN]
```

例如：

```text
baseM=256, baseN=256, baseK=128
```

表示底层以如下块为基本单位组织计算：

```text
A[256,128] × B[128,256] → C[256,256]
```

这并不表示整个矩阵只有这么大。对于 `M=600,N=700`，M/N 平面会形成大致如下的 tile 网格：

```text
M 方向有效长度：256, 256, 88
N 方向有效长度：256, 256, 188
```

也就是一个 3×3 的基础任务网格。最后一行/列属于尾块，其有效大小小于 `baseM/baseN`。`baseM/baseN` 仍保持规则的硬件块大小，kernel 另外记录或计算尾块的有效范围。

`baseK` 与 M/N 不同：K 是归约轴。若 `K=4096,baseK=128`，完成同一个 C tile 需要沿 K 循环：

```text
4096 / 128 = 32 个 K 基础块
```

每个 K 块产生部分乘加结果，32 次结果在 C 上累加，才得到最终输出。

### 2.5 `singleCoreM/N/K`：一份单核任务的有效范围

名字可以拆成：

```text
singleCoreM = 当前分给一个核的任务在 M 方向有多少有效元素
singleCoreN = 当前分给一个核的任务在 N 方向有多少有效元素
singleCoreK = 当前任务需要归约多少个 K 元素
```

在当前量化 tiling 的公共初始化中：

```cpp
singleCoreM = min(M, baseM);
singleCoreN = min(N, baseN);
singleCoreK = K;
```

直观上，一份常规任务负责一个 M/N 基础块，但要遍历完整 K：

```text
一个任务计算 C 的 [singleCoreM,singleCoreN] 区域，
并沿 singleCoreK 完成全部乘加归约。
```

例如 `M=600,N=700,K=4096`，中间初始化可能是：

```text
singleCoreM=256
singleCoreN=256
singleCoreK=4096
```

对于右下角尾块，有效范围可能变成：

```text
singleCoreM=88
singleCoreN=188
singleCoreK=4096
```

但“singleCore”不能机械理解为“从算子开始到结束，一个物理核永远只做这一块”：

- scheduler 会把多个 group 和 M/N tile 映射到可用核；
- 一个核完成一个任务后可能继续取下一个任务；
- 尾块较大或核数有剩余时，scheduler 还可能把一个尾 tile 再拆给多个核；
- MX Basic API kernel 的 scheduler 直接使用 `M/N/K` 和 `baseM/N/K` 计算任务，不读取最终 tiling data 中不存在的 `singleCore*` 字段。

所以更准确的说法是：`singleCore*` 是通用 tiling/kernel 协议中“一次单核任务的有效 shape”，而不是整个算子的静态分核总量。

### 2.6 `depthA1/depthB1`：L1 能形成多深的 A/B 流水

`A1/B1` 中的 `1` 指 L1 层。一个最基本的 A/B K 块是：

```text
A1 单份 = [baseM,baseK]
B1 单份 = [baseK,baseN]
```

`depthA1/depthB1` 描述 host 计划让 L1 为 A/B 建立多深的 tile 流水。它们不是元素数，也不是字节数；理解为“以基础 K tile 为单位的缓存深度”更合适。

为什么需要多份：如果每次 Cube 算完一个 K 块才开始从 GM 搬下一个块，计算单元会等待搬运。流水化会让两类动作重叠：

```text
时刻 1：计算 K 块 0，同时搬运 K 块 1
时刻 2：计算 K 块 1，同时搬运 K 块 2
```

双缓冲时，depth 通常包含两套缓冲区。当前代码通过：

```cpp
stepKa = depthA1 == 1 ? 1 : depthA1 / 2;
stepKb = depthB1 == 1 ? 1 : depthB1 / 2;
```

把包含双缓冲的 depth 换算为单套缓冲实际覆盖的 K tile 数。例如：

```text
depthA1=8 → stepKa=4
depthB1=4 → stepKb=2
```

可以粗略理解为 A 有两套缓冲，每套容纳 4 个 `baseK`；B 有两套缓冲，每套容纳 2 个 `baseK`。

depth 越大不一定越好：它会占更多 L1。tiling 会在 L1 容量、GM 搬运带宽、K 大小和双缓冲之间选择。

### 2.7 `stepKa/stepKb`：单套 L1 缓冲覆盖多少个 baseK

`stepKa` 和 `stepKb` 的单位是 `baseK` 的倍数：

```text
stepKa=4：A 的单套 L1 缓冲覆盖 4 个 baseK
stepKb=2：B 的单套 L1 缓冲覆盖 2 个 baseK
```

A 和 B 可以不同，因为：

- A tile 字节数是 `baseM×baseK×sizeof(A)`；
- B tile 字节数是 `baseN×baseK×sizeof(B)`；
- M/N、dtype、转置和 scale 附加空间可能不同；
- tiling 希望在有限 L1 中平衡两边搬运。

代码还会限制：

- `stepK×baseK` 不应无意义地远超 K；
- A/B 的 stepK 尽量保持合适的倍数关系；
- 特殊量化模式可能限制最大 stepK。

### 2.8 `kAL1/kBL1`：把 stepK 换算成实际 K 元素数

MX Basic API 最终不把 `stepKa/stepKb` 直接写给 kernel，而是写：

```cpp
kAL1 = stepKa * baseK;
kBL1 = stepKb * baseK;
```

二者单位都是 K 方向的元素个数。

例如：

```text
baseK=128
stepKa=4 → kAL1=512
stepKb=2 → kBL1=256
```

含义：

- A 的一套 L1 数据覆盖 512 个 K 元素；
- B 的一套 L1 数据覆盖 256 个 K 元素。

FP8 下对应的数据字节数还要乘 M/N：

```text
A 一套数据字节数 ≈ baseM × kAL1 × 1 B
B 一套数据字节数 ≈ baseN × kBL1 × 1 B
```

因此 `kAL1=512` 不是“L1 使用 512 字节”，而只是 A 在 K 方向覆盖 512 个元素。

### 2.9 `scaleKAL1/scaleKBL1`：MX scale 覆盖的 K 范围

MXFP8 不仅搬运 A/B，还要搬运与它们配套的 scale。当前实现中：

```text
MX_GROUP_SIZE=32
```

即 K 方向每 32 个数据元素对应一个 scale group。若：

```text
scaleKAL1=1024
```

那么它覆盖 1024 个 K 元素，对应大约：

```text
1024 / 32 = 32 个 K scale group
```

实际 scale 元素总数还要乘对应的 M 或 N 维度，并考虑 scale 的具体布局和对齐。

Basic API 当前令：

```cpp
scaleKBL1 = scaleKAL1;
```

它先根据 `scaleFactorA/B`、`stepKa/stepKb` 和 `baseK` 计算一个能同时满足 A/B scale 流水的 K 覆盖范围，再限制不超过实际 K。

不要混淆：

```text
kAL1/kBL1       描述 A/B 数据本身一次覆盖多少 K
scaleKAL1/KBL1 描述配套 scale 一次覆盖多少 K
```

scale 的复用范围可能比一套 A/B 数据的范围更大，因此这些值不要求相等。

### 2.10 `L1_ALIGN_SIZE=32`：连续搬运长度的字节粒度

`L1_ALIGN_SIZE` 的单位是字节：

```cpp
L1_ALIGN_SIZE=32 B
```

当某个矩阵维度是内存连续方向时，代码希望该方向的 tile 长度占用 32 B 的整数倍。因为维度字段以元素计数，先做换算：

```text
FP32: 32 B / 4 B = 8 个元素
FP16: 32 B / 2 B = 16 个元素
FP8 : 32 B / 1 B = 32 个元素
FP4 : 32 B / 0.5 B = 64 个元素
```

例如 `N=50`、weight 为 FP8 且 N 是连续方向：

```text
50 个元素 = 50 B
向上对齐为 64 个元素 = 64 B = 2×32 B
```

这里主要是在对齐连续数据块的长度；不能简单理解为“矩阵指针地址变成 32”。

### 2.11 `CUBE_BLOCK=16`：M/N 方向的矩阵计算粒度

`CUBE_BLOCK` 的单位是元素：

```cpp
CUBE_BLOCK=16 个元素
```

它反映 Cube 矩阵计算在 M/N 方向使用的基础粒度。若某个 M/N 维度不是当前存储布局的连续内轴，代码通常按 16 个元素对齐：

```text
M=100 → align_up(100,16)=112
N=257（若不先受 256 上限限制）→ align_up(257,16)=272
```

`CUBE_BLOCK=16` 和 `L1_ALIGN_SIZE=32 B` 解决的是不同问题：

| 常量 | 关注对象 | 单位 | 目的 |
| --- | --- | --- | --- |
| `CUBE_BLOCK` | 矩阵计算的 M/N 形状 | 元素 | 满足 Cube 计算粒度 |
| `L1_ALIGN_SIZE` | 连续方向的数据长度 | 字节 | 满足数据搬运/存储粒度 |

### 2.12 把所有字段串到一个例子中

假设：

```text
M=600, N=700, K=4096
x/weight=FP8 E4M3
transA=false, transB=false
MX scale=FP8 E8M0
```

基础块可能为：

```text
baseM=256
baseN=256
baseK=128
```

含义：输出 M/N 平面按约 3×3 个 tile 处理，每个 C tile 沿 K 再累计 32 个 `baseK` 块。

若为了说明概念，假设 L1 计算得到：

```text
depthA1=8, depthB1=4
```

则：

```text
stepKa=4, stepKb=2
kAL1=4×128=512
kBL1=2×128=256
```

表示 A/B 的每套 L1 缓冲分别覆盖 512/256 个 K 元素。假设最终 `scaleKAL1=scaleKBL1=1024`，则一次 scale 装载覆盖 1024 个 K 元素，即约 32 个 MX K-scale group。

这里的 depth 和 scale 数值只是帮助理解字段关系的示例，实际值必须由芯片 L1 大小、M/N/K、bias、scale shape 和代码中的容量公式计算。

## 3. 总体调用链

```text
aclnnGroupedMatmulV4GetWorkspaceSize()
  └─ aclnnGroupedMatmulGetWorkspaceSizeCommon(..., GMMApiVersion::V4, ...)
      └─ l0op::GroupedMatmul()
          ├─ INFER_SHAPE(GroupedMatmul)
          └─ ADD_TO_LAUNCHER_LIST_AICORE(GroupedMatmul)
              └─ TilingGMM()
                  └─ DAV_3510 + 双低比特输入
                      └─ TilingRegistry::DoTilingImpl(context, {0, 1})
                          ├─ priority 0: GroupedQmmBasicApiTiling
                          └─ priority 1: GroupedQmmTiling
```

模板注册位于 `grouped_quant_matmul_tiling.cpp`：

```cpp
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmul, GroupedQmmBasicApiTiling, 0);
REGISTER_OPS_TILING_TEMPLATE(GroupedMatmul, GroupedQmmTiling, 1);
```

注册表按 `{0, 1}` 依次调用模板的 `DoTiling()`：

1. 模板 0 能处理则直接成功返回；
2. 模板 0 的 `IsCapable()` 返回 `false` 时，`DoTiling()` 返回 `GRAPH_PARAM_INVALID`；
3. 注册表继续尝试模板 1。

## 4. A5 量化场景如何被选中

总入口 `TilingGMM()` 首先读取第 0 个动态输入的 descriptor：

```cpp
auto xDesc = context->GetDynamicInputDesc(X_INDEX, 0);
auto w0Desc = context->GetDynamicInputDesc(WEIGHT_INDEX, 0);
ge::DataType xDType = xDesc->GetDataType();
ge::DataType weightDtype = w0Desc->GetDataType();
```

这里的 descriptor 是 tensor 元数据，不是 device 上的实际数据。它包含 dtype、format 等信息。shape 则通过 `GetDynamicInputShape()` 获取。

在 `DAV_3510` 上，双 FP8 都是一字节数据，因此命中量化条件：

```cpp
bool isQuant = ... ||
    (ge::GetSizeByDataType(xDType) == 1 &&
     ge::GetSizeByDataType(weightDtype) == 1);
```

随后进入量化模板注册表。

## 5. TilingBaseClass 的执行阶段

两个量化模板都继承 `TilingBaseClass`。其 `DoTiling()` 固定按以下顺序执行：

```text
GetShapeAttrsInfo()   解析输入、输出、属性
GetPlatformInfo()     获取 AIC/AIV 数量以及 UB/L1/L0 大小
IsCapable()           当前模板是否支持此场景
DoOpTiling()          填充算子级/分组级 tiling 参数
DoLibApiTiling()      计算 base block、L1 流水等核心参数
GetWorkspaceSize()    设置 workspace
PostTiling()          将 tiling data 写入 raw buffer
GetTilingKey()        生成 kernel 模板选择 key
```

需要区分：

- `TilingGMM()` 是顶层分流入口；
- `DoTiling()` 是 tiling 生命周期编排；
- `DoLibApiTiling()` 才是基础切块和 L1 参数计算的主要逻辑。

## 6. 输入解析阶段

### 6.1 `GetShapeAttrsInfo()`

`GroupedQmmBasicApiTiling` 复用父类的解析逻辑：

```cpp
inputParams_.Reset();
return GroupedQmmTiling::GetShapeAttrsInfo();
```

父类依次调用：

```cpp
AnalyzeDtype();
AnalyzeAttrs();
AnalyzeInputs();
```

### 6.2 `AnalyzeDtype()`

主要读取：

- `x/weight/y` dtype；
- `weight` format；
- bias 是否存在以及 bias dtype；
- `scale` dtype；
- `perTokenScale` dtype。

双 FP8 MX 场景的典型结果：

```text
aDtype             = DT_FLOAT8_E4M3FN
bDtype             = DT_FLOAT8_E4M3FN
scaleDtype         = DT_FLOAT8_E8M0
perTokenScaleDtype = DT_FLOAT8_E8M0
```

### 6.3 `AnalyzeAttrs()`

主要读取：

- `splitItem`；
- `groupType`；
- `groupListType`；
- `actType`；
- `transA/transposeX`；
- `transB/transposeWeight`。

### 6.4 `AnalyzeInputs()`

这是输入解析的主函数，依次完成：

1. 获取 `x` 和 `weight` 的 origin/storage shape；
2. `SetGroupNum()`：从 `groupList` shape 解析 group 数；
3. `SetMKN()`：从最后两个维度解析 M/K/N；
4. 获取 `scale` 和 `perTokenScale` shape；
5. `SetQuantMode()`：识别 per-tensor/per-token/per-channel/MX/per-block；
6. `CheckQuantParams()`：检查 scale shape；
7. 检查 NZ、激活函数、FP4 等约束；
8. `SetKernelType()`：决定 fixpipe、vector dequant 或 per-block kernel；
9. `CheckCoreNum()`：检查 AIC/AIV 比例。

### 6.5 `SetMKN()` 如何解析最后两个维度

常量定义：

```cpp
LAST_FIRST_DIM_INDEX = 1;  // 倒数第 1 维
LAST_SECOND_DIM_INDEX = 2; // 倒数第 2 维
```

`GetDimNum()` 返回维度数量，`GetDim()` 使用从 0 开始的下标，因此对于 rank 为 `r` 的 shape：

```text
最后一维   = GetDim(r - 1)
倒数第二维 = GetDim(r - 2)
```

解析关系：

| 参数 | `trans=false` 的存储 shape | `trans=true` 的存储 shape |
| --- | --- | --- |
| x | `[..., M, K]` | `[..., K, M]` |
| weight | `[..., K, N]` | `[..., N, K]` |

因此：

```cpp
M = transA ? x[-1] : x[-2];
K = transA ? x[-2] : x[-1];
N = transB ? weight[-2] : weight[-1];
```

### 6.6 MX 模式识别

`IsMicroScaling()` 的判断非常直接：

```cpp
return inputParams_.scaleDtype == ge::DT_FLOAT8_E8M0;
```

命中后 `SetQuantMode()` 设置：

```cpp
aQuantMode = QuantMode::MX_PERGROUP_MODE;
bQuantMode = QuantMode::MX_PERGROUP_MODE;
```

模板 0 的 `IsCapable()` 要求：

1. `IsMicroScaling()` 为 true；
2. x 是 FP8 或 FP4 类型。

双 `FLOAT8_E4M3FN` + `FLOAT8_E8M0` scale 因而进入 `GroupedQmmBasicApiTiling`。

## 7. `DoOpTiling()`：填充算子级参数

MX Basic API 的 `DoOpTiling()` 主要把已经解析的值写入 `GMMQuantParams`：

```text
groupNum
activeType
aQuantMode / bQuantMode
singleX / singleW / singleY
groupType / groupListType
hasBias
```

这里基本不做 `baseM/baseN/baseK` 计算。核心切块在下一阶段。

## 8. `DoLibApiTiling()`：核心 tiling 逻辑

MX Basic API 的主函数：

```cpp
ge::graphStatus GroupedQmmBasicApiTiling::DoLibApiTiling()
{
    GroupedQmmTiling::CalBasicBlock();
    CalL1Tiling();
    // 将中间结果写入最终 mmTilingData
}
```

它分为两大部分：

1. `CalBasicBlock()`：决定 `baseM/baseN/baseK`；
2. `CalL1Tiling()`：根据 L1 容量决定 depth、stepK 和 MX scale 覆盖范围。

## 9. `CalBasicBlock()` 详细解析

### 9.1 `baseM/baseN/baseK` 的含义

基础 tile：

```text
A tile: [baseM, baseK]
B tile: [baseK, baseN]
C tile: [baseM, baseN]
```

这些字段单位都是元素个数。tile 超出原始 M/N/K 的部分由 kernel 的 tail/padding 逻辑处理。

### 9.2 `baseM`

先限制最大值为 256：

```cpp
baseM = min(M, 256);
```

再根据 `transA` 对齐：

```cpp
baseM = !transA
    ? CeilAlign(baseM, CUBE_BLOCK)
    : CeilAlign(baseM, GetShapeWithDataType(L1_ALIGN_SIZE, aDtype));
```

- `transA=false`：x 存储为 `[M,K]`，M 是外轴，按 Cube 的 16 元素粒度对齐；
- `transA=true`：x 存储为 `[K,M]`，M 是连续内轴，要求其连续字节长度按 32 B 对齐。

对于 FP8：

```text
transA=false: baseM = align_up(min(M,256), 16)
transA=true : baseM = align_up(min(M,256), 32)
```

### 9.3 特殊 GB 分支

如果：

```text
aQuantMode = PERGROUP_MODE
bQuantMode = PERBLOCK_MODE
```

则固定选择：

```cpp
baseN = (N <= 128 || baseM > 128) ? 128 : 256;
baseK = 128;
return;
```

该分支直接返回，不执行普通/MX 逻辑。MXFP8 不属于这个分支。

### 9.4 `baseN`

普通/MX 分支先限制：

```cpp
baseN = min(N, 256);
```

再根据 `transB` 对齐：

```cpp
baseN = transB
    ? CeilAlign(baseN, CUBE_BLOCK)
    : CeilAlign(baseN, GetShapeWithDataType(L1_ALIGN_SIZE, bDtype));
```

- `transB=false`：weight 存储为 `[K,N]`，N 是连续内轴，按 32 B 对齐；
- `transB=true`：weight 存储为 `[N,K]`，N 是外轴，按 16 个元素的 Cube 粒度对齐。

对于 FP8：

```text
transB=false: baseN = align_up(min(N,256), 32)
transB=true : baseN = align_up(min(N,256), 16)
```

### 9.5 `baseK`

普通逻辑：

```cpp
baseK = CeilAlign(
    min(GetShapeWithDataType(128, aDtype), K),
    GetShapeWithDataType(32, aDtype));
```

含义是：

1. 基础 K 块最多承载约 128 B 的 A 数据；
2. 不超过逻辑 K；
3. 连续 K 方向按 32 B 粒度向上对齐。

当 K 足够大时：

| dtype | 普通 `baseK` |
| --- | ---: |
| FP32 | 32 元素 |
| FP16/BF16 | 64 元素 |
| FP8/INT8 | 128 元素 |
| FP4/INT4 | 256 元素 |

MX 模式还有额外约束：

```cpp
baseK = CeilAlign(baseK, 64);
```

因此 FP8 MX 可以简化为：

```text
baseK = align_up(min(K, 128), 64)
```

如果 K 足够大，最终 `baseK=128` 个 FP8 元素，也就是 128 B。

FP4 MX 且 `transB=false` 时，代码还要求 `baseN` 按 64 元素对齐；FP8 不进入该分支。

### 9.6 一个完整的基础块算例

输入：

```text
M=100, N=700, K=4096
x/weight=FLOAT8_E4M3FN
scale/perTokenScale=FLOAT8_E8M0
transA=false, transB=false
```

计算：

```text
baseM = align_up(min(100,256), 16) = 112
baseN = align_up(min(700,256), 32) = 256
baseK = align_up(min(4096,128), 32) = 128
MX 再按 64 对齐，仍为 128
```

最终基础 tile：

```text
A: [112,128]
B: [128,256]
C: [112,256]
```

FP8 下 A/B 单份 tile 大小：

```text
A tile = 112 × 128 × 1 B = 14336 B
B tile = 128 × 256 × 1 B = 32768 B
```

## 10. `CalL1Tiling()` 详细解析

`CalBasicBlock()` 只确定一份基础 tile。`CalL1Tiling()` 继续回答：L1 能同时缓存多少份 A/B tile，以及一次缓存覆盖多长的 K。

MX Basic API 路径：

```text
CalL1Tiling()
  ├─ InitCommonL1TilingFields()
  ├─ CalcLeftL1Size()
  └─ CalL1Depth(leftL1Size)
      ├─ 计算 A/B/scale 单份 tile 字节数
      ├─ 估算 depthA1/depthB1
      ├─ ModifyDepthForUnalign()
      ├─ CalStepKs()
      └─ CalScaleFactors()
```

### 10.1 `InitCommonL1TilingFields()`

初始化：

```cpp
stepM = 1;
stepN = 1;
singleCoreM = min(M, baseM);
singleCoreN = min(N, baseN);
singleCoreK = K;
iterateOrder = 0;
```

并检查 L0C 是否能容纳双缓冲：

```text
baseM × baseN × sizeof(L0C) × 2 <= L0C size
```

满足则 `dbL0C=2`，否则 `dbL0C=1`。

### 10.2 `CalcLeftL1Size()`

从总 L1 中扣除需要常驻的 bias 或 scale 空间：

```text
leftL1Size = totalL1Size - bias/scale reserved bytes
```

MX scale tile 的空间会在 `CalL1Depth()` 中和 A/B tile 一起计算。

### 10.3 单份 tile 的字节数

```cpp
baseASize = bytes(baseM × baseK, aDtype);
baseBSize = bytes(baseN × baseK, bDtype);
```

MX 模式还计算：

```text
baseScaleASize ≈ baseM × ceil(baseK / 32) × sizeof(perTokenScale)
baseScaleBSize ≈ baseN × ceil(baseK / 32) × sizeof(scale)
```

`MX_GROUP_SIZE=32`，表示每 32 个 K 元素对应一组 MX scale。实际实现还会对 scale K 维进行偶数/布局对齐。

### 10.4 `depthA1/depthB1`

`depthA1/depthB1` 表示 L1 流水中能够容纳的 A/B tile 深度。Basic API 路径主要考虑：

- L1 总容量；
- A/B/scale tile 大小；
- 至少约 64 KB 连续 GM 读取有利于获得较高带宽；
- depth 尽量按 2 的幂调整；
- K 尾块非对齐时，尝试利用剩余 L1 增加一侧 depth；
- A/B 和其 scale 的覆盖关系。

如果按带宽目标计算出的 depth 使 L1 超限，会回退到按容量均分得到的 `depthInit`。

### 10.5 `CalStepKs()`

从 depth 推导一次 L1 装载包含多少个 `baseK`：

```cpp
stepKa = depthA1 == 1 ? 1 : depthA1 / 2;
stepKb = depthB1 == 1 ? 1 : depthB1 / 2;
```

这里除以 2 对应双缓冲语义。随后还会：

- 避免 `stepK × baseK` 超过实际 K 太多；
- 调整 `stepKa/stepKb` 的倍数关系；
- 重新令 `depthA1=stepKa×2`、`depthB1=stepKb×2`。

最终写给 MX kernel 的是：

```cpp
kAL1 = stepKa * baseK;
kBL1 = stepKb * baseK;
```

例如：

```text
baseK=128, stepKa=4, stepKb=2
kAL1=512, kBL1=256
```

表示 A 一次 L1 装载覆盖 512 个 K 元素，B 覆盖 256 个 K 元素。

### 10.6 `CalScaleFactors()`

MX scale 也沿 K 方向分块。该函数计算 `scaleFactorA/B`，主要受三类约束：

1. 整个 K 需要多少次 `stepK×baseK` 才能覆盖；
2. 单次 scale 搬运尽量达到高带宽数据量，但不能超过上限 127；
3. A/B tile 占用后剩余的 L1 空间。

最终：

```cpp
scaleKAL1 = min(
    max(scaleFactorA * stepKa, scaleFactorB * stepKb) * baseK,
    K);
scaleKBL1 = scaleKAL1;
```

它表示一次 scale 装载对应的 K 覆盖范围。Basic API 对 A/B scale 使用相同的最终覆盖长度。

## 11. 最终 TilingData

MX Basic API 使用：

```cpp
struct GMMQuantBasicApiTilingData {
    GMMQuantParams gmmQuantParams;
    QuantBasicApiMMTiling mmTilingData;
};
```

其中 `QuantBasicApiMMTiling` 的关键字段：

```cpp
uint32_t m;
uint32_t n;
uint32_t k;
uint32_t baseM;
uint32_t baseN;
uint32_t baseK;
uint32_t kAL1;
uint32_t kBL1;
uint32_t scaleKAL1;
uint32_t scaleKBL1;
uint8_t isBias;
uint8_t dbL0C;
```

`PostTiling()` 调用 `SaveTilingDataToContext()`：

1. 设置 block dim；
2. 将结构体复制到 `context_->GetRawTilingData()`；
3. 设置 tiling data size。

kernel 的 `GM_ADDR tiling` 最终指向这份序列化数据。

## 12. TilingKey 和 kernel 映射

量化 tiling key 由三个参数组成：

```cpp
GET_TPL_TILING_KEY(transB, transA, kernelType)
```

`kernelType`：

| 值 | 含义 | 主要 kernel |
| ---: | --- | --- |
| 0 | fixpipe/随路反量化 | `GmmASWKernel` 或 MX Basic API kernel |
| 1 | vector dequant/mix | `GQmmMixRegbaseKernel` |
| 2 | perGroup-perBlock | `GmmCgmctPerTileKernel` |

MX 模式在 `SetKernelType()` 中保持 `kernelType=0`。结合 `scale` dtype 的编译宏，A5 kernel 总入口进入：

```text
grouped_matmul_apt.cpp::grouped_matmul()
  └─ V310_GMM_QUANT
      └─ V310_GMM_QUANT_MX
          └─ GMM_QUANT_MX_BASIC_API_IMPL_CLASS(...)
              └─ GmmCgmctMxKernel(...)
                  └─ Kernel::KernelQGmmMx
```

宏名 `V310_GMM_QUANT` 不代表这里不是 A5；该宏正是在 A5/AscendC 310 编译条件下选择 arch35 量化实现。

布局映射由 `transA/transB` 和 weight format 决定，例如 ND、A/B 均不转置时：

```text
x layout      = RowMajor
weight layout = RowMajor
y layout      = RowMajor
```

## 13. 非 MX FP8 会走哪里

仅知道 `x/weight=FLOAT8_E4M3FN` 还不能唯一确定 tiling 模板。scale dtype/shape 会继续影响选择：

| scale/量化方式 | tiling 模板/主要 kernel |
| --- | --- |
| `FLOAT8_E8M0`，MX | `GroupedQmmBasicApiTiling` → `GmmCgmctMxKernel` |
| `UINT64/INT64` 或 fixpipe 场景 | `GroupedQmmTiling` → `GmmASWKernel` |
| BF16/FP32 scale、per-token/vector 后处理 | `GroupedQmmTiling` → `GQmmMixRegbaseKernel` |
| perGroup-perBlock | `GroupedQmmTiling` → `GmmCgmctPerTileKernel` |

模板 1 的总体过程和 MX Basic API 相似，也使用：

```text
GetShapeAttrsInfo()
DoOpTiling()
DoLibApiTiling()
  ├─ CalBasicBlock()
  └─ CalL1Tiling()
```

但最终 tiling data 是 `GMMQuantTilingData`，包含 `GMMArray` 和完整的 `TCubeTiling`，L1 depth/scale 计算细节也针对更多模式做了兼容。

## 14. 推荐阅读顺序

如果目标是理解 MXFP8 的切块参数，建议按以下顺序阅读：

1. `TilingGMM()`：确认进入 A5 量化注册表；
2. `GroupedQmmTiling::GetShapeAttrsInfo()`：观察 dtype/attr/input 解析链；
3. `GroupedQmmTiling::SetMKN()`：确认 M/N/K；
4. `GroupedQmmTiling::SetQuantMode()`：确认是 `MX_PERGROUP_MODE`；
5. `GroupedQmmBasicApiTiling::IsCapable()`：确认命中模板 0；
6. `GroupedQmmBasicApiTiling::DoLibApiTiling()`：核心入口；
7. `GroupedQmmTiling::CalBasicBlock()`：计算 `baseM/baseN/baseK`；
8. `GroupedQmmBasicApiTiling::CalL1Tiling()`：计算 L1 depth；
9. `GroupedQmmTiling::CalStepKs()`：计算 `stepKa/stepKb`；
10. `GroupedQmmBasicApiTiling::CalScaleFactors()`：计算 MX scale 覆盖范围；
11. `GroupedQmmTiling::GetTilingKey()`：确认 kernel key；
12. `grouped_matmul_apt.cpp`：对照 key 和编译宏找到 kernel 分支。

## 15. 调试时建议打印的参数

为了确认实际命中的路径，建议重点观察：

```text
npuArch
xDtype / weightDtype / yDtype
scaleDtype / perTokenScaleDtype
x/weight origin shape 和 storage shape
weight format
transA / transB
groupType / groupListType / groupNum
aQuantMode / bQuantMode
kernelType
M / N / K
baseM / baseN / baseK
depthA1 / depthB1
stepKa / stepKb
kAL1 / kBL1
scaleFactorA / scaleFactorB
scaleKAL1 / scaleKBL1
dbL0C
tilingKey
```

判断 MXFP8 路径的最短检查链：

```text
npuArch == DAV_3510
xDtype == FLOAT8_E4M3FN
weightDtype == FLOAT8_E4M3FN
scaleDtype == FLOAT8_E8M0
aQuantMode == MX_PERGROUP_MODE
bQuantMode == MX_PERGROUP_MODE
priority 0 / GroupedQmmBasicApiTiling 命中
kernelType == 0
kernel 进入 V310_GMM_QUANT_MX
```

## 16. 常见误区

### 16.1 `baseK=128` 不一定表示 128 字节

`baseK` 是元素数。只有 FP8/INT8 下 `baseK=128` 才正好是 128 B；FP16 下 `baseK=64` 才是 128 B，FP4 下 `baseK=256` 才是 128 B。

### 16.2 向上对齐后可以超过实际尾块

例如 FP8 MX 的 `K=20`，`baseK` 可能向上对齐到 64。它描述 kernel tile 的物理/指令粒度，实际有效元素仍只有 20，尾部由 kernel 处理。

### 16.3 `singleCoreM` 不等于整个算子的单核最终工作量

这里的 `singleCoreM/N` 与基础 tile 和 scheduler 配合使用。实际 group/tile 到核的映射由 kernel scheduler 继续完成，不能只看一个字段推断完整分核策略。

### 16.4 32 字节对齐不是把维度单位改成字节

维度仍以元素计数。代码只是把 32 B 换算为当前 dtype 的元素数，再对维度做 `CeilAlign()`。

### 16.5 V4 不对应一份独立 kernel

V4 在 API 层以 `GMMApiVersion::V4` 做参数约束和执行器构造，最终仍调用同一个 L0 `GroupedMatmul`，并与其他版本共享 host tiling 和 A5 kernel。
