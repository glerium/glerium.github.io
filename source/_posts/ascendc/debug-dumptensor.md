---
title: "Ascend C DumpTensor精度调试工作流"
date: 2026-07-16 20:25:00
categories:
    - 学习笔记
tags:
    - AI Infra
    - Ascend C
    - 算子开发
    - 使用文档
    - 含AI生成内容
---

## 1. 概述

`AscendC::DumpTensor` 用于在 AscendC Kernel 中导出输入、中间结果或输出 Tensor。配合 `compare_dumps.py`，可以比较基准版本（golden）和待测版本（test）的日志，逐步定位数值差异最早出现的位置。

推荐使用两个 Git worktree 同时维护两个待比较版本。每个 worktree 拥有独立的源码、构建产物和运行环境，不需要反复切换分支，也能避免两个版本互相覆盖。

<!--more-->

典型流程如下：

1. 在两个版本的对应位置添加语义一致的 `DumpTensor`。
2. 分别构建并以完全相同的输入和参数运行。
3. 保存两份完整日志。
4. 使用 `compare_dumps.py` 按 `desc` 比较对应 Tensor。
5. 从输入到输出查找第一个出现差异的位置。

## 2. 前置条件

- 两个待比较的 Git revision，例如分支、tag 或 commit。
- 可正常编译和运行目标 AscendC 算子的环境。
- Python 3 和 NumPy。
- 本目录中的 `compare_dumps.py`；`desc_labels.json` 为可选的标签配置。

检查 Python 依赖：

```bash
python3 -c "import numpy; print(numpy.__version__)"
```

如果未安装 NumPy：

```bash
python3 -m pip install numpy
```

## 3. 在 Kernel 中使用 DumpTensor

### 3.1 函数原型

`DumpTensor` 可以导出 `LocalTensor` 或 `GlobalTensor`，将其打印到标准输出流中：

```cpp
// LocalTensor，例如 UB、L1 或 L0C 中的数据
template <typename T>
__aicore__ inline void DumpTensor(
    const LocalTensor<T> &tensor,
    uint32_t desc,
    uint32_t dumpSize);

// GlobalTensor，例如 GM 中的数据
template <typename T>
__aicore__ inline void DumpTensor(
    const GlobalTensor<T> &tensor,
    uint32_t desc,
    uint32_t dumpSize);
```

| 参数 | 说明 |
|---|---|
| `tensor` | 要导出的 Tensor；使用当前位置对应的 `LocalTensor` 或 `GlobalTensor` |
| `desc` | 用户自定义的非负整数标识，用来区分不同数据或观测位置 |
| `dumpSize` | 从该 Tensor 导出的元素数量，不是字节数 |

`DumpTensor` 没有返回值。`dumpSize` 不应超过当前 Tensor 从指定起点开始的有效元素数。

### 3.2 调用示例

导出 LocalTensor：

```cpp
constexpr uint32_t kInputDesc = 1;
AscendC::DumpTensor(localTensor, kInputDesc, elementCount);
```

导出 GlobalTensor 的一段数据：

```cpp
constexpr uint32_t kOutputDesc = 30;

AscendC::GlobalTensor<half> outputView;
outputView.SetGlobalBuffer((__gm__ half *)outputBuffer + elementOffset);
AscendC::DumpTensor(outputView, kOutputDesc, elementCount);
```

应为不同语义的数据分配不同 `desc`，并在 golden 和 test 版本中保持一致。例如：

```cpp
AscendC::DumpTensor(inputTensor,        1, inputElementCount);
AscendC::DumpTensor(intermediateTensor, 10, intermediateElementCount);
AscendC::DumpTensor(outputTensor,       30, outputElementCount);
```

### 3.3 启用 DumpTensor

构建包含 `DumpTensor` 调用的代码后，在运行前开启 DumpTensor 输出：

```bash
export ASCENDC_DUMP=1
```

关闭时可执行：

```bash
unset ASCENDC_DUMP
```

修改 `DumpTensor` 调用后需要重新构建。若日志中没有相应输出，应先确认程序实际执行到了插桩位置。

### 3.4 输出格式

日志记录通常包含头部和数值数组，例如：

```text
CANN Version: XX.XX, TimeStamp: XXXXXX
DumpTensor: desc=10, addr=0, data_type=float16, position=UB, dump_size=4
[0.125, 0.250, 0.375, 0.500]
```

`compare_dumps.py` 使用下面的头部识别记录：

```text
DumpTensor: desc=<非负整数>,
```

应完整保留头部、方括号和其中的数值，不要手工修改日志。

## 4. desc 标签配置

`desc_labels.json` 将数字 `desc` 转换成报告中的可读名称。假设代码使用 `1`、`10` 和 `30`：

```json
{
  "1": "input",
  "10": "intermediate",
  "30": "output"
}
```

配置时应满足：

- JSON key 与 `DumpTensor` 的第二个参数完全一致。
- 一个 `desc` 只表达一种数据含义。
- golden 和 test 使用相同的编号与语义。
- 修改代码中的编号后同步更新配置。

配置文件不存在或 JSON 无效时，脚本仍可运行，但报告将显示原始编号，例如 `desc=10`。可用以下命令检查 JSON：

```bash
python3 -m json.tool desc_labels.json
```

## 5. 双 worktree 调试策略

### 5.1 为什么使用两个 worktree

如果在单一工作目录中反复切换版本，临时插桩、构建缓存、安装目录和动态库容易相互污染。两个 worktree 可以同时保留 golden 与 test 的源码状态，便于确认两边的插桩和运行参数确实对应。

隔离源码还不够。两个版本应同时使用各自独立的：

- 构建目录；
- 安装或部署目录；
- 运行时库路径；
- 输出日志。

### 5.2 创建 worktree

先进入目标 Git 仓库并更新所需 revision。以下命令使用 detached worktree，适合不提交调试插桩的场景：

```bash
cd <repo-root>

git worktree add --detach <golden-worktree> <golden-ref>
git worktree add --detach <test-worktree> <test-ref>
git worktree list
```

例如，`<golden-ref>` 和 `<test-ref>` 可以是两个 commit、tag 或远端跟踪分支。`<golden-worktree>` 与 `<test-worktree>` 必须是不同且尚未被占用的目录。

如果调试改动需要提交，可去掉 `--detach`，并为两个 worktree 分别使用可写分支。

### 5.3 在两个版本中添加相同观测点

分别进入两个 worktree，在语义对应的位置添加 `DumpTensor`：

```bash
cd <golden-worktree>
# 编辑目标 Kernel，为各观测点分配 desc。

cd <test-worktree>
# 在对应位置使用相同的 desc 和 dumpSize。
```

检查两边时重点确认：

- 对应 `desc` 表示相同阶段的数据；
- Tensor 的数据类型、shape 和有效元素范围一致；
- `dumpSize` 一致；
- 插桩位于相同循环、核或任务语义下。

若两个版本的实现结构不同，应以数据语义对应为准，而不是机械地追求相同行号。

### 5.4 分别构建

在每个 worktree 中执行项目自己的构建命令，并使用不同的构建与安装路径：

```bash
cd <golden-worktree>
<build-command> --build-dir <golden-build-dir> --install-dir <golden-install-dir>

cd <test-worktree>
<build-command> --build-dir <test-build-dir> --install-dir <test-install-dir>
```

`<build-command>` 及其参数是通用占位符，应替换成项目的真实命令。如果构建系统不接受这些选项，也应通过其实际配置方式保证输出目录隔离。

### 5.5 分别运行并保存日志

建议在两个独立终端中设置各自的运行环境，避免残留的库路径指向另一版本：

```bash
# golden 终端
cd <golden-worktree>
export ASCENDC_DUMP=1
export LD_LIBRARY_PATH=<golden-runtime-libs>:${LD_LIBRARY_PATH:-}
<golden-run-command> <identical-run-arguments> > <golden-log> 2>&1
```

```bash
# test 终端
cd <test-worktree>
export ASCENDC_DUMP=1
export LD_LIBRARY_PATH=<test-runtime-libs>:${LD_LIBRARY_PATH:-}
<test-run-command> <identical-run-arguments> > <test-log> 2>&1
```

替换占位符时，应确保两次运行具有相同的：

- 输入数据和随机种子；
- Tensor shape 与数据类型；
- tiling、精度模式和其他算子参数；
- 设备选择、任务数量和循环次数；
- `desc` 与 `dumpSize`。

使用 `>` 覆盖日志，不要使用 `>>` 追加历史结果。建议将日志保存到 worktree 之外，或至少使用明确不同的文件名。

运行后先确认日志确实包含数据：

```bash
grep -n "DumpTensor:" <golden-log>
grep -n "DumpTensor:" <test-log>
```

### 5.6 比较日志

```bash
python3 compare_dumps.py \
  --golden <golden-log> \
  --test <test-log> \
  --desc-config desc_labels.json \
  --threshold 1e-6 \
  --report compare_report.txt
```

## 6. 使用 compare_dumps.py

### 6.1 最简命令

```bash
python3 compare_dumps.py \
  --golden dump_golden.txt \
  --test dump_test.txt
```

脚本默认从当前目录读取 `desc_labels.json`，使用 `1e-6` 作为最大绝对误差阈值，并将结果输出到终端。

### 6.2 参数说明

| 参数 | 必填 | 默认值 | 说明 |
|---|:---:|---|---|
| `--golden` | 是 | 无 | 基准版本生成的 DumpTensor 日志 |
| `--test` | 是 | 无 | 待测版本生成的 DumpTensor 日志 |
| `--desc-config` | 否 | `desc_labels.json` | `desc` 到可读名称的 JSON 映射文件 |
| `--threshold` | 否 | `1e-6` | `MaxAbsDiff` 的通过阈值 |
| `--report` | 否 | 无 | 在终端输出之外，将同一报告写入指定文件 |

查看当前脚本的参数：

```bash
python3 compare_dumps.py --help
```

配置文件不在当前目录时，应显式指定路径：

```bash
python3 compare_dumps.py \
  --golden dump_golden.txt \
  --test dump_test.txt \
  --desc-config <path-to-desc-labels.json>
```

### 6.3 阈值

每个 Tensor 的判断条件是：

```text
MaxAbsDiff <= threshold
```

阈值应来自项目自身的精度标准，而不是仅根据示例值决定。例如，要求完全一致的整数数据可使用 `0`；浮点数据应根据数据类型、算法和误差预算选择阈值。当前脚本对所有 Tensor 使用同一个阈值。

## 7. 结果解读

报告示例：

```text
-----------------------------------------------------------------------------------------
desc  name                        count     MaxAbsDiff           RMSE     Rel(%)   Status
-----------------------------------------------------------------------------------------
      input                        1024     0.0000e+00     0.0000e+00     0.0000     PASS
      intermediate                 4096     2.4414e-04     1.2378e-05     0.0123     DIFF
      output (size mismatch)          0            nan            nan        nan     SKIP
-----------------------------------------------------------------------------------------
Overall: DIFF
-----------------------------------------------------------------------------------------
```

| 列或状态 | 含义 |
|---|---|
| `name` | `desc_labels.json` 中配置的名称，未配置时显示原始 `desc` |
| `count` | 参与比较的元素数量 |
| `MaxAbsDiff` | 对应元素的最大绝对误差，是 PASS/DIFF 的判断依据 |
| `RMSE` | 均方根误差，用于观察整体误差水平 |
| `Rel(%)` | 最大绝对误差相对于 golden 最大绝对值的百分比 |
| `PASS` | `MaxAbsDiff` 不超过阈值 |
| `DIFF` | `MaxAbsDiff` 超过阈值 |
| `SKIP` | 两份日志缺少对应数据或元素数量不一致，无法比较 |

只有所有结果均为 `PASS` 时，`Overall` 才是 `PASS`。出现 `DIFF` 或 `SKIP` 时，整体结果为 `DIFF`。

定位问题时，应按照数据流从输入到输出查看各观测点：最后一个 `PASS` 与第一个 `DIFF` 之间通常是优先排查范围。相对误差在 golden 数据接近零时可能很大，应结合 `MaxAbsDiff`、`RMSE` 和数据范围综合判断。

## 8. 使用限制与最佳实践

- `DumpTensor` 指令当前仅支持打印存储位置为 UB/L1/L0C/Global Memory 的 Tensor 信息。针对 Atlas 350 加速卡，不支持打印 L1 Buffer 上的Tensor信息。
- 单次调用 `DumpTensor` 打印的数据总量不可超过 1MB（还包括少量框架需要的头尾信息，通常可忽略）。使用时应注意，如果超出这个限制，则数据不会被打印。
- 当前工具对每个 `desc` 只输出一项比较结果；同一 `desc` 多次出现时，不保证每条记录都会被比较。因此，一次目标运行中，一个 `desc` 应只产生一条待比较记录。
- 如果需要观察不同位置、迭代、核或任务，应使用不同 `desc`，或者拆成多次独立运行。
- 不要让同一 `desc` 表示多个 shape 相同但语义不同的 Tensor，否则报告可能无法反映预期数据。
- 大 Tensor 会产生大量日志。先使用可复现的小规模输入缩小问题范围，再按需扩大。
- 某些内部存储格式不能按逻辑 Tensor 的线性布局直接解释。应先将数据转换或搬运到适合观察的布局，再调用 `DumpTensor`。
- `DumpTensor` 仅用于调试和验证。完成定位后应移除或禁用插桩，并重新构建正式版本。


## 9. 常见问题

### 9.1 日志中没有 DumpTensor 记录

先检查：

```bash
grep -n "DumpTensor:" <log-file>
```

如果没有结果，确认：

- 已设置 `ASCENDC_DUMP=1`；
- 修改后的 Kernel 已重新构建并被当前程序实际加载；
- 程序执行到了插桩位置；
- 标准输出和标准错误都被保存；
- 运行时库和算子安装路径来自正确的 worktree 构建结果。

### 9.2 出现 size mismatch 或 SKIP

检查两版代码和两次运行的：

- `desc` 是否对应同一种数据；
- `dumpSize`、shape 和数据类型是否一致；
- 输入与运行参数是否一致；
- 日志是否截断、缺失或混入历史运行结果。

### 9.3 两个 worktree 的结果异常相似或交叉污染

确认两个版本没有共用构建目录、安装目录或生成文件，并检查各终端中的动态库和算子加载路径。必要时在干净终端中分别设置环境后重新运行。
