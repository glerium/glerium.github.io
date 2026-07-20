---
title: "Ascend C Printf 上板时延测量工作流"
date: 2026-07-16 20:26:00
categories:
    - 学习笔记
tags:
    - AI Infra
    - Ascend C
    - 算子开发
    - 使用文档
    - 含AI生成
---

## 1. 背景

在 Ascend 950 (A5) 芯片上，部分 profiling 工具展示的指令级 cycle 数据可能来自静态查表，适合观察指令组成，但无法完整反映实际运行时的 pipeline stall、同步等待、访存延迟以及跨核通信开销。

为了测量代码在硬件上的实际执行周期，可以使用：

- `GetSystemCycle()` 读取硬件 cycle；
- `PipeBarrier<PIPE_ALL>()` 排空相关流水线；
- `printf` 输出开始时间、结束时间和持续周期。

该方法适合定位某段代码的真实耗时，以及不同执行阶段之间的等待和流水间隙。

<!--more-->

---

## 2. 基本测量方法

在需要测量的代码前后加入以下逻辑：

```cpp
PipeBarrier<PIPE_ALL>();  // 等待前序操作完成
uint64_t start = GetSystemCycle();

// ... 被测代码 ...

PipeBarrier<PIPE_ALL>();  // 等待被测操作完成
uint64_t end = GetSystemCycle();

if (GetBlockIdx() == 0) {
    printf("[PROBE_NAME] start=%lu end=%lu dur=%lu\n",
        start, end, end - start);
}
```

其中：

- 第一个 `PipeBarrier<PIPE_ALL>()` 用于排空前序流水，避免前面的异步操作计入测量区间；
- 第二个 `PipeBarrier<PIPE_ALL>()` 用于等待被测操作真正完成；
- `start` 和 `end` 为绝对 cycle；
- `dur` 为被测区间的持续周期；
- `PROBE_NAME` 应替换为能够描述测量内容的标签。

需要注意，`PipeBarrier<PIPE_ALL>()` 本身会改变代码原有的流水执行方式。因此，这种测量更适合定位局部阶段耗时，而不应认为加入 probe 后的整体执行时序与原程序完全一致。

---

## 3. Probe 管理方式

建议所有 probe 默认关闭，并使用统一格式管理：

```cpp
#if 0  // TIMING_PROBE: PROBE_NAME
PipeBarrier<PIPE_ALL>();
uint64_t probeStart = GetSystemCycle();
#endif

// ... 被测代码 ...

#if 0  // TIMING_PROBE: PROBE_NAME
PipeBarrier<PIPE_ALL>();
uint64_t probeEnd = GetSystemCycle();

if (GetBlockIdx() == 0) {
    printf("[PROBE_NAME] start=%lu end=%lu dur=%lu\n",
        probeStart, probeEnd, probeEnd - probeStart);
}
#endif
```

需要启用某个 probe 时，将对应的：

```cpp
#if 0
```

改为：

```cpp
#if 1
```

使用时应注意：

1. 每次只启用一个 probe；
2. 同一 probe 的开始和结束部分必须同时启用；
3. 所有 probe 使用统一注释格式，便于全局搜索；
4. 调试结束后应恢复为默认关闭状态。

例如，可以统一搜索：

```text
TIMING_PROBE:
```

然后只启用本次需要测量的 probe。

每次只启用一个 probe，可以减少额外 `PipeBarrier` 和 `printf` 对执行时序的干扰。

其中，`printf` 的单次开销实际上很大，虽然在相同环境下通常相对稳定，但多个 probe 同时启用时，前一个 probe 的打印开销会延迟后续代码执行，使后续 probe 的 `start`、`end` 以及阶段间的 Pipeline gap 混入额外的 `printf` 开销，进而污染统计结果。因此，不同 probe 应分别编译或分别运行采集，不能通过一次运行同时启用多个 probe 来还原原始执行时间线。

---

## 4. 输出范围控制

为避免多核同时输出大量日志，建议仅让一个 Block 打印：

```cpp
if (GetBlockIdx() == 0) {
    printf(...);
}
```

需要注意，`GetBlockIdx() == 0` 仅限制 Block 编号。在同时包含不同核类型或子核的执行模型中，仍可能有多个执行单元满足该条件。

如果代码同时运行在不同核类型或不同子核上，应根据实际执行模型进一步限制输出范围，确保日志来源明确。

需要观察多个 Block 时，可以将 Block 编号加入日志：

```cpp
uint32_t blockIdx = GetBlockIdx();

printf("[PROBE_NAME] block=%u start=%lu end=%lu dur=%lu\n",
    blockIdx, start, end, end - start);
```

循环内部的 probe 建议额外输出迭代编号：

```cpp
printf("[PROBE_NAME] block=%u iter=%u start=%lu end=%lu dur=%lu\n",
    blockIdx, iter, start, end, end - start);
```

这样可以区分不同 Block 和不同迭代产生的日志。

---

## 5. 异步边界原则

对于包含异步流水、跨核同步或生产者—消费者关系的代码，不应简单地把外层函数调用返回视为阶段完成。

真实边界通常应由以下事件确定：

- 数据生产阶段结束：以数据真正写入目标位置，或同步通知发出为准；
- 数据消费阶段开始：以等待同步信号返回为准；
- 异步搬运结束：以对应流水完成或同步事件完成为准；
- 跨核阶段切换：以 `SetCrossCoreSync`、`WaitCrossCoreSync` 等实际同步点为准。

### 5.1 生产阶段

例如，一个阶段在计算完成后通过同步信号通知下游：

```cpp
uint64_t start = GetSystemCycle();

// 数据计算、搬运或流水操作
// ...

SetCrossCoreSync(flag);

uint64_t end = GetSystemCycle();
```

该区间测量的是从阶段入口到通知下游可以继续执行的真实时间，而不是某个封装函数在主流程中的局部执行时间。

### 5.2 消费阶段

对于消费阶段，可以在等待同步完成后读取开始时间：

```cpp
WaitCrossCoreSync(flag);
uint64_t start = GetSystemCycle();

// 消费数据
// ...

SetCrossCoreSync(nextFlag);
uint64_t end = GetSystemCycle();
```

这样可以避免把等待上游完成的时间错误计入消费阶段本身。

如果希望单独分析等待时间，可以将等待过程拆成独立 probe：

```cpp
uint64_t waitStart = GetSystemCycle();

WaitCrossCoreSync(flag);

uint64_t waitEnd = GetSystemCycle();

if (GetBlockIdx() == 0) {
    printf("[STAGE_WAIT] start=%lu end=%lu dur=%lu\n",
        waitStart, waitEnd, waitEnd - waitStart);
}
```

### 5.3 外层测量与内部测量的区别

下面这种外层测量：

```cpp
uint64_t start = GetSystemCycle();

RunStage();

uint64_t end = GetSystemCycle();
```

只能表示 `RunStage()` 在当前调用方看来经历了多长时间。

如果 `RunStage()` 内部包含异步任务提交，那么函数返回时，硬件上的实际计算可能尚未完成。

因此，对于异步代码，应尽量把 probe 放到真正的同步边界附近，而不是只测量外层函数调用。

---

## 6. 绝对 Cycle 对齐

由于共享函数或深层函数无法直接访问Host侧的全局变量，probe 只能输出绝对 cycle。为了比较不同位置的时间关系，可以在主流程入口记录一个统一基准：

```cpp
uint64_t base = GetSystemCycle();

if (GetBlockIdx() == 0) {
    printf("[T_BASE] %lu\n", base);
}
```

其他 probe 输出：

```text
[STAGE_A] start=1049000 end=1049500 dur=500
[STAGE_B] start=1050500 end=1050800 dur=300
```

相对主流程入口的 offset 可按以下方式计算：

```text
relative_start = probe_start - T_BASE
relative_end   = probe_end   - T_BASE
```

例如：

```text
[T_BASE] 1048576
[STAGE_A] start=1049000 end=1049500 dur=500
```

则：

```text
STAGE_A relative start = 1049000 - 1048576 = 424 cycles
```

这表示该阶段在基准点之后 424 cycles 开始。

如果要绘制完整时间线，可以将每个 probe 转换为：

```text
[relative_start, relative_end]
```

例如：

```text
STAGE_A: [424, 924]
```

需要注意，不同核、不同子核或不同硬件计时域读取到的绝对 cycle，不一定天然可以直接比较。进行跨核时间线分析前，应先确认这些执行单元的 cycle 计数器是否具有共同基准。

---

## 7. Probe 设计建议

可以按以下层次布置 probe：

```text
主流程入口
  │ [T_BASE]
  │
  ├─ 阶段 A
  │   ├─ [STAGE_A]
  │   ├─ [STAGE_A_WAIT]
  │   └─ [STAGE_A_COMPUTE]
  │
  ├─ 阶段 B
  │   ├─ [STAGE_B]
  │   ├─ [STAGE_B_LOAD]
  │   └─ [STAGE_B_WRITEBACK]
  │
  └─ 阶段 C
      ├─ [STAGE_C]
      └─ [STAGE_C_SYNC]
```

建议先添加粗粒度 probe，确定主要瓶颈所在阶段，再在耗时较大的阶段内部增加细粒度 probe。

常见 probe 类型包括：

| Probe 类型 | 测量内容 |
|---|---|
| 阶段总耗时 | 某个执行阶段从真实入口到真实完成边界的时间 |
| 等待耗时 | 等待跨核同步、流水事件或资源就绪的时间 |
| 计算耗时 | 不包含等待的核心计算时间 |
| 数据搬运耗时 | GM、UB、L1、L0 等存储层级之间的数据移动时间 |
| 写回耗时 | 结果转换、搬运并写回 GM 的时间 |
| Pipeline gap | 上一阶段结束到下一阶段开始之间的间隔 |

推荐按照以下顺序逐步细化：

1. 先测量完整阶段；
2. 判断哪个阶段耗时最大；
3. 将该阶段拆分为等待、计算和搬运；
4. 判断耗时主要来自同步等待还是实际计算；
5. 继续细化最耗时的子阶段。

不要一开始就在大量位置同时插入 probe，否则 `PipeBarrier` 和 `printf` 本身会明显改变执行过程。

---

## 8. Warmup 与输出抑制

### 8.1 为什么至少需要一次 Warmup

进行周期测量时，正式计时前建议至少执行一次 warmup，不要直接将第一次算子执行结果作为稳定性能数据。

第一次执行可能受到以下一次性或冷启动因素影响：

- 运行时、执行器或算子相关资源尚未完全初始化；
- 工作空间、页表或相关设备资源首次访问；
- 代码、指令或数据缓存处于冷状态；
- 流水线和跨核同步尚未进入相对稳定状态；
- 首次执行可能包含后续迭代不会重复出现的额外开销。

因此，当 `warmup=0` 时，probe 测得的 cycle 往往波动更明显，单次结果的误差也可能偏大。

除非正在专门分析冷启动行为，否则建议：

```text
warmup >= 1
```

调试 kernel 内部 probe 时，为减少日志量，可以使用：

```text
warmup = 1
repeat = 1
```

例如，在 BSA benchmark 中可以这样运行：

```bash
./benchmark_bsa \
  --batch 1 \
  --q-seq 256 \
  --kv-seq 256 \
  --heads 1 \
  --kv-heads 1 \
  --head-dim 128 \
  --block-x 128 \
  --block-y 128 \
  --density 1.0 \
  --warmup 1 \
  --repeat 1 \
  --inner-precise 4 \
  --device 4
```

需要获得更稳定的整体性能统计时，可以增加 warmup 和 repeat，例如：

```text
warmup >= 3
repeat >= 10
```

Warmup 仅用于让算子运行状态趋于稳定，不应计入最终耗时统计。

### 8.2 为什么需要抑制 Warmup 输出

如果 kernel 中加入了 `printf` probe，那么 warmup 阶段和正式测量阶段都会产生相同格式的日志。

例如：

```text
[STAGE_A] start=... end=... dur=...
[STAGE_A] start=... end=... dur=...
```

这些输出无法直接判断哪一条来自 warmup，哪一条来自正式测量，也可能干扰后续日志解析。

因此，可以在 warmup 阶段临时将进程标准输出 `stdout` 重定向到 `/dev/null`：

```text
保存当前 stdout
    ↓
将 stdout 重定向到 /dev/null
    ↓
执行所有 warmup
    ↓
同步 stream，等待 warmup kernel 执行完成
    ↓
等待异步 kernel printf 刷出
    ↓
恢复原来的 stdout
    ↓
开始正式计时
```

在 `stdout` 指向 `/dev/null` 期间，写入标准输出的内容会被丢弃。

### 8.3 BSA Benchmark 中的实现

BSA benchmark 使用以下代码抑制 warmup 阶段的输出：

```cpp
// 先刷新 stdout，避免重定向前的日志被误吞。
fflush(stdout);

// 备份当前标准输出文件描述符。
// stdout 可能指向终端，也可能指向用户重定向的日志文件。
int stdoutBak = dup(STDOUT_FILENO);

// 打开 /dev/null。
int devNull = open("/dev/null", O_WRONLY);

// 将 STDOUT_FILENO 重定向到 /dev/null。
dup2(devNull, STDOUT_FILENO);

// dup2 完成后，STDOUT_FILENO 已持有对应引用，
// 原始 devNull 文件描述符可以关闭。
close(devNull);

// Warmup 阶段。
// 此时 stdout 指向 /dev/null，相关输出不会显示。
for (int32_t i = 0; i < args.warmup; ++i) {
    int launchRet = launchOnce();
    if (launchRet != ACL_SUCCESS) {
        return launchRet;
    }
}

// 算子调用通常是异步提交的。
// 必须等待 warmup kernel 真正执行完成后才能恢复 stdout。
CHECK_ACL(aclrtSynchronizeStream(stream));

// kernel printf 或设备日志可能通过异步日志通路送达。
// 即使 stream 已同步完成，日志仍可能稍晚才刷新到主机。
// 因此在 stdout 仍指向 /dev/null 时额外等待一段时间。
usleep(500000);

// 清理静默阶段可能残留的 stdout 用户态缓冲。
fflush(stdout);

// 恢复原来的标准输出。
dup2(stdoutBak, STDOUT_FILENO);

// 关闭备份文件描述符。
close(stdoutBak);
```

恢复 stdout 后，再开始正式计时：

```cpp
aclrtEvent startEvent = nullptr;
aclrtEvent endEvent = nullptr;

CHECK_ACL(aclrtCreateEvent(&startEvent));
CHECK_ACL(aclrtCreateEvent(&endEvent));

CHECK_ACL(aclrtRecordEvent(startEvent, stream));

for (int32_t i = 0; i < args.repeat; ++i) {
    int launchRet = launchOnce();
    if (launchRet != ACL_SUCCESS) {
        return launchRet;
    }
}

CHECK_ACL(aclrtRecordEvent(endEvent, stream));
CHECK_ACL(aclrtSynchronizeStream(stream));

float elapsedMs = 0.0f;

CHECK_ACL(
    aclrtEventElapsedTime(
        &elapsedMs,
        startEvent,
        endEvent));

double averageUs =
    static_cast<double>(elapsedMs) * 1000.0 / args.repeat;
```

完整执行顺序为：

```text
重定向 stdout
    ↓
执行 warmup
    ↓
aclrtSynchronizeStream
    ↓
等待 kernel printf 异步刷新
    ↓
恢复 stdout
    ↓
记录 startEvent
    ↓
执行正式 repeat
    ↓
记录 endEvent
    ↓
同步并计算正式耗时
```

### 8.4 需要包含的头文件

上述代码使用了 `dup`、`dup2`、`close`、`usleep`、`open` 和 `O_WRONLY`，需要包含：

```cpp
#include <cstdio>
#include <fcntl.h>
#include <unistd.h>
```

其中：

- `<cstdio>` 提供 `fflush` 和 `stdout`；
- `<fcntl.h>` 提供 `open` 和 `O_WRONLY`；
- `<unistd.h>` 提供 `dup`、`dup2`、`close` 和 `usleep`。

---

## 9. Warmup 输出抑制的常见陷阱

### 9.1 完全不进行 Warmup

```bash
./benchmark_bsa ... --warmup 0
```

这种情况下，第一次执行中的初始化、冷缓存和首次资源访问开销都会进入测量结果。

这通常会导致 cycle 波动更加明显，测量误差偏大。

除非专门测试冷启动，否则建议至少使用：

```bash
--warmup 1
```

### 9.2 Warmup 循环结束后不进行 Stream 同步

下面的写法存在问题：

```cpp
for (int32_t i = 0; i < args.warmup; ++i) {
    launchOnce();
}

// 错误：设备上的 warmup kernel 可能还没有执行完成。
dup2(stdoutBak, STDOUT_FILENO);
```

因为 `launchOnce()` 可能只完成了异步提交。

恢复 stdout 前必须调用：

```cpp
aclrtSynchronizeStream(stream);
```

### 9.3 Stream 同步完成后立即恢复 Stdout

某些 kernel 日志并不会随着 stream 同步立即出现在主机端。

如果同步后立刻恢复 stdout，仍可能出现少量迟到的 warmup 输出：

```cpp
aclrtSynchronizeStream(stream);

// 可能恢复得过早。
dup2(stdoutBak, STDOUT_FILENO);
```

可以在 stdout 仍被抑制时增加一个较短的日志刷新窗口：

```cpp
aclrtSynchronizeStream(stream);
usleep(500000);
dup2(stdoutBak, STDOUT_FILENO);
```

### 9.4 重定向前没有调用 `fflush`

stdout 在连接终端时通常采用行缓冲，在重定向到文件时可能采用全缓冲。

如果重定向前不调用：

```cpp
fflush(stdout);
```

此前尚未刷新到文件描述符的数据，也可能被写入 `/dev/null`。

### 9.5 异常返回时没有恢复 Stdout

以下写法存在问题：

```cpp
for (int32_t i = 0; i < args.warmup; ++i) {
    int launchRet = launchOnce();

    if (launchRet != ACL_SUCCESS) {
        return launchRet;
    }
}
```

如果 warmup 执行失败并直接 `return`，stdout 此时仍指向 `/dev/null`。

后续错误信息和清理日志也可能全部丢失。

更稳妥的方法是：

- 在异常返回前显式恢复 stdout；
- 或使用 RAII 类，在作用域退出时自动恢复 stdout。

### 9.6 只运行一次就下结论

即使已经进行了 warmup，单次执行仍可能受到系统负载、设备调度或其他运行时因素影响。

对于重要结论，建议进行多次测量，并观察：

- 最小值；
- 中位数；
- 平均值；
- 最大值；
- 标准差或波动范围。

---

## 10. 编译与运行建议

添加 probe 后，需要重新编译并运行程序。

具体编译命令和运行参数由用户当前工程决定。

建议调试时：

- 至少执行一次 warmup；
- 调试单个 probe 时减少 repeat 次数；
- 使用较小且稳定的输入；
- 固定设备、输入 shape 和运行参数；
- 避免同时运行其他高负载任务；
- 确认 warmup 不在正式计时区间内；
- 多次运行并比较结果，排除偶然波动。

仅调试 kernel 内部 probe、希望减少日志量时，可以使用：

```text
warmup = 1
repeat = 1
```

需要获得较稳定的性能统计时，可以使用：

```text
warmup >= 3
repeat >= 10
```

如果需要测量 kernel 内部不同迭代的周期，应谨慎设置 `repeat`。

`repeat` 大于 1 时，每次 kernel 执行都会输出 probe 日志，容易产生大量重复内容。此时可以：

- 将 `repeat` 设置为 1；
- 或在日志中加入执行轮次；
- 或只对某一轮执行启用 probe。

---

## 11. 数据解读

输出示例：

```text
[T_BASE] 1048576
[STAGE_A] start=1049000 end=1049500 dur=500
[STAGE_B_WAIT] start=1049510 end=1050500 dur=990
[STAGE_B_COMPUTE] start=1050500 end=1050800 dur=300
[STAGE_C] start=1051000 end=1051600 dur=600
```

可以重点分析以下指标：

| 指标 | 计算方式 | 含义 |
|---|---|---|
| 阶段耗时 | 对应 probe 的 `dur` | 某阶段的完整实际执行时间 |
| 等待占比 | `WAIT dur / 阶段总 dur` | 判断瓶颈是否来自同步等待 |
| 计算占比 | `COMPUTE dur / 阶段总 dur` | 判断核心计算是否为主要瓶颈 |
| 搬运占比 | `LOAD` 或 `WRITEBACK` 的 `dur` | 判断数据移动是否拖慢执行 |
| Pipeline gap | 下一阶段 `start` 减上一阶段 `end` | 判断阶段之间是否存在空闲或通信延迟 |
| 相对启动时间 | `probe start - T_BASE` | 判断各阶段在完整执行流程中的位置 |

### 11.1 阶段耗时

```text
stage_duration = stage_end - stage_start
```

通常直接对应日志中的 `dur`。

### 11.2 等待占比

```text
wait_ratio = wait_duration / total_stage_duration
```

如果等待占比较高，说明该执行单元的大量时间用于等待：

- 上游数据；
- 跨核同步；
- 流水事件；
- 数据搬运完成；
- 共享资源释放。

### 11.3 计算占比

```text
compute_ratio = compute_duration / total_stage_duration
```

如果计算占比较高，说明瓶颈主要位于实际计算过程。

如果阶段总耗时很大，但计算占比较低，则更可能是同步或访存问题。

### 11.4 Pipeline Gap

```text
pipeline_gap =
    next_stage_start - previous_stage_end
```

例如：

```text
[STAGE_A] end=1049500
[STAGE_B] start=1050500
```

则：

```text
pipeline_gap = 1050500 - 1049500 = 1000 cycles
```

这个间隔可能来自：

- 跨核同步传播；
- 下游调度延迟；
- 数据尚未可见；
- 资源冲突；
- 输出日志本身造成的扰动；
- 两个 probe 不在完全相同的计时域。

因此，Pipeline gap 应结合代码中的同步关系共同分析。

### 11.5 结果波动

如果某个 probe 的持续时间波动明显，应增加运行次数，观察最小值、中位数和最大值，而不是只依据单次结果判断。

对于性能优化，最小值通常可以反映系统干扰较小时的执行能力，中位数更适合表示常见运行水平。

---

## 12. 注意事项

- `PipeBarrier<PIPE_ALL>()` 会改变原始执行时序，因此 probe 本身可能产生额外开销；
- 一次运行只启用一个 probe，不要同时采集多个 probe；
- `printf` 的单次开销很大，虽然通常相对稳定，但多个 `printf` 会累积并延迟后续代码执行，使后续 probe 的时间戳和 Pipeline gap 混入打印开销；
- `printf` 也会引入额外开销，应尽量减少输出次数；
- 正式测量前建议至少执行一次 warmup；
- warmup 不应计入正式计时区间；
- 应抑制或明确标记 warmup 阶段的输出；
- 对异步代码，应以真实同步点作为测量边界；
- 外层函数返回不一定代表设备侧异步任务完成；
- 不同核或不同流水线读取到的绝对 cycle，应先确认是否可以直接比较；
- 循环内部的 probe 应输出迭代编号，便于区分不同轮次；
- 完成调试后，应关闭或移除 probe，避免影响正式运行性能。
