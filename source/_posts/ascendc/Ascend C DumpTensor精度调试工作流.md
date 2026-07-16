## 1. 工具简介

`compare_dumps.py` 用于读取程序运行过程中由 `AscendC::DumpTensor` 生成的日志，并对日志中的 DumpTensor 数据进行解析和对比。

使用时，用户需要在自己的算子代码中选择需要观测的位置，并自行添加 `AscendC::DumpTensor` 指令。

## 2. 在算子代码中添加 DumpTensor

在需要导出 Tensor 数据的位置加入如下调用：

```cpp
AscendC::DumpTensor(tensor, desc, dumpSize);
```

其中：

- `tensor`：需要导出的 Tensor；
- `desc`：该份 Dump 数据的标识，用于区分不同输入、中间结果或输出；
- `dumpSize`：需要导出的数据长度，应根据 Tensor 的实际大小以及 `DumpTensor` 接口要求填写。

例如：

```cpp
AscendC::DumpTensor(inputTensor, 1, inputSize);
AscendC::DumpTensor(outputTensor, 2, outputSize);
```

建议遵循以下规则：

- 不同含义的 Tensor 使用不同的 `desc`；
- 同一种数据在不同位置或不同版本代码中尽量保持相同的 `desc`；
- 不要让同一个 `desc` 同时表示多种不同含义的数据，否则会影响日志识别和结果对比；
- 添加或修改 `DumpTensor` 后，需要重新编译并运行程序，生成新的日志。

## 3. 配置 desc_labels.json

`desc_labels.json` 用于定义 `desc` 与可读标签之间的对应关系。`compare_dumps.py` 会根据该文件，将日志中的数字 `desc` 转换为对应的数据名称。

在运行 `compare_dumps.py` 之前，用户必须根据自己代码中的 `DumpTensor` 调用修改 `desc_labels.json`。

配置时需要保证：

1. `desc_labels.json` 中定义的 `desc` 与代码中 `DumpTensor` 的第二个参数完全一致；
2. 代码中需要分析的每个 `desc` 都应在配置文件中定义对应标签；
3. 修改代码中的 `desc` 后，应同步更新 `desc_labels.json`；
4. 标签应准确描述对应 Tensor 的含义，便于查看和区分对比结果。

例如，代码中包含：

```cpp
AscendC::DumpTensor(inputTensor, 1, inputSize);
AscendC::DumpTensor(qkResultTensor, 10, qkResultSize);
AscendC::DumpTensor(outputTensor, 40, outputSize);
```

则需要按照 `desc_labels.json` 的现有格式，为 `1`、`10` 和 `40` 分别定义对应标签，例如 `input`、`qk_result` 和 `output`。

> 注意：这里的数字必须与代码中的 `desc` 完全一致。标签名称可以由用户自行定义，但建议保持简洁、明确且不重复。

## 4. 使用流程

### 步骤 1：修改算子代码

在自己的算子代码中找到需要观测的数据位置，添加 `AscendC::DumpTensor` 指令，并为每类数据分配合适的 `desc`。

### 步骤 2：编译并运行程序

重新编译算子或工程，然后运行程序，使程序生成包含 DumpTensor 数据的日志。

### 步骤 3：修改标签配置

在运行对比脚本之前，修改 `desc_labels.json`，确保其中的标签定义与代码中的所有 `desc` 对应。

### 步骤 4：运行对比脚本

```bash
python3 compare_dumps.py
```

`compare_dumps.py` 会自动读取程序生成的日志，识别其中的 DumpTensor 记录，并依据 `desc_labels.json` 中的配置显示和对比相应数据。

## 5. 注意事项

- 请先运行加入 `DumpTensor` 的程序，确认已经生成日志，再执行 `compare_dumps.py`；
- 运行脚本前必须检查 `desc_labels.json`，避免继续使用旧代码对应的标签配置；
- 如果日志中存在未配置的 `desc`，对应数据可能无法显示为预期标签；
- 如果 `desc_labels.json` 中的编号与代码不一致，脚本可能将数据标记错误，进而影响对比结果的可读性；
- 若程序多次运行会保留历史日志，建议在对比前确认脚本读取的是本次运行生成的日志，必要时清理无关旧日志；
- `DumpTensor` 仅建议用于调试和验证。完成问题定位后，可根据需要移除相关调用，以免产生额外日志或影响运行效率。

## 6. 文件说明

- `compare_dumps.py`：自动读取程序生成的日志，解析并对比 DumpTensor 数据；
- `desc_labels.json`：定义代码中 `desc` 与可读标签之间的映射关系。

用户只需在自己的算子代码中添加 `DumpTensor`，并维护与之匹配的 `desc_labels.json`。
