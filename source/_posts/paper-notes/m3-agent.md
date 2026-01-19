---
title: "[论文笔记] Seeing, Listening, Remembering, and Reasoning: A Multimodal Agent with Long-Term Memory"
date: 2026/01/11 16:01
tags:
    - Agent
    - Agent Memory
    - 多模态
    - 多模态Agent
    - 强化学习
categories:
    - 论文笔记
---

目前状态：ICLR 2026在投，得分846

## 方法动机

现有方法受限于两个缺点：

1. 无法应对无限长的上下文：基于上下文摘要的方法依然受到LLM上下文窗口的限制

2. 无法构建世界知识：更加侧重于表面的视觉知识，忽略深层次的语义知识

<!--more-->

## M3-Bench Benchmark

### 动机

先前的 LVQA benchmarks 主要聚焦于视觉理解（动作识别或空间感知），无法应对真实世界中需要长期记忆或者需要提取信息的场景，所以需要构建新的 benchmark

### 数据集

构建了两个 benchmark，包含视频、QA对、回答对应的参考事件（后两者由人工标注）：

- M3-Bench-robot：机器人传感器在日常场景下，以第一人称视角录制的100个视频，涉及机器人与人类、人类之间的互动；每个视频中包含若干个事件

- M3-Bench-web：从网络上抓取的920个视频

问题的标准：明确性、客观性、答案单一无歧义，能完全由视频内容推导得出

### 评测

通过 GPT-4o 进行自动评测，评测器输出 yes or no

## 方法

M3-Agent 包含一个用来回答的MLLM和一个长上下文的记忆组件。运行时记忆进程和控制进程并行启动，一个负责读取环境、写入记忆；另一个负责读取记忆、完成指令。

记忆组织方式：entity-centric，把 entity 和事件按照无向图的形式记录。每个实体对应一个节点，包含以下信息：

- id

- type: text / image (face) / audio (voice)

- content

- embedding: for similarity search

- weight: confidence of node, related to the voting mechanism

- extra_data: JSON metadata

查询方式有两种：

- search_node：查找 top-k 相关的实体

- search_clip：查找 top-k 相关的 memory clip（sementic memory / episodic memory）

### 记忆的实现

记忆分类（遵循上周讲的 Mem-alpha）：

- episodic memory：情境记忆，记录xxx做了xxx事情

- semantic memory：语义记忆，通过提炼情境得到的更高层知识，例如xxx是xxx的朋友

如何解决实体一致性问题？

- 直接在长期记忆中保留原始的多模态信息（face / voice），而不是生成自然语言作为key（后者具有模糊性，在长期记忆中容易出现问题）

- 具体来说，M3-Agent 通过调用面部识别和声纹识别工具，来判定信息对应的人物

- 事件只和 face 或 voice 相关联，而不与对应的 character 关联

- 也可以进一步拓展到其他元素上，如位置或者对象

此外，Agent 还能进行跨模态推理，建立节点间的等价关系，e.g. <face_3> = <voice_0>；这些等价关系会保存在一个 <character_2> 下

这就是用图存储的好处，可以直接用类似并查集的方法维护等价关系，而无需依赖于不可靠的LLM

记忆图中，face/voice与memory node连边；memory之间不连边

冲突信息可以采用投票机制解决，正确信息对应的边权会随着逐渐访问而变大，避免了偶然错误对系统稳定性的影响，流程如下：

![image.png](image.webp)

- 简要解释：给定一个视频以后，考虑某张脸和某个声音同时出现在画面里的时间作为权重，只保留声音对应时间最长的脸

    - 预先筛掉了同时有多张脸和多个声音的数据，以及同一张脸对应多个声音的数据，以保证稳定性

### 控制的实现

自主执行多轮迭代推理，流程如下（RL优化）：

![image.png](image%201.webp)

### 训练

训练两个 agent：记忆功能基于 Qwen2.5-Omni-7b，控制功能基于 Qwen3-32b

记忆功能训练采用 SFT，基于大模型合成的数据集 memory-7b-sft 进行模仿学习

控制功能训练采用 RL（DAPO），以是否回答正确作为奖励（0/1）

- 此外只允许它在 memory-7b-sft 构建的记忆中进行查找（有点类似于 teacher forcing）

    1. 提高训练稳定性，防止小的错误在中间累积，导致后面拿不到奖励

    2. 这样就可以做到解耦控制和记忆的训练，更加灵活

思考：

- 为什么记忆模块训练用 SFT 而不是 RL？→ 因为记忆模块的训练与 QA 解耦，无法评判写入记忆的有效性。

- 为什么不联合训练控制和记忆模块？→ 目标和基座模型不同，没办法联合训练；记忆模块需要多模态理解能力，需要 MLLM；而控制模块需要强大的推理能力，需要 reasoning LLM

## 评测

基线方法：

- Socratic Models：一个视频记忆&理解框架，思路是把视频切片后生成描述，需要时通过 RAG 的方法抽取对应记忆

- 在线的视频理解方法：MovieChat、MA-LMM、Flash-VStream

- 基于 agent 的方法：

    - Gemini-Agent：分别对两个 Gemini-1.5-Pro 做 prompt 用于记忆和控制进程（相当于没有强化学习）

    - Gemini-GPT4o-Hybrid：GPT-4o 负责控制，Gemini-1.5-Pro 负责记忆

数据集：

- M3-Bench-robot, M3-Bench-web

- VideoMME-Long：另一个长视频QA benchmark


可改进的地方：

1. 过度依赖语义上的相关性，而忽略了逻辑相关性和时间上的连续性

    2. 过度依赖：在retrieval的过程中，只考虑余弦相似度top-k的memory

        3. 语义相关性高也不代表内容相似，比如"i do love cats"和"i do not love cats"之间的相关性也很高，但表达的意思完全相反

    4. 逻辑相关性：比如因果关系，A导致B，但A可能与B相关性不高

    5. 时间连续性：我在吃饭之后第一件事做了什么？

    6. 分析原因：事件之间不存在连边；可以考虑在事件之间添加边，比如逻辑边表示因果先后，或者时间边表示时间发生的顺序

思考：

1. 整体的设计有点像知识图谱（不过暂时没想到怎么用，而且他们文中没引用）

2. 为什么用图而不是其他结构？

    3. 最直接的：便于读者理解，而且可解释强

    4. 和自然语言相比，更加符合数学本质

    5. 最重要的：通过图表示，可以把高熵的、需要LLM辅助理解的自然语言，转化为便于计算机算法处理的结构化信息

        - 比如，可以采用并查集来合并不同face和voice，同一个集合构成一个character；访问两个人之间的关系也可以

1. 为什么把face/voice作为构成事件的实体，而不是character？后者其实更加合理，而且便于查询

    2. 便于在线快速存储信息？

    3. 保证entity的一致性？防止某个识别错误的face或voice污染整个character的信息？→ 正解

4. 为什么先查找事件而不是先查找人？entity的相关性和语义相关性本质不同；也不可能遍历所有人，因为长期记忆下entity会非常多

5. 记忆会随时间衰减，也可能失效，需要处理这一点。因为类似的记忆可能出现多次，需要及时删除无用的记忆才能保证语义检索的正确性（不知道实际有没有做？→ 没有做）

6. RL是为了优化什么？→ control agent