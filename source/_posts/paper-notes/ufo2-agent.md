---
title: "[论文笔记] UFO^2 The Desktop AgentOS"
date: 2026/01/19 18:07:14
tags:
    - Agent
    - 多模态
    - 多模态Agent
categories:
    - 论文笔记
---

- 任务：计算机操作Agent

- 动机：

    - OS集成浅：用不到API、进程状态等

    - 基于截图的交互脆弱：界面改版或遇到非标准界面时容易出错

    - 用户体验差：执行过程容易中断，和用户抢鼠标

<!--more-->

- 传统方法：基于脚本 → 灵活性不足，而且需要人工设计

- 传统CUA（computer-using agents）：可以理解模糊和复杂指令，侧重于视觉定位和语言推理，但忽视了和OS/App的system-level integration

    - 依赖视觉截图作为输入，输出模拟鼠标/键盘：存在噪声和冗余，增大LLM的认知负担

    - 很少利用系统无障碍接口、API和进程状态

- 方法：

    - HostAgent + 定制AppAgents，每个agent可以访问领域知识和特定api

        - AppAgent：输入截图、元数据（界面控件等）、截图的目标标注

    - 在虚拟桌面中独立操作，互不影响

    - 图形化 → 结构化

        ![image.png](image.webp)

        ![image.png](image%201.webp)

- 记忆结构：

    - 短期记忆：Shared Blackboard，各个AppAgent共享，存放key observations、intermediate results、execution metadata

        - 每次推理都会完整注入到AppAgent的提示词中

    - 长期记忆：

        - 文档记忆 (help document)：把用户手册等内容输入到vector store

        ```JSON
        {
          "request": "How to ...",
          "guidance": [
            "Click the ...",
            "Press ...",
            ...]
        }
        ```

        - 经验记忆 (self-experience)：把成功经验的轨迹总结成signature + step-by-step guidance存入vector store

        ```JSON
        {
          "subtask": "Create a doc for...",
          "plan": [
            "Open the ...",
            "Click on ...", 
            ...]
        }
        ```

        ![image.png](image%202.webp)

- 评测

  - Benchmark

    - Windows Agent Arena (WAA)：需要操作常用软件的154个自动化任务，每个任务配备有验证脚本

    - OSWorld-W：和WAA类似，涵盖办公软件、浏览器交互等任务

  - Baseline

    - UFO

    - NAVI：只采用屏幕截图输入

    - OmniAgent：纯视觉定位，采用GPT进行计划

    - Agent S：经验驱动的多智能体系统

    - Operator：OpenAI推出的基于屏幕截图的鼠标操作方法

  - 指标

    - 成功率

    - 操作次数

