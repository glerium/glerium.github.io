---
title: GPT-2 复现笔记
date: 2026/01/24 21:30
tags:
    - LLM
categories:
    - 学习笔记
---
本文是基于 karpathy/build-nanogpt 项目复现 GPT-2 过程中记录下的笔记。

## 正文

- GPT-2 中的位置编码没有沿用 transformer 论文中的正余弦编码，而是将其视作一种可学习参数进行训练

<!--more-->

- GELU 激活函数：$\text{GELU}(x) = x*\Phi(x)$

    - 其中 $\Phi(x)$ 是高斯分布的累积分布函数

    - GELU 还有一种基于 tanh 的近似形式。GPT-2 采用的就是这种形式，因为当时 tensorflow 在计算 $\Phi(x)$ 时非常慢；不过现在已经没有了采用 tanh 的必要

    - ReLU 在值小于零时，不会产生任何梯度，从而导致神经元死亡。GELU 可以避免这一问题，实践中效果也更好。

- 在原始 transformer 论文和 GPT-2 中，输入层的 token embedding 矩阵和输出层的 lm_head 互相共享权重参数。

    - 直观理解：如果两个 token 相似，那么他们应当拥有相似的 token embedding，以及相似的 output logits

    - 输入层是模型理解自然语言的过程，输出层则是把自己的理解映射回自然语言；直观来说，这两个过程应该是类似的过程；如果独立训练，模型还需要主动对这两个过程进行对齐，进而增大训练难度

    - 好处：模型性能提升，还节省了参数量（~30% in 124M）

- GPT-2 在初始化残差层参数的时候，会把权重乘以 $1/\sqrt{N}$ （N是残差层数），这样可以防止每层的方差逐渐累加，造成训练不稳定。

    - 具体做法：在残差层的每个组件中，将最后一个线性层初始化的方差乘以 $1/\sqrt{N}$

- Xavier 初始化会将标准差设为 $1/\sqrt{K}$（K是进入该层的特征数量）

- bf16 相比 float16 而言，牺牲了尾数精度来换取和 float32 相同的表示范围（指数位长度）

![image.png](image.webp)

- 在A100上，tf32相比float32可以得到~8x的加速

- 模型训练加速策略：

    - `torch.set_float32_matmul_precision('high')`：采用tf32在tensor core中进行计算

    - `torch.autocast`：混合精度训练，将一部分计算转移到bfloat16上

    - `torch.compile`：预先对模型的代码进行编译处理，摆脱Python解释器只能顺序执行的局限性

    - FlashAttention：采用算子融合的方式，重写attention机制，减少数据传输

        - 虽然tflops更多但运行速度显著更快

        - 不会实际生成巨大的 attn 矩阵，而是做流式 softmax

    - vocab_size: 50257 → 50304（变成规整的数字，更符合 cuda kernel 的设计）

- clip global grad norm: 将模型参数的梯度进行缩放，使得其L2范数等于不超过1

    - 限制梯度幅度的上限，防止脏数据产生过大梯度，干扰模型训练

- LLM 在应用 weight_decay 的时候，一般只应用于线性层的权重参数，排除偏置项和 layer_norm 的 weight & bias

- 显存不足时，可以采用梯度累积（gradient accumulation），通过串行计算来模拟大显存

    - `loss.backward()` 总是会累计梯度，而不是覆盖梯度

    - 记得对累加的梯度按照累加步数取平均值，以遵循loss的计算公式

- 使用DDP训练模型时，需要注意将种子广播到所有卡上，防止不同卡产生不同初始化参数的模型

- DDP的原理：正向传播与单卡没有区别；反向传播梯度后，将梯度AllReduce取平均值，然后同步到所有卡上

- 使用同样的种子对同一个batch进行优化，发现单卡和多卡的loss有一点差异，原因是多卡会略微改变数据之间的reduce顺序，微小的浮点误差被深层神经网络所放大，从而造成loss差异

- 数据来源：

    - GPT-2：递归抓取 Reddit 上点赞≥3帖子的外链

    - GPT-3：混合数据集，大部分来自 CommonCrawl （~60%）

        - CommonCrawl 随机抓取互联网上的网页，原本数据质量较差，需要进行人工清洗

    - 其他优秀数据集：

        - Red Pajama

        - FineWeb

- 评估方法：

    - 训练集 / 测试集 loss

    - HellaSwag：多选题的形式，采用语言模型生成对抗选项，被GPT-3采用

        - 具体方法：向LLM输入问题和选项，要求它以自然语言形式输出答案；计算每个选项各token在输出中的概率乘积，取argmax作为模型选择的选项

    - Luther 评估

- 结果：

    - 仅用 10B token 的训练，就超过了 100B 的 GPT-2-124M 在 HellaSwag 上的表现

        - GPT-2 的训练语料可能涵盖更加广泛的范围，如多语种内容，或 HellaSwag 没有涉及的内容

        - 数据集泄露：HellaSwag 出现时间早于 fineweb 训练集，其中的一部分可能已经融入到训练集中

        - 数据集本身质量有所提升

## 参考资料

* karpathy/build-nanogpt https://github.com/karpathy/build-nanogpt
* Let's reproduce GPT-2 (124M) https://www.youtube.com/watch?v=l8pRSuU81PU
* Andrej Karpathy 中英从零构建 GPT（重制版） https://www.bilibili.com/video/BV1mqrTBvEaf