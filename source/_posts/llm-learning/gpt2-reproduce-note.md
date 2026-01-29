---
title: GPT-2 复现笔记
date: 2026/01/24 21:30
tags:
    - LLM
categories:
    - 学习笔记
---
本文是基于 karpathy/build-nanogpt 项目复现 GPT-2 过程中记录下的笔记。

## 知识点

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

- 交叉熵和熵的关系

    - $H(p,q)=H(p)+\text{KL}(p||q)$

    - 在 one-hot 下 H(p)=0 是一个常量，norm 实际上就是 KL 散度的梯度

    - 在 softmax + cross-entropy 的条件下，对 logits 施加的梯度是 p - y

        - 其中 p 是预测概率，y 是 0-1 label

## 数据来源

- GPT-2：递归抓取 Reddit 上点赞≥3帖子的外链

- GPT-3：混合数据集，大部分来自 CommonCrawl （~60%）

    - CommonCrawl 随机抓取互联网上的网页，原本数据质量较差，需要进行人工清洗

- 其他优秀数据集：

    - Red Pajama

    - FineWeb

## 评估方法

- 训练集 / 测试集 loss

- HellaSwag：多选题的形式，采用语言模型生成对抗选项，被GPT-3采用

    - 具体方法：向LLM输入问题和选项，要求它以自然语言形式输出答案；计算每个选项各token在输出中的概率乘积，取argmax作为模型选择的选项

        - 实际上就是交叉熵损失

- Luther 评估

## 结果

### HellaSwag 评估

仅用 10B token 的训练，就达到了接近 100B 训练量的 GPT-2-124M 在 HellaSwag 上的表现

分析原因：

- GPT-2 的训练语料可能涵盖更加广泛的范围，如多语种内容，或 HellaSwag 没有涉及的内容

- 数据集泄露：HellaSwag 出现时间早于 fineweb 训练集，其中的一部分可能已经融入到训练集中

- 数据集本身质量有所提升

![accuracy.png](accuracy.webp)

### grad_norm 曲线

- 前100个step异常大，之后呈现先增后减的趋势；具体来说，700 step之前增长，之后逐渐下降，且下降速率逐渐减小

- 前100个 step 很大：模型随机初始化，预测呈现出均匀分布，几乎无法预测正确的 token，整体交叉熵非常大

- 100 step 以后，grad norm 的趋势受两部分影响

    - 第一部分是模型参数本身的 L2 范数，这一部分本身有增长的倾向，但会逐渐被正则化所压制，呈现出先增后降的趋势（Norm图橙色曲线）

    - 另一部分是模型预测交叉熵所产生的梯度，这部分随着 step 而逐渐下降，因为模型对正确选项的预测越来越自信（Loss图）

    - 在700 step之前，第一部分的增长趋势压过第二部分的下降趋势，所以 grad norm 逐渐增长；700 step之后，模型范数被正则化压制，两部分都呈现出下降趋势，所以 grad norm 下降（Norm图紫色曲线）

- 此外，在训练过程中还出现了几次异常高的 grad norm，猜测可能和数据有关（Norm图紫色曲线，待验证）

![loss.png](loss.webp)

### Loss 曲线

- 整体趋势：val loss 稳定下降，而 train loss 呈现波动下降趋势

    - 分析：

        - val loss 下降稳定，说明模型泛化性强

        - val loss 比 train loss 下降更加稳定，因为 val loss 数据集一直是固定的；而 train loss 每次都是随机抽样的数据集，受数据本身影响

- 一个奇怪的地方：

    - 在 ~4500 step 之后，loss 的波动明显变小。观察 norm 曲线可以看出，在 ~4200 和~4400 step 分别出现了两次异常大的 grad norm，随后 loss 波动开始减小

    - 猜测可能是因为这两次的梯度波动，把模型参数推向了梯度更加平稳的区域

### 生成文本采样

- 参数设定：max_tokens=32, temperature=1, top-k=50

- step 250：随机生成token，没有可识别的语法结构和语义

    - *In computer science, artificial intelligence* (journal and arc (API planting – US) Chapter or naturally solated (340 4 score dysich ( array (2007 rib (leg) cloth ( Get

    - *In computer science, artificial intelligence* disposed in exports are in insert in depict if, and ruined. were siege in cartoon control of South architecture to fired in heard for nursing decadeidge (anduses

    - *In computer science, artificial intelligence*<|endoftext|> end School
    \- > guarantee here, the components of energy of blockade Essies going to answer. These historic Organization has been say you have been more complex

    - *In computer science, artificial intelligence* with the non conscience are adapted from hot system. The idea been believed far burned the caruuming favorite disorder designing the UK, the abdomen. 2,

- step 1000：习得英语的基本语法结构；捕捉到了一些短语之间的语义相关性；但依然没有可以识别的语义，句子结构支离破碎

    - *In computer science, artificial intelligence*, and tragedy's idea, which has been used in philosophy with today's debt is official and not marked at the time, but that almost

        - *在计算机科学、人工智能和悲剧思想中，如今的债务在哲学中已被正式提出，但当时并未明确标记，但几乎*

    - *In computer science, artificial intelligence*, digital telecommunication policy development, and decision- Digital security concerns and the use of automateding technologies to find solutions to more effectively being attractive to optical-based
        - *在计算机科学、人工智能、数字电信政策制定和决策领域，数字安全问题以及利用自动化技术寻找更有效的解决方案，对基于光纤的通信方式更具吸引力。*

    - *In computer science, artificial intelligence*, neural exercise and architecture. In up, quantum mechanics lets a complete perform during a read-inversion of the outer new right CPU.
Tony has an

        - *在计算机科学、人工智能、神经网络训练和架构领域，量子力学使得在外部新右侧 CPU 的读取反转期间能够完成整个过程。
托尼有一个*

    - *In computer science, artificial intelligence* and brain philosophy.
On then, however much after, modern technology advancement in forensic science, and more.
Demherence can be manufactured through back to digital
        - *在计算机科学、人工智能和脑哲学领域。
然而，此后很长一段时间，法医学等领域的现代技术进步，以及其他更多领域。
失语症可以通过回归数字化来制造。*

- step 3000：句子结构明显成熟，拥有完整的主谓宾；句子长度变长，而且前后句有了一些语义连贯性，说明模型开始拥有上下文能力

    - *In computer science, artificial intelligence* can provide improved utility and speed. The Austrian demagogue sub-rinsing algorithm used an artificial intelligence (AI) program, which inhibits AI data from being
        - *在计算机科学领域，人工智能可以提高效用和速度。奥地利煽动家的次级清洗算法使用了人工智能（AI）程序，这阻碍了人工智能数据的传播。*

    - *In computer science, artificial intelligence* would converge. From the northern shore to the ocean, miniature waves of intelligent pilot jerky motionless movements.
They had hit, more and more limitless launching
        - *在计算机科学领域，人工智能将会汇聚。从北岸到海洋，无数智能飞行员如同微型波浪般涌动，做出断断续续的静止动作。
它们已经击中目标，越来越多、无限地发射出去。*

    - *In computer science, artificial intelligence*, and university performance robotics is a mandatory component of New Disability and SAGE NG curriculum, build- increases in graduation.
In addition to that speaker'
        - *在计算机科学、人工智能和大学机器人技术领域，机器人技术是新残疾教育和SAGE NG课程的必修组成部分，有助于提高毕业率。
此外，演讲者*

    - *In computer science, artificial intelligence* tells us how to use a machine learning environment that could offer a great score—that would make a fortune king's fortune. Bias: Big Data
        - *在计算机科学领域，人工智能告诉我们如何使用机器学习环境来获得极高的分数——这足以让富翁发大财。偏见：大数据*

- step 8000：模型开始使用较为复杂的句子结构；虽然幻觉依然严重，但此时句子拥有了较为完整的语义；上下文的衔接进一步成熟

    - *In computer science, artificial intelligence* and AI (AI) is as much a novelty as its innovations have been during untold human evolution–" inexplicably, so the very advancement that created this
        - *在计算机科学领域，人工智能（AI）和人工智能（AI）就像人类漫长进化过程中不断涌现的创新一样，都是一项全新的技术——难以解释的是，正是这项技术的发展创造了这一切。*

    - *In computer science, artificial intelligence* is one of the main weapons in detection, management and improved intelligence. Modern artificial intelligence platforms allow programmers to manipulate such an artificial state results in competitions inside science and
        - *在计算机科学领域，人工智能是检测、管理和提升智能的主要手段之一。现代人工智能平台允许程序员操控这种人工状态，从而在科学界引发竞争。*

    - *In computer science, artificial intelligence* and machine learning approaches, in the making of artificial intelligence or how to web query — which is part of Artificial Intelligence.
An artificial intelligence-based AI based
        - *在计算机科学、人工智能和机器学习方法中，人工智能的构建或网络查询（这是人工智能的一部分）都属于人工智能的范畴。
基于人工智能的AI*

    - *In computer science, artificial intelligence* and artificial intelligence, or kinetic and physical systems, aren't that bad. People who don't know how to use machines shouldn;t
        - *在计算机科学领域，人工智能，或者说动能和物理系统，并没有那么糟糕。不懂如何使用机器的人不应该*

- step 13000：开始使用连词衔接上下文，说明模型具备了一定的逻辑能力；语义已经较为成熟自然

    - *In computer science, artificial intelligence* is a branch of computer science that deals with the composition of the information output, the attention of today's machine-aided design algorithms and the proper
        - *在计算机科学中，人工智能是计算机科学的一个分支，它研究信息输出的构成、当今机器辅助设计算法的关注点以及适当的*

    - *In computer science, artificial intelligence* is what pretty much is when it's done by humans but it's really becoming more sophisticated and its even more applicable now that you can just go in and put
        - *在计算机科学中，人工智能基本上就是人类所做的事情，但它正变得越来越复杂，而且现在应用范围更广，因为你可以直接输入代码进行操作。*

    - *In computer science, artificial intelligence* systems have evolved from the ancient history era to the present day. CT technology, or "soft machines" it has been going on for 3,000
        - *在计算机科学领域，人工智能系统从古代发展至今。计算机技术，或称“软机器”，已经发展了3000年。*

    - *In computer science, artificial intelligence*, and genetics, we use the term "software programs" as it is coined by Albert Einstein.
Educational technologies all lead to two types of student performance
        - *在计算机科学、人工智能和遗传学领域，我们使用“软件程序”一词，这个词是由阿尔伯特·爱因斯坦提出的。
教育技术最终都会带来两种类型的学生表现。*

- step 19000：虽然依然存在内容幻觉，但输出内容开始呈现出有深度的见解，而不仅仅描述浅层事实，说明模型开始具备理解抽象概念的能力

    - *In computer science, artificial intelligence* may be accelerating the rate at which humans become more successful.
Yet the ground was often uncharted. Most notably, the results of a survey on help from
        - *在计算机科学领域，人工智能或许正在加速人类走向成功。
然而，这片领域往往充满未知。最值得注意的是，一项关于人工智能帮助的调查结果显示*

    - *In computer science, artificial intelligence* (AI) is about a collaboration of many things. One key advantage of AI is that the local intelligence, itself, is interdisciplinary be it in the face
        - *在计算机科学中，人工智能（AI）涉及诸多方面的协作。人工智能的一个关键优势在于，其局部智能本身就是跨学科的，无论面对何种情况。*

    - *In computer science, artificial intelligence* is one of the cornerstones of computer science. Despite the growth of technologies, AI has faced numerous growth among the industries. It is no surprise that AI has
        - *在计算机科学领域，人工智能是计算机科学的基石之一。尽管技术飞速发展，人工智能在各行各业的发展却举步维艰。人工智能的出现也就不足为奇了。*

    - *In computer science, artificial intelligence*) is well-defined technologies that result from the engineering process and decision-making that was required in common test environments, particularly by the United States.2
        - *在计算机科学领域，人工智能（AI）是指在常见的测试环境中，特别是美国，通过工程流程和决策而产生的、定义明确的技术。2*

## 参考资料

- karpathy/build-nanogpt https://github.com/karpathy/build-nanogpt

- Let's reproduce GPT-2 (124M) https://www.youtube.com/watch?v=l8pRSuU81PU

- Andrej Karpathy 中英从零构建 GPT（重制版） https://www.bilibili.com/video/BV1mqrTBvEaf

