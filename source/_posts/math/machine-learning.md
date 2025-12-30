---
title: 机器学习课程笔记
date: 2024/06/18 14:07:00
tags:
    - 机器学习
    - 数学
categories:
    - 课程笔记
---

本文是机器学习的课程学习笔记。

<!--more-->

- 例题集锦

    - 2-4

        ![image.webp](image.webp)
        

    - 3-7

        ![image.webp](image%201.webp)

    - 4-7

        ![image.webp](image%202.webp)

    - 4-13

        ![image.webp](image%203.webp)

        ![image.webp](image%204.webp)

    - 5-2

        ![image.webp](image%205.webp)

    - 6-6

        ![image.webp](image%206.webp)

    - 11-8

    ![image.webp](image%207.webp)

    ![image.webp](image%208.webp)

    ![image.webp](image%209.webp)

    ![image.webp](image%2010.webp)

    ![image.webp](image%2011.webp)


    ![image.webp](image%2012.webp)

    - 12-5

        - Gini指数算决策树，答案同上

    - 13-4

        ![image.webp](image%2013.webp)

        ![image.webp](image%2014.webp)

    - 13-15

        ![image.webp](image%2015.webp)

        ![image.webp](image%2016.webp)

    - 16-30（第一张是random policy，第二张是optimal policy）

        ![image.webp](image%2017.webp)

        ![image.webp](image%2018.webp)

    - 21-7

        ![image.webp](image%2019.webp)

- numpy函数整理

    - `np.arange(a, b, step)` 为[a,b)左开右闭区间

    - `np.random.normal(mu, sigma, shape)`：生成正态分布的随机数

    - `np.linalg.eig(C)`：求特征向量、特征值

    - `np.diag()`：对角矩阵

    - `np.linalg.inv(C)`：求逆矩阵

    - `np.linalg.pinv(C)`：矩阵的伪逆

    - `np.linalg.det(C)`：求行列式

    - `np.where(condition)`：返回condition为真的index list，元素长度等同于condition维度

        - `np.where(condition, x, y)`：当condition在对应位置为真时返回x，否则返回y

    - `np.kron(x, y)`：得到一个 $(a1*a2, b1*b2)$ 的tensor（二维情况）

- plt函数整理

    - `plt.scatter(x, y)`：绘制散点图

- 1_syllabus_and_basics

    - 独立事件的self information具有可加性

- 2_graph_and_matrixCalculus

    - 香农熵的定义式：$H(x)=-\sum p_i \log_2 p_i=-\mathrm{E}\left[\log_2 p(x)\right]$

        - 表示how much "choice" is involved in the selection of the event **or** how uncertain we are of outcome

        - if the prob of Bounilli distrib has p and 1-p, then the Shannon Entropy is maximized when p=0.5

    - 协方差矩阵：$\Sigma(X)$；若样本有m个，特征为n个，则协方差矩阵为$n\times n$的

        - $\Sigma(A+X)=\Sigma(X)$（A为常矩阵）

        - $\Sigma(BX)=B\Sigma(X)B^{T}$

            - $\Sigma(X)$为半正定的（semi-definite）

        - 若m组数据为均值为0的样本 $\mathbb{X}$，则 $\Sigma=\frac{1}{m-1}\mathbb{X}^T\mathbb{X}$

    - 向量对向量求导：分子布局（numerator layout，分子的shape在前）和分母布局（denominator layout，与分子布局相反）

        - 通常用分子布局

        ![image.webp](image%2020.webp)

- 3_numerical_computation

    - 上溢和下溢：

        - underflow：接近于0的数字ε被round为0

        - overflow：过大的数字导致溢出

        - softmax 防止上下溢的方法：

        ![image.webp](image%2021.webp)

    - 梯度下降的理论基础：对标量函数而言，$f(x-\varepsilon\cdot\mathrm{sign}(f^{'}(x)))<f(x)$

        - 导数为0时，可能为最大值/最小值/鞍点

            - 如何判断？对Hessian矩阵进行特征值分解

            ![image.webp](image%2022.webp)

- 4_machine_learning_basis

    - 常见的有监督/无监督学习方法：

        ![image.webp](image%2023.webp)

    - confusion matrix（混淆矩阵）：每个元素表示（target=i, output=j）的数据个数

    - 评价指标：

        - Accuracy：准确率

        - Sensitivity/Recall：TP/(TP+FN)（模型能筛选出多少正样本）

        - Precision：TP/(TP+FP)（输出的正样本中有多少是准确的）

        - F1-Score：Precision和Recall的调和平均数（2/F1=1/Recall+1/Precision）

        - ROC曲线：x为False Positive Rate (FP/(FP+TN))，y为True Positive Rate (TP/(TP+FN))

            - ROC曲线中的特殊点【曲线越接近(0,1)效果越好（结合AUC指标的意义来记）】

            ![image.webp](image%2024.webp)

            - AUC：ROC曲线下的面积，越接近1越好

        - MCC：数据集**不平衡**时采用，公式：

            $MCC=\dfrac{TP\times TN-FP\times FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}}$

    - k折交叉验证最终要取k次实验的**平均值**作为评价指标

    - bias-variance tradeoff：

        - bias偏差，通常情况下loss的优化目标；variance方差，similar inputs should give similar outputs

        - 不可能同时减小bias与variance，因此更加复杂的模型并不一定有更好的表现

- 5_perceptron

    - 矩阵求导tricks：

    ![image.webp](image%2025.webp)

    - 感知机

        - loss优化算法

            ![image.webp](image%2026.webp)

        - 总体时间复杂度为 O(Tmn) （T为循环轮数）

        - Rosenblatt：对于线性可分数据集而言，感知机可在有限迭代轮数内收敛

        - 衡量感知机模型的优度：均方误差，计算Σ(target - output)^2

- 6_MLP

    - 数学原理：略

    - 交叉熵损失函数：$Loss=-\sum_{k=1}^{n}t_k\log(y_k)$

        - $t_k$可能为0，因此放在log外面

    - 防止陷入鞍点的方法

        - 多跑几次，用不同的初始参数得到结果

        - 动量法（Momentum）：在更新权重时添加上次迭代中权重的变化量，类似于惯性

            ![image.webp](image%2027.webp)

    - Universal Approximation Th.：任何能被MLP解决的问题都能被单隐藏层的MLP解决（实践中最多使用两层MLP）

    - Early Stopping：当测试集loss增长/下降趋缓时停止训练

- 7_LDA_PCA

    - LDA（线性判别分析）：将数据集投射到某个超平面上，使得投射后的数据线性可分

        - LDA的思想：假设输入数据的两种类别分别服从多元正态分布$(\mu_0,\Sigma_0)$和$(\mu_1,\Sigma_1)$；将输入变量进行进行**线性变换** $y=\mathbf{w}x$；在此基础上最大化类间方差$\sigma^2_B$，最小化类内方差$\sigma_W^2$，即最大化 $S=\frac{\sigma_B^2}{\sigma_W^2}$.

        - 结论：$\mathbf{w} \propto (\Sigma_1 + \Sigma_0)^{-1} (\mu_1 - \mu_0)$

    ---

    - PCA（主成分分析）：将数据集投射在若干条正交直线上，使得投影在每条直线上的数据方差最大。这些正交直线即为主成分。

        - 数据预处理：均值为0，方差为1

        - 推导过程：

            ![image.webp](image%2028.webp)

        - 于是方差最大问题被转化为二次型的最大化问题，对$\Sigma$进行特征值分解即可，前k大的特征值即为最大方差之和，其对应的特征向量即为k个主成分；$\Sigma$为正交矩阵，因此可以保证k个主成分之间正交

- 8_ICA_KNN

    - ICA（独立成分分析）：m条数据，每条数据x都是由原始数据s经线性变换A作用得到的，需求出unmixing matrix $W=A^{-1}$

        - 鸡尾酒聚会问题

        - Method：假设Sigmoid作为默认的原始数据的累积分布函数，然后利用概率密度变换公式，求出使得似然概率最大的unmixing matrix

        - Remark

            - 如果有关于原始数据的先验知识，可以采用其他概率分布函数

            - 在样本之间可能存在相关性的情况下，如果**训练数据足够多**或**提前将数据打乱**，ICA的性能不会受到太大影响

    - KNN（k近邻算法）

        - 若k较小则对噪声敏感；k取过大则会影响准确率

        - KNN可以用于分类（输出投票类别）或回归（输出k近邻标签的均值）

        - 常规KNN的复杂度较高，可以采用KD树优化查询复杂度

            - 构造方法：选取方差最大的维度，以其中位数为界限把剩余数据划分为两份

                ![image.webp](image%2029.webp)

            - KD树的期望时间复杂度为 $O(\log n)$，极端情况下可退化为 $O(n)$

- 9_linear_method

    - 线性回归：均方误差作为损失函数

    - 避免过拟合的正则化方法：

        - 岭回归（Ridge Regression）：添加L2范数惩罚项

        - Lasso Regression：添加L1惩罚项

        - 弹性网回归（Elastic Net）：同时添加L1惩罚与L2惩罚，两者加权到Loss函数中

            $$J(\theta)=\frac{1}{2m}\sum_{i=1}^n(t_i-h_\theta(x_i))^2+\alpha\cdot(\frac{1-\lambda}{2}\sum_{i=0}^n\theta_i^2+\lambda\cdot\sum_{i=0}^n\left|\theta_i\right|)$$

    - Logistic Regression：将分类概率视为 $y_i=\frac{e^{h_\theta(x_i)}}{1+e^{h_\theta(x_i)}}$（$h_\theta(x_i)$为模型输出）

    - Softmax函数

    - GLM（广义线性模型 Generalized Linear Models）：给定输入 x 后，目标变量 t 服从某个指数族分布（例如正态分布、泊松分布等）。假设这个分布的自然参数与 x 线性相关。于是可以用GLM对这个参数与x的变换矩阵进行回归分析。

        - GLM三大前提：

            1. $t|x;\theta\sim EF(\eta)$

            2. 对于给定的x，模型需要预测 $h(x)=\mathbb{E}\left[t|x\right]$

            3. 自然参数 $\eta$ 与 x 线性相关：$\eta=\theta^T x$

        - 若EF为高斯分布，可以假定 $\sigma^2=1$

        - 在线性回归模型中，**最大化对数似然函数等价于最小化均方误差**。因此传统的回归分析通常采用最小二乘法估计模型参数。

- 10_EM

    ![image.webp](image%2030.webp)

    - 步骤：

        - E-step：计算

        $$Q(\theta|\theta^t)=E_{z|X,\theta^t}[\log L(\theta;x,z)]$$

        - M-step：更新

        $$\theta^{t+1}=\argmax_\theta Q(\theta|\theta^t)$$

        - 其中z是离散的，而θ是连续的，且对每个数据点而言都存在一个θ与之对应

    - 手推抛硬币问题/两个正态分布的问题

- 11_DecisionTree

    - top-down, recursive, and greedy algorithm

    - 普通的误分类率在某些情况下不能区分决策树的好坏，因此采用熵进行衡量

        - 定义信息增益如下：

        ![image.webp](image%2031.webp)

        - 构建决策树的过程中，贪心地选择信息增益最大的特征（手推决策树）

        ![image.webp](image%2032.webp)

- 12_GiniIndex_Pruning

    - Gini Index（基尼指数）：$G=\sum_{i=1}^n\sum_{j=1,~j\neq i}^{n}p_i p_j=1-\sum_{i=1}^np_i^2$（其中 $p_i$ 为每种类别的数据所占比例）

        - 衡量一个randomly chosen data被错误分类的概率

        - **$G=0$ 表示所有数据属于同一分类**；$G=1/n$ 表示 $n$ 种类别平均分布

    - 奥卡姆剃刀原理：在多个假设可以解释同一现象的情况下，应该选择假设最少的那一个解释。

    - TDIDT（top-down inductive decision tree）

    - 剪枝（pruning）：为了缓解过拟合，大致分为两种：

        - pre-pruning/forward pruning：在生成树的时候提前终止生成过程（由阈值决定）

        - pst-pruning/backward pruning：在生成完成以后删去某些边

- 13_SVM

    - SVM可用于分类或回归

    - 假设样本线性可分，找到一个分离超平面 $\omega$，使得样本在两个类之间的间隔尽可能大

    - 支持向量：位于 $t_i\cdot(\omega x_i^T+b)=1$ 平面上的样本点

    - Method：在保证 $t_i\cdot(\omega x_i^T+b) \ge 1$ 的前提下，最小化 $f(\omega)=\frac12||\omega||=\frac 12\omega^T \omega$，具体可以采用Lagrange方法进行优化

    - 这种方法是硬间隔（Hard Margin），还有软间隔方法：最小化的式子变为

    $$\left[ \frac 1n\sum_{i=1}^n \max(0,~1-t_i(\omega x_i^T+b)) \right]+\sigma\omega^T\omega$$

    这种方法允许一定样本在两类支持向量之间的带状区域内（由σ调整权重）

    - 对于非线性可分的样本而言，可以采用Kernel trick将问题化为线性可解的问题

        - 具体地说，找到一个核函数，使得将样本变换到核空间（Kernel space）内（这个空间的维度通常比原始空间高得多）

        - 计算成本大大提高，但能将线性不可分的数据变为线性可分

- 14_Soft_SVM

    - 允许一部分点在错误的一侧，并对这部分点进行惩罚 $\xi_i$ 满足 $t_i\cdot(\omega x_i+b)=1-\xi_i$

    - 优化 $\min f(\omega,b,\xi)=\frac 12 \omega\omega^T+C\sum_{i=1}^n\xi_i$

        - 限制条件 $1-\xi_j-t_j(\omega x_j^T+b)\le 0$，$\xi_j\ge 0$

    ![image.webp](image%2033.webp)

    - C是误分类惩罚项的权重

    - **当C较小时，对噪声不敏感，但容易欠拟合**

    - **当C较大时，对噪声很敏感，但容易过拟合**

    - $C \to \infty$时，为hard margin SVM

- 15_ensemble_and_GA

    - Ensemble method

        - 同时使用多个learner进行学习，每个了learner在数据上获得略有不同的结果，再将这些结果结合起来，效果会优于一个learner

        - 评价的两个指标：偏差（bias）与方差（variance）

    - Boosting：采用多个弱分类器的组合

        - 为正确的样本赋予较小的权重，而错误样本赋予较大权重，这一点通过为learner赋权实现（存疑，ppt中没有原文表述）

        - 每次选择一个最优（Loss）最小的模型加入到复合分类器中，并给新添加的模型赋予一个权重

        - Remark

            - 思想类似于微积分中用阶跃函数的组合逼近复杂函数

            ![image.webp](image%2034.webp)

            - 不断增大效果更优的模型权重，否则降低权重

        - 目的：减小bias

    - Bagging（Bootstrap Aggregating）

        - 步骤：

            1. Bootstrap采样：从原数据集中随机有放回采样，$n_{new} \le n_{old}$

            2. 训练模型：使用以上每组数据集分别训练模型

            3. 分类：所有模型同时预测，投票决定

        - 目的：减小variance

    - Stacking：将数据分为AB两部分；用A数据训练模型，然后将模型对B进行预测得到 $output_B$，然后将 $(data_B,output_B)$ 送入模型继续训练

        - 常与k-fold交叉验证结合使用

        - 作用：减小bias

    - 随机森林（Random forest）：采用多个决策树分别预测，对输出进行投票

        - 首先使用Bagging方法抽取**和训练集相同个数**的样本

        - 对于M个特征而言，每次延伸树枝时，随机选择其中大小为 $m=\sqrt{M}$ 的特征子集，从这个子集中选取最优特征进行延伸

        - 每棵树都尽可能最大程度生长，**不需要剪枝**

        - 优势：准确高效、可用于大数据集、**能够处理缺失值**、可以理解特征之间的交互关系、**可用于回归问题**

    - 遗传算法（Genetic Algorithm）

        - 基本步骤：

            - **问题表示方法**：需要一种方法将问题表示为染色体（chromosomes），通常使用某种字母表的字母组成一个字符串

            - **计算适应度的方法**：需要一种方法来计算解决方案的适应度（fitness），以评估每个解决方案的好坏

            - **选择父母的方法**：需要一种选择父母的选择方法，通常根据适应度来选择

            - **生成子代的方法**：需要一种通过繁殖父母来生成子代（offspring）的方法

        - P与NP：

            - P：可在多项式复杂度内解决的问题

            - NP：可在多项式复杂度内验证的问题

            - NP完全问题：若所有NP完全问题都能被多项式复杂度解决，则所有NP问题都可以在多项式内解决

            - P=NP？：二者是否为同一集合？

        - 背包问题的例子

            - 适应度设计：总价值减去两倍的超出部分

            - 父母选择（将合适的父母放入交配池）：

                - **锦标赛选择（Tournament Selection）**：重复从种群池中有放回地挑选四个字符串，并将最适合的两个放入交配池。

                - **截断选择（Truncation Selection）**：选择一些最好的字符串，将其余的忽略。例如，取前50%的最优字符串放入交配池，每个字符串出现两次以使交配池的大小合适。然后将交配池随机打乱以形成配对。

                    - 易于实现，倾向于开发利用现有优秀解的潜力，但限制了探索新的解的能力。

            - 适应度选择：在交配池中选择适应度≥1的个体进行一定次数复制，添加到新的总体中

            - 后代生成：

                - 交叉：父母基因相混合

                - 变异：基因以1/L的概率变异（L为string长度）

            - 下一代种群：

                - 替换父母

                - 精英选择：将下一代的最优字符串放入种群池，删除不适合的个体

                - 锦标赛选择

            - 防止过度收敛：

                - **小生境技术（Niching）**：将种群分为多个子种群，独立进化一段时间，然后偶尔将一个子种群的几个成员作为“移民”注入另一个子种群。

                - **适应度共享（Fitness Sharing）**：将某个字符串的适应度平均分配到该字符串在种群中出现的次数上。这种方法偏向于不常见的字符串，但会对非常常见的好解进行选择。

- 16_ReinforcementLearning

    - tradeoff between exploration & exploitation

    - 手算Gridworld

    - value function可以是静态的/随时间改变

    - $Q_t(a)$ 的迭代策略：$Q_{new}=Q_{old}+step\cdot [Q_{target}-Q_{old}]$

        - 这里的step可以为常量或1/n

        - step满足下两式时一定可以收敛

        ![image.webp](image%2035.webp)

    - selection methods：

        - greedy直接选argmax

        - eps-greedy：大概率选择argmax，小概率随机选择

        - UCB（upper-confidence-method）：结合上两者优点

        ![image.webp](image%2036.webp)

    - Bellman方程：

        $$v_\pi(s)=\sum_a\pi(a|s)\sum_{s',~r}p(s',r|s,a)\left[r+\gamma v_\pi(s')\right]$$

- 18_DP_and_MC & 19_TD

    - 几种RL方法比较：

        - DP：动态规划，给定环境模型，根据其他状态更新估计值 $v_\pi(s)$

        - MC：蒙特卡罗模拟，无需环境模型，不根据其他状态更新 $q_\pi(s,a)$

        - TD：时间差分，无需环境模型，根据当前状态和其他状态更新 $q_\pi(s,a)$

        ---

        - **DP方法**使用预期更新，即基于所有可能结果的平均或期望来更新。

        - **MC和TD方法**使用样本更新，即基于实际观察到的单个样本或一步观察结果来更新。

    - GPI（Generalized Policy Iteration）

        - 异步动态规划：策略迭代和价值迭代可以以任意顺序交替进行

        - 收敛性：如果评估和改进的过程都稳定下来，那么根据贝尔曼方程，价值函数和策略应该是最优的

    - MC

        - episode必须能在有限步数内结束

        - 在一个episode结束时才会评估价值并更新策略

        - first-visit method：优化第一次遇到状态s时的收益期望

            - every-visit method：优化每次遇到s的期望均值

        - 根据大数定律，预测值可以收敛到实际值（$n\to\infty$）

        - 例子：blackjack（一种棋牌游戏）

        - 对每个状态的预测都是独立的（非bootstrap方法）

    - TD

        - **TD(0)** 方法在这种设置下可以提供更准确的预测，因为它考虑了状态间的依赖关系和转移概率

            **初始化**：

            - 初始化所有状态的价值函数（通常可以设为零或随机值）。

            **与环境交互**：

            - 从起始状态开始，不断地选择动作并与环境交互，观察奖励和下一状态。

            **更新价值函数**：

            - 对于每个时间步 $t$，从状态 $s_t$ 采取动作 $a_t$ 转移到状态 $s_{t+1}$，并获得即时奖励 $r_{t+1}$。

            - 更新状态 $s_t$ 的价值函数： $V(s_t)←V(s_t)+α[r_{t+1}+γV(s_{t+1})−V(s_t)]$ 

                其中：

                - α 是学习率（取值范围为0到1）。

                - γ 是折扣因子（取值范围为0到1），用于折扣未来奖励。

            **迭代更新**：

            - 重复步骤2和步骤3，直到价值函数收敛或达到预定的迭代次数。

        - **批处理MC** 方法虽然在训练数据上最小化均方误差，但它可能没有考虑状态之间的依赖性，从而可能在新数据上表现不佳

        - 作为一种在策略方法，SARSA直接基于当前策略（即利用当前学到的 *Q* 值进行动作选择）来更新 *Q* 值

        - 这与“离策略”（off-policy）方法如Q-learning的区别在于，Q-learning会根据从当前策略派生的最佳可能动作来更新 *Q* 值，而不一定遵循当前策略

- 23_transformer

    ![image.webp](image%2037.webp)

    ![image.webp](image%2038.webp)

    ![image.webp](image%2039.webp)



