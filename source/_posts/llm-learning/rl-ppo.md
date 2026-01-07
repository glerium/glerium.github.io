---
title: RL & PPO
date: 2026/01/07 13:18:21
tags:
  - 强化学习
  - PPO
  - LLM
categories:
  - 学习笔记
---

本文是强化学习基础与PPO算法的学习笔记。

首先介绍是强化学习的基本概念（如Agent、State、Action、Reward等，具体见附录）。

RL的基本目标是为了让总体 return 最大，根据对期望回报的数学推导，我们可以导出 policy gredient 的式子： 

$$\frac1N \sum_{n=1}^n \sum_{t}^{T_n}R(\tau^{(n)})\log P_\theta(a_n^{(t)}|s_n^{(t)})$$

<!--more-->

但这个公式存在两个问题：首先，一步 action 的权重同时作用到整次采样的总 reward 上，这是不合适的，因为实际上它只会作用于离它最近的几次 reward 上；为了解决这个问题，需要引入一个指数衰减因子，并把采样步数限制在临近的几步内。第二个问题是，有时 reward 不仅仅和操作有关，而是和状态强相关；比如在优势状态下，进行任何的 action 都会获得正的收益，这会导致网络误以为需要增加所有操作的权重，导致训练不稳定，因此还需要引入一个 state-value function，来对当前状态的期望回报进行估计。此外，由于每次采样得到的 reward 有很强的随机性，所以还引入了 action-value function，用于对给定状态和动作下的期望 reward 进行估计。结合上述几点，最终我们需要把原公式中的 $R(\tau^{(n)})$ 替换成带衰减因子的 advantage function，即 $A_\theta(s,a)=Q_\theta(s,a)-V_\theta(s)$。

这种做法有个缺点：我们需要同时训练两个网络，一个动作估计网络 Q，还有一个状态估计网络 V，这样才能估计出最终的优势函数 A。但通过数学推导可以发现，可以只通过状态估计函数来计算出优势函数，这样就只需要训练一个网络了。

不过在这种情况下，我们依然需要人为设定采样步数这个超参数。为了避免这个问题，有学者提出了 Generalized Advantage Estimation，也就是 GAE。它通过数学推导，可以把从每步向后采样若干步的方法，转化为一次直接采样无限步，然后整体对优势函数进行估计。GAE 的最终公式为 

$$A_\theta^{\text{GAE}}(s_t,a)=\sum_{b=0}^{\infty}(\tau\lambda)^b \delta_{t+b}^V$$

这种做法不仅让整体式子变得简洁，无需确定采样步数；还避免了从每步都向前采样的做法，减少额外计算的同时，也增强了训练的稳定性。

至此，强化学习训练的准确性和稳定性已经可以得到保证，但还存在一个问题：传统 on-policy 训练的策略需要我们不断从当前的 policy network 中进行采样，而且每次采样只能用于一次训练，接着马上就被丢弃，这导致大部分时间都被浪费在采集数据上，效率很低。

为了解决这一问题，PPO 作为一种 off-policy（纠误：应为近似 on-policy，只不过允许重复同一批数据多次进入训练）的方法应运而生。它不需要训练的决策 policy 和从中采样的 policy 完全相同，而是可以通过一个属性相近的 policy 来优化另一个 policy 的参数，这允许我们将一次采样的数据用于之后多次训练中，也允许我们对模型进行更大规模的分布式训练。具体做法上，PPO 通过引入重要性采样的方法，成功地把期望积分的测度转化另一个分布上的积分。最终推导出的公式为 

$$
\mathcal{L}
= -\frac{1}{N}
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
A^{\mathrm{GAE}}_{\theta'}
\!\left(s_n^{t}, a_n^{t}\right)
\frac{P_\theta\!\left(a_n^{t}\mid s_n^{t}\right)}
     {P_{\theta'}\!\left(a_n^{t}\mid s_n^{t}\right)}
$$

这可以看作通过两个 policy 输出概率的比值，来引导当前 policy 进行学习，输出比值越大则说明需要调整的越多。

需要注意的是，在 PPO 中 $\theta$ 和 $\theta'$ 这两个 policy 之间差距不能太大，否则就失去了参考意义，进而无法训练（也可以从数学角度考虑，差距过大，会导致两个概率之间的比值上溢或下溢，估计的方差极大）。针对这一问题，人们提出了两种解决方案。第一种是添加 KL 散度的惩罚项来约束两个 policy 之间的差距，即 

$$
\mathcal{L}
= -\frac{1}{N}
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
A^{\text{GAE}}_{\theta'}(s_n^t, a_n^t)\,
\frac{P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)}
+ \beta\, \mathrm{KL}(P_\theta, P_{\theta'})
$$

第二种方法是通过裁剪两个 policy 之间比值，防止单步更新的数值过大，导致训练不稳定，具体公式为

$$\mathcal{L}
= -\frac{1}{N}
\sum_{n=1}^{N}
\sum_{t=1}^{T_n}
\min \Bigg(
A^{\text{GAE}}_{\theta'}(s_n^t, a_n^t)
\frac{P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)},
\mathrm{clip}\!\left(
\frac{P_\theta(a_n^t \mid s_n^t)}{P_{\theta'}(a_n^t \mid s_n^t)},
1-\epsilon,
1+\epsilon
\right)
A^{\text{GAE}}_{\theta'}(s_n^t, a_n^t)
\Bigg)$$

在实践中，KL-PPO 很少直接被使用，而 Clipped-PPO 则成为了主流。

## 附录：强化学习的概念

![image.png](image.webp)

Environment（环境）：整体的环境，包括所有可能与Agent之间交互的物体、状态。

State（状态）：当前环境下的状态，会随着与Agent的交互等发生改变。

Observation（观察）：有时Agent无法观测到整个环境的状态，而是只能观察到State的一部分，我们把这部分叫做observation。以下不区分state与observation。

Agent（智能体）可以根据当前的State（状态），作出相对应的Action（动作），目标是获得尽可能多的Reward（奖励）。

Action space（策略空间）：可供agent选择的动作集合

Policy：策略函数，输入state，输出action的概率分布，一般用 $\pi$ 表示

- 强化学习中一般不直接贪心选取概率最大的action，而是从概率分布中采样，原因如下：

    - 训练的时候让agent探索更多的可能性，进而得到更好的策略

    - 推理时让输出具有多样性

Trajectory（轨迹）：一连串状态和动作的序列，用 $\tau$ 表示，也被称作 episode, rollout

- $\{s_0,~a_0,~s_1,~a_1,~\cdots\}$

- 状态的转移可能是随机或确定的

Return（回报）：从当前时间点到游戏结束的 reward 的累计和

- 通常追求的是 return 最大，而不是贪心让当前步 reward 最大

强化学习的目标：训练一个 policy 神经网络，在所有的 trajectory 中，得到 return 的期望最大。

## 参考文献

![零基础学习强化学习算法：ppo](https://www.bilibili.com/video/BV1iz421h7gb)
