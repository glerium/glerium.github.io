---
title: 数理统计课程笔记
date: 2024/05/27 17:41:48
tags:
    - 数理统计
    - 数学
categories:
    - 课程笔记
---
![image.webp](image.webp)

![image.webp](image%201.webp)

![image.webp](image%202.webp)

![image.webp](image%203.webp)

正态分布方差的性质：

$$E(s^2) = \sigma^2 $$

$$E(s^{\star2}) = \frac{n-1}{n}\sigma^2$$

二阶矩公式：

$$E (X^2) = Var(x)+(EX)^2$$

Fisher信息量：

$$I_X(\theta) = E\left(\frac{\partial L(x;\theta)}{\partial\theta}\right)^2=-E\left(\frac{\partial^2 L(x;\theta)}{\partial^2 \theta}\right)$$

C-R下界：

$$Var(\hat{g}(X)) \ge \frac{(g^{'}(\theta))^2}{I_X(\theta)}$$

常见分布的性质：

$$\begin{aligned}
P(\lambda) &= \Gamma(1,\lambda) \\
\chi^2(k) &= \Gamma(\frac{k}{2},\frac 12) \\
c\Gamma(\alpha,\lambda) &= \Gamma(\alpha, \frac{\lambda}{c}) \\ 
\Gamma({\alpha_1},{\lambda})+\Gamma(\alpha_2,\lambda)&=\Gamma(\alpha_1+\alpha_2,\lambda) \\
\chi^2(k_1) + \chi^2(k_2) &= \chi^2(k_1+k_2)
\end{aligned}$$

枢轴量方法：

求 $\theta$ 的置信区间，首先构造其估计 $\hat{\theta}$，然后构造枢轴量 $G(\theta,\hat{\theta})$，其分布 $f$ 与参数无关。然后求区间使得 $\mathbb{P} \left(a \le G \le b\right)=1-\alpha$。将 $a \le G \le b$ 作为关于 $\theta$ 的不等式解出来，解出的区间 $L_a \le \theta \le l_b$即为所求置信区间

由于使得概率等于 1-\alpha 的区间有无数种，因此可行的置信区间有无数个。通过一定的限制条件（如区间长度最短）可以将区间固定下来。

常见分布的期望和方差：

|**名称**|**期望**|**方差**|
|-|-|-|
|泊松分布 P(λ)|λ|λ|
|指数分布 λexp(-λx)|1/λ|1/(λ^2)|
|卡方分布 χ^2(n)|n|2n|
|Gamma分布|α/λ|α/(λ^2)|

![image.webp](image%204.webp)

修偏后的样本方差是对原始方差的无偏估计

---

![image.webp](image%205.webp)

---

![image.webp](image%206.webp)

---

![image.webp](image%207.webp)
