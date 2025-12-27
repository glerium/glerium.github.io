---
title: 实变函数课程笔记
date: 2024-06-13 22:33:00
tags:
    - 数学
categories:
    - 课程笔记
---
## 第一章

- 任何无限集必有一个可列子集

  - 对无限集而言，一定存在某个它的真子集，使得该子集与原集合对等

- [0,1] 是不可列的（用闭区间套定理证明）

- 任意个开集的并是开集

  - 任意个闭集的交是闭集

- **有限个**开集的交是开集

  - 有限个闭集的并是闭集

- 聚点：E 为点集，a 的任一邻域中都含有 E 中异于 a 的点（据此可以构造一个含于 E 的点列趋近于 a）

  - 聚点 a 的任意邻域中都有无穷多个异于 a 的点

  - 若集合中的任意一点都为聚点，则该集合为闭集

- 开集定义略，闭集为开集关于R的补集

- 导集：E 中一切聚点构成的集合 E'

  - 任何集合的导集一定为闭集（通过取补集证明）

- 孤立点：集合中不是聚点的点 E \ E'

- 闭包：集合与其导集的并 $\overline{E}=E \cup E{'}$

  - 闭集的闭包与其本身相等（由闭集的导集含于原集合这一点可直接推出）

- 完全集：E = E'

  - 若闭集中没有孤立点，则其为完全集

- $f(x)$ 连续 $\iff$$f(x)$ 关于任意开集的原像为开集

  - 需注意，连续函数在开集上的像未必为开集

---

- 任意有界非空开集 G 可以表示为至多可列个互不相交的构成区间的并（证明思路：对于每个G中的点x，可以取x周围的区间 $(\alpha,\beta)$ [同时可以保证任意两个区间的交集为空集]，从每个构成区间中取一个有理点，则存在构成区间到有理点的单射，因此构成区间的个数至多可列）

$G=\cup_{k} \left(\alpha_k,~\beta_k\right)$

- 康托尔三分集：$P_0$，其关于 $[0,1]$ 的补集记作 $G_0$

- 直径：$d(A)=\sup_{x,y\in A} \rho(x,y)$

- 点到集合的距离：$\rho(a,A)=\inf_{x\in A}\rho(a,x)$

  - 据此可构造含于 A 中的点列无限趋近于 a

  - 性质：若 A 为非空闭集，则下确界可在 A 中的某点上取到

    - 这一性质可以推广到集合与集合之间的距离函数，即当 A 和 B 都为闭集时，若其中之一有界，则点点距离的下确界一定可以在 A 与 B 中的某两点上取到

      - 注意该推论要求其中之一为有界集，否则可能不成立

---

- $\mu$ 为某无限集的势，则 $\mu+\mu=\mu$

- $\mu$ 为势，则 $2^{\mu} > \mu$

- 伯恩斯坦定理（Bernstein Th.）：若 $\lambda$ 与 $\mu$ 分别为势，且成立 $\lambda \le \mu,~ \mu \le \lambda$，则 $\lambda=\mu$

  - 推论：势与势之间，＞=＜三者有且仅有一者成立

- 策梅洛选择公理（Zermelo）

## 第二章

- 卡拉泰奥多里条件（Caratheodory）：有界集 $E$ 可测等价于，对任意集合 $A$，都成立等式

$$m^{\star}A=m^{\star}(A \cap E)+m^{\star}(A\cap \mathscr{C}E)$$

- $E \subset (a,b)$ 则 

$$m_{*}E+m^{*}E^c=b-a$$
$$m^{*}E+m_{*}E^c=b-a$$

- 内外测度的性质：

  - $m_{*}E \le m^{*}E$

  - $E_1 \subset E_2$ 则 $m_{*}E_1\le m_{*}E_2$ 且 $m^{*}E_1 \le m^{*} E_2$

  - 若 $E=\cup_{i=1}^{\infty}E_k$ 则 $m^{*}E \le \sum_{k=1}^{\infty}m^{*}E_k$

  - 上述条件中，若 $E_k$ 互不相交，则有 $m_{*}E \ge \sum_{k=1}^{\infty}m_{*}E_k$

- 零测集的任意子集可测且测度为0

- $E_0$ 为零测集，E为有界集，则 $E_0 \cup E$ 与 E 的可测性相同

- 可测性关于可列交、可列并、差、补运算封闭

- 有界集E可测等价于：对任意 $\varepsilon>0$，存在开集 $G \supset E$ 与闭集 $F \subset E$ 使得  

$$m(G -F) < \varepsilon$$

- 定理3.6

  - $E_k$为 (a,b) 中渐张可测集列 $E_1 \subset E_2 \subset \cdots$ 则 $E=\cup_{i=1}^{\infty}E_k$可测且 $mE=\lim_{n\to\infty} mE_k$

  - $E_k$为 (a,b) 中渐缩可测集列 $E_1 \supset E_2 \supset \cdots$ 则 $E=\cap_{i=1}^{\infty}E_k$可测且 $mE=\lim_{n\to\infty} mE_k$

- 博雷尔集（Borel）：以开集、闭集为对象作至多可列次交并运算

  - 博雷尔集可测

  - 可列个开集的交：$G_\delta$集、可列个闭集的并：$F_\sigma$集

- E可测，则存在 $G_\delta$集 A 与 $F_\sigma$集 B，使得 $A \supset E \supset B$ 且 $mA=mB=mE$

- 拓展到无界集上E的测度：$\lim_{n\to\infty}m\{ (-\alpha,\alpha) \cap E \}$（可能为无穷大）

  - 集合（有界或无界）的可测性对可列并和可列交运算均封闭

- 勒贝格测度具有平移不变性

- 一维不可测集是存在的（在承认选择公理的前提下）

- σ环：基本集X下关于差运算和可列并运算封闭的非空子集类（若将可列并运算弱化为有限并运算则为环）；若包含X本身，则称σ代数

  - 若Y为X的一个子集类，则称包含Y的最小环为由Y产生的环；由Y产生的σ环类似

## 第三章

- 定义在可测集上的连续函数必然可测；可测函数是连续函数的推广

- S在E上几乎处处成立：S不成立的点集的测度$E_0$为0；记作 $S, \text{a.e.}$

- $f_n(x)$可测，则 $\sup_n f_n(x)$ 和 $\inf_n f_n(x)$ 均可测

- $f(x)$ 可测，则 $f_{+}(x)$、$f_{-}(x)$ 和 $|f(x)|$ 均可测（提示：$|f(x)| = \sup\left\{ f(x),-f(x) \right\}$）

  - 推论：$\varlimsup_n f_n(x)$ 与 $\varliminf_n f_n(x)$ 均可测（用上下极限的定义证明）

- 任意可测函数都可以用简单函数来逼近：

  若f(x)在E上可测，则存在非负递增简单函数列 $\varphi_n(x)$ 

$$0\le\varphi_1(x)\le\varphi_2(x)\le\cdots$$

使得 $\lim_n \varphi_n(x)=f(x)$ 在E上处处成立

- 函数的可测性关于和差积商运算封闭（除法运算要求分母几乎处处不为零）

- 叶果罗夫定理（Egorov Th.）：设E为可测集， $mE<\infty$，$f_n(x)(n \in \N)$ 与 $f(x)$ 都在 E 上几乎处处有限且可测，且 $\{ f_n(x) \}$ 在 E 上几乎处处收敛于 $f(x)$。则对任意 $\delta>0$，存在可测集 $E_\delta \subset E$，$m(E-E_\delta)<\delta$，使得 $f_n(x)$ 在 $E_\delta$上一致收敛于 $f(x)$

  - 在有界可测集E上，几乎处处收敛与近一致收敛等价

- 依测度收敛：$f_n(x)$ 和 $f(x)$在E上可测，对任意 $\varepsilon>0$ 有 $\lim_{n\to\infty}m(|f_n(x)-f(x)|>\varepsilon)=0$

- 几乎处处收敛 → 依测度收敛

- 里斯定理（Riesz Th.）：$mE<\infty$，则 $\{f_n(x)\}$ 测度收敛于 $f(x)$ 等价于：对其任意子列 $f_{n_k}(x)$ 而言，都存在其子列 $f_{n_{k_i}}(x)$ 几乎处处收敛于 $f(x)$

  - 正向条件可以减弱： $f_n(x)$依测度收敛到$f(x)$，则存在其子列 $f_{n_k}(x)$ 几乎处处收敛于$f(x)$

- 测度收敛性关于线性运算、绝对值运算和 $\sup/\inf$ 运算封闭

- 鲁津定理（Lusin Th.）：对于有界集E上几乎处处有限的函数 f(x) 而言，f(x)为可测函数等价于：对任意 $\varepsilon>0$，存在闭集 $F \subset E,~m(E-F)<\varepsilon$，且 $f(x)$ 限制在F上连续

- 连续函数逼近有限可测函数：若可测函数f(x)在有限集E上定义且几乎处处有界，则对任意 $\varepsilon$ 而言，存在连续函数 g(x) 满足 $mE(f \neq g) < \varepsilon$

## 第四章

- 绝对连续性：f(x)在可测集E上可积，则对任意 $\varepsilon>0$ 存在 $\delta>0$，使得当 $me<\delta$ 便有

$$\left| \int_ef(x)dm \right| < \varepsilon$$

- σ可加性：f(x)在有界可测集E上可积，$E=\cup_{k=1}^{\infty}E_k$，所有 $E_k$均可测且两两不相交，则

$$\int_E f(x)dm=\sum_{i=1}^{\infty}\int_{E_k}f(x)dm$$

- 线性性1：f(x)在E上可积，则对任意 $c \in \R$，$cf(x)$ 可积且

$$\int_E cf(x)dm=c\int_E f(x)dm$$

- 线性性2：f和g在E上可积，则f+g也可积且

$$\int_E (f+g)dm=\int_E fdm + \int_E gdm$$

- 单调性：f和g在E上可积，且f≤g则

$$\int_Efdm \le \int_E gdm$$

- 唯一性定理：f在E上可积，则 $\int_E \left|f\right|dm=0$ 等价于 $f \sim 0$

- 连续函数逼近可积函数：若 $f(x)$ 可积于[a,b]，则对任意 ε 都存在连续函数 g(x)，使得 $\int_{[a,b]}|f(x)-g(x)|<\epsilon$

- 定理3.1：$f(x),u_n(x)$ 在可测集E上非负可测，且 $f(x)=\sum_{i=1}^n u_n(x)$，则

$$\int_E f(x)dm=\sum_{i=1}^n \int_E u_n(x)dm$$

- 勒维定理（Levi Th.）：$f_n(x)$ 在可测集E上可测，且满足

$$0 \le f_1(x) \le f_2(x) \le \cdots,\quad\lim_{n \to \infty}f_n(x)=f(x)$$

则有

$$\int_E f(x)dm=\lim_{n \to \infty} \int_Ef_n(x)dm$$

- 法杜定理（Fatou Th.）：$f_n(x)$ 在可测集上可测且 $f_n(x) \ge 0$ 则

$$\int_E \varliminf_{n\to\infty} f_n(x)dm \le \varliminf_{n\to \infty}\int_Ef_n(x)dm$$

（Levi与Fatou相互等价）

- 勒贝格控制收敛定理：$f_n(x)$ 在 E 上可测，且满足 $f_n(x)$ 依测度收敛于 $f(x)$，同时有可积函数 g(x) 使得几乎处处成立 $\left| f_n(x) \right| \le g(x)$ ，则 f 可积且有

$$\int_Ef(x)dm=\lim_{n\to\infty} \int_E f_n(x)dm$$

- 有界收敛定理（Lebesgue控制收敛的推论）： $mE \le \infty$，$f_n(x)$ 在 E 上可测且满足 $\left| f_n(x) \right| \le M$，$f(x)=\lim_{n \to \infty} f_n(x)$，则 f(x) 可积且

$$\int_Ef(x)dm=\lim_{n\to\infty} \int_E f_n(x)dm$$

（注意Lebesgue并不限制E为有界集，而有界收敛定理要求E有界）

- 勒贝格-维它利定理：$mE < \infty$， $\{ f_n \}$ 依测度收敛于 f 且 $f_n$ 可积，则

$$\lim_{n \to \infty} \int_E \left| f_n-f \right|dm=0$$

等价于 $f_n$ 在E上有等度的绝对连续积分

- 定义在有限区间上的函数若为R可积，则必L可积，且积分值相等

