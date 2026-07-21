---
title: "口胡内容（集训队讲题）"
date: 2020-10-22 20:50:00
categories:
    - 算法竞赛
tags:
    - STL
    - 位运算
    - 差分
    - 哈希
    - A*
    - 迭代加深
    - 倍增
    - 二维ST表
    - 二分
---

## STL

### 算法

* sort(): greater();
* upper_bound(), lower_bound(), binary_search();

<!--more-->

### 数据结构

* vector: push_back(), pop_back(), swap(), clear(), insert() **{O(n)}**;
* pair: first, second, make_pair();
* iterator: begin(), end();
* map: insert(), count(), size(), erase(map::iterator), swap(), clear();
* multimap: 可重集的map;
* set, stack, queue, priority_queue
* deque: front(), back(), push_front(), push_back(), pop_front(), pop_back(), **at()**

[UVA10125 Sumsets](https://www.luogu.com.cn/problem/UVA10125)

---

**位运算：[P2114 [NOI2014]起床困难综合症](https://www.luogu.com.cn/problem/P2114)**  
题意：有 $n(2 \le n \le 10^5)$ 扇门，每扇门有操作符 $op~(XOR, AND,OR)$ 和操作数 $t$，经过每扇门后自己的攻击力 $x$ 变为 $x~~op~~t(0 \le t \le 10^9)$，初始攻击力 $a$ 取 $0$ 到 $m(2 \le m \le 10^9)$ 的一个整数。求 $m$ 取多少可以使经过这 $n$ 扇门后的攻击力最大。

**输入格式：**

第 $1$ 行包含 $2$ 个整数，依次为 $n, m$，表示有 $n$ 扇防御门，初始攻击力为 $0$ 到 $m$ 之间的整数。

接下来 $n$ 行，依次表示每一扇防御门。每行包括一个字符串 $op$ 和一个非负整数 $t$，两者由一个空格隔开，且 $op$ 在前，$t$ 在后，$op$ 表示该防御门所对应的操作，$t$ 表示对应的参数。

**输出格式：**

一个整数，表示最大攻击力。

---

逐位贪心，对 $m$ 的每一位选择能使当前答案最大的值即可。$dfs$ 实现

```cpp
void work(int len,bool full) {
    if(!len) return;
    int up=full?bin[len]:1;
    opt[0]=opt[1]=-1;
    for(int i=0;i<=up;++i) opt[i]=solve(len,i);
    ans+=max(opt[0],opt[1])*(1<<(len-1));
    if(!full) work(len-1,false);
    else work(len-1,bin[len]?(opt[1]>opt[0]):true);
}

work(len,1);
```

---

**差分：[P4552 [Poetize6] IncDec Sequence](https://www.luogu.com.cn/problem/P4552)**

$Problem:$ 给定一个长度为 $n$ 的数列 ${a_1,a_2,\cdots,a_n}$，每次可以选择一个区间 $[l,r]$ ，使这个区间内的数都加 $1$ 或者都减 $1$。 

请问至少需要多少次操作才能使数列中的所有数都一样，并求出在保证最少次数的前提下，最终得到的数列有多少种。

**输入格式：**

第一行一个正整数 $n$。
接下来 $n$ 行,每行一个整数,第 $i+1 $行的整数表示 $a_i$。

**输出格式：**

第一行输出最少操作次数。
第二行输出最终能得到多少种结果。

---

$Solution:$ 定义差分数组 $c_i$，其中大于 $0$ 的元素个数为 $X$, 小于 $0$ 的个数为 $Y$。

原问题转化为：将差分数组中 $c_2$ ~ $c_n$ 变为 $0$ 的最小步数

每步操作分三种讨论：

1. 选取一个**正数 ($c_x$) 和一个负数 ($c_y$)**，使正数减1，负数加1，这样经过N次操作，我们一定可以以最少的代价将绝对值较小的一方归零，代价为 $abs(min(X,Y))$
2. 选取一个**正数 ($c_x$)**，使其与 *a[1]* 配对，这样，我们经过N次操作，一定可以将它归零，代价为：$abs(X)$
3. 选取一个**负数 ($c_y$)**，使其与 *a[n+1]* 配对，这样，我们经过N次操作，一定可以将它归零，代价为：$abs(Y)$

最小步数应该优先选第1种操作，再选2、3种方案

则 $ans1=\min(X,Y)+abs(X-Y)=\max(X,Y)$。

第二种操作后共有$abs(X-Y)+1$种可能，于是 $ans2=abs(X-Y)+1$。

---

**[P1438 无聊的数列](https://www.luogu.com.cn/problem/P1438)**

维护一个数列 $a_i$，支持两种操作：
 - `1 l r K D`：给出一个长度等于 $r-l+1$ 的等差数列，首项为 $K$，公差为 $D$，并将它对应加到 $[l,r]$ 范围中的每一个数上。即：令 $a_l=a_l+K,~a_{l+1}=a_{l+1}+K+D,~\ldots~,~a_r=a_r+K+(r-l) \times D$。
 - `2 p`：询问序列的第 $p$ 个数的值 $a_p$。

**输入格式：**

第一行两个整数 $n,m$ 表示数列长度和操作个数。

第二行 $n$ 个整数，第 $i$ 个数表示 $a_i$。

接下来的 $m$ 行，每行先输入一个整数 $opt$。

若 $opt=1$ 则再输入四个整数 $l\ r\ K\ D$；

若 $opt=2$ 则再输入一个整数 $p$。

**输出格式：**

对于每个询问，一行一个整数表示答案。

---

考虑**差分**，用线段树维护从 $1$ 到 $n$ 的差分数组 $c_i$

- 修改 $a_l$ 到 $a_r$ 时，令 $c_l=c_l+k,~c_i=c_i+d(l+1 \le i \le r),~c_{r+1}=c_{r+1}-k-(r-l+1) \times d$

- 单点查询 $a_k$，输出 $\sum_{i=1}^k c_i$ 即可。 

```cpp
scanf("%d",&op);
if(op==1){
	scanf("%d%d%d%d",&x,&y,&k,&d);
	add(1,x,x,k);
	if(y>x) add(1,x+1,y,d);
	if(y+1<=n) add(1,y+1,y+1,-1ll*d*(y-x)-k);
}else if(op==2){
	scanf("%d",&x);
	printf("%lld\n",query(1,1,x));
```

---

**树上差分：[P3128 [USACO15DEC]Max Flow P](https://www.luogu.com.cn/problem/P3128)**

Farmer John给他的牛棚的 $N(2≤N≤50000)$ 个隔间之间安装了 $N-1$ 根管道，隔间编号从 $1$ 到 $N$。所有隔间都被管道连通了。
他有 $K(1≤K≤100,000)$ 条运输牛奶的路线，第 $i$ 条路线从隔间 $s_i$ 运输到隔间 $t_i$。一条运输路线会给它的两个端点处的隔间以及中间途径的所有隔间带来一个单位的运输压力。输出压力最大的隔间的压力。

---

$Solution:$ **已知路径求树上所有节点被路径覆盖次数**

对每条路径的起点 $s_i$ 和终点 $t_i$ 的权值 $+1$ , 对 $lca(s_i,~t_i)$ 的权值 $-1$ , 对 $lca(s_i,~t_i)$ 的父节点权值 $-1$。

从根节点开始深搜，回溯时将其本身的权值加上所有子节点的权值。

每个节点的权值也是其被路径覆盖的次数。

---

**已知路径求被所有路径覆盖的边**

首先对已知的这 $n$ 条路径的 起点 $a$ 和 终点 $b$ 的权值 $+1$ ，并对 $lca(a, b)$ 的权值 $-2$。

从根节点开始深搜，回溯时将其本身的权值加上所有子节点的权值。

那么满足要求的边就是权值等于 $n$ 的节点 与其 父节点 所连的边。

---

**Hash表:** 利用Hash函数计算出哈希值，插入到对应的邻接表处。

[Acwing137. 雪花雪花雪花 ](https://www.acwing.com/problem/content/139/)

有 $n$ 片雪花，每片雪花由六个角组成，每个角都有长度。

第 $i$ 片雪花六个角的长度从某个角开始顺时针依次记为 $a_{i,1},a_{i,2},a_{i,6}$。

因为雪花的形状是封闭的环形，所以从任何一个角开始顺时针或逆时针往后记录长度，得到的六元组都代表形状相同的雪花。

例如 $a_{i,1},a_{i,2},a_{i,6}$ 和 $a_{i,2},a_{i,3},a_{i,6}，a_{i,1}$ 就是形状相同的雪花。

$a_{i,1},a_{i,2},a_{i,6}$ 和$a_{i,6},a_{i,5},a_{i,1}$也是形状相同的雪花。

我们称两片雪花形状相同，当且仅当它们各自从某一角开始顺时针或逆时针记录长度，能得到两个相同的六元组。

求这N片雪花中是否存在两片形状相同的雪花。

**输入格式：**

第一行输入一个整数 $n$，代表雪花的数量。

接下来 $n$ 行，每行描述一片雪花。

每行包含 $6$ 个整数，分别代表雪花的六个角的长度（这六个数即为从雪花的随机一个角顺时针或逆时针记录长度得到）。

**输出格式：**

如果不存在两片形状相同的雪花，则输出：`No two snowflakes are alike.`

如果存在两片形状相同的雪花，则输出：`Twin snowflakes found.`

**数据范围：** $1≤n≤100000,~0≤a_{i,j}<10000000$

---

$Solution:$ 将每个雪花对应的序列代入Hash函数$\sum_{i=1}^{6}a_i*6^i$，存进邻接表暴力判重即可。

---

**Hash：**

[P6739 [BalticOI 2014 Day1] Three Friends](https://www.luogu.com.cn/problem/P6739)

有一个字符串 $S$，对他进行操作：
1. 将 $S$ 复制为两份，存在字符串 $T$ 中
2. 在 $T$ 的某一位置上插入一个字符，得到字符串 $U$
现在给定 $U$，求 $S$。

**输出格式：**

- 如果不能通过上述的步骤从 $S$ 推到 $U$，输出 `NOT POSSIBLE`。
- 如果从 $U$ 得到的 $S$ 不是唯一的，输出 `NOT UNIQUE`。
- 否则，输出一个字符串 $S$。

---

字符串Hash：

```cpp
const int maxn=2e6+10,base=233;
int n;
int h[maxn];	//从1到i的哈希值
int p[maxn];	//base的i次方
char s[maxn];	//修改后的字符串

//获取删去第k的个字符后的哈希值
int getdel(int k){return h[k-1]*p[n-k]+(h[n]-h[k]*p[n-k]);}

int main(){
	scanf("%d%s",&n，s+1);
	if(n%2==0) {puts("NOT POSSIBLE");return 0;}	//长度为奇数不存在
	p[0]=1; h[0]=0;
	for(int i=1;i<=n;i++) p[i]=p[i-1]*base;
	for(int i=1;i<=n;i++) h[i]=h[i-1]*base+s[i];
	bool found=false;
	int fdidx;	//删除第fdidx个字符满足条件
	char ans[maxn]="";
	int mid=n/2;
	for(int i=1;i<=n;i++){
		int right=getdel(i),cur;
		if(i<=mid) cur=h[n]-h[n-mid]*p[mid];
		else cur=h[mid];
		if(cur*(p[mid]+1)==right){		//找到符合条件的串
			if(!found){
				found=true;
				fdidx=i;
				int cnt=1;
				for(int j=1;cnt<=mid;j++)
					if(i!=j) ans[cnt++]=s[j];
			}else if(getdel(fdidx)!=getdel(i)){		//原串不唯一
				puts("NOT UNIQUE");
				return 0;
			}
		}
	}
	if(!found) puts("NOT POSSIBLE");
	else printf("%s\n",ans+1);
	return 0;
}
```

---

**A*算法:** 类似 $bfs$，设计函数 $f(n)=g(n)+h(n)$。$g(n)$ 为已走过的步数，$h(n)$为预计还要走的步数（估价函数）。

将结点按照 $f(n)$ 的值加入优先队列，并不断从中从小到大取出结点进行搜索。

---

**~~一道红题~~：[Acwing178. 第K短路](https://www.acwing.com/problem/content/180/)**

$Problem:$ 给定一张 $N(1 \le N \le 1000)$ 个点（编号 $1,2…N$），$M(1 \le M \le 10^5)$ 条边的有向图，求从起点 $S$ 到终点 $T$ 的第 $K(1 \le K \le 1000)$ 短路的长度。如果第K短路不存在，则输出 `-1`。

**输入格式：**

第一行包含两个整数 $N$ 和 $M$。

接下来 $M$ 行，每行包含三个整数 $A,B,L$，表示点 $A$ 与点 $B$ 之间存在有向边，且边长为 $L$。

最后一行包含三个整数 $S,T,K$，分别表示起点 $S$，终点 $T$ 和第 $K$ 短路。

---

$Solution:$ 

- 最短路求出每个点到终点的距离
- A* 爆搜，估价函数定为该节点到终点的最短路长度。每次从优先队列中取出 $dis(x)+h(x)$ 最小的结点。当第 $k$ 次取出终点时，该路径即为第 $k$ 短路。

```cpp
void A_star(){
    if(dis[S]==dis[0]){puts("-1");return;}
    memset(vis,0,sizeof(vis));
    q.push(make_pair(dis[S],S));
    while(q.size()){
        int x=q.top().second,d=q.top().first+dis[x];
        q.pop(); vis[x]++;
        if(vis[T]==K) {printf("%d\n",d);return;}
        for(int i=head[x];i;i=e[i].next){
            int y=e[i].y;
            if(vis[y]!=K)q.push(make_pair(d+e[i].to+dis[y],y));
        }
    }
    puts("-1"); 
}
```

---

**迭代加深：**

**[UVA12558 埃及分数 Egyptian Fractions (HARD version)](https://www.luogu.com.cn/problem/UVA12558)**

- 给定一个分数 $\dfrac{a}{b}$，然后给定 $k$ 个数 $q_i$。
- 要求把 $\dfrac{a}{b}$ 分为几个不同的分子为 $1$ 的最简分数，要求分成的分数的分母中不能出现 $q_i$。
- 若有多种情况，输出拆分出分数最少的一种。
- 如果还有多种情况，应使第一大的分母最小，如果还不行，第二大最小，以此类推。
- **本题有 $t$ 组数据，** $t \le 100$。
- 其他数据范围：$2 \le a, b \le 876$，$0 \le k \le 5$，$\gcd(a,b)=1$，$2 \le q_i \le 1000$。

---

**倍增：**[P3865 【模板】ST表 - 洛谷](https://www.luogu.com.cn/problem/P3865)

```cpp
int main(){
	read(n),read(m);
	for(int i=1;i<=n;i++) read(a[i]),f[i][0]=a[i];
	for(int i=1;(1<<i)<=n;i++){		//枚举区间 
		for(int j=1;j+(1<<i)-1<=n;j++)	//枚举起点 
			f[j][i]=max(f[j][i-1],f[j+(1<<(i-1))][i-1]);
	}
	for(int i=1,l,r;i<=m;i++){
		read(l),read(r);
		int k=log2(r-l+1);		//中间值 
		write(max(f[l][k],f[r-(1<<k)+1][k]));
	}
	return 0;
}
```

---

**二维ST表：[P2216 [HAOI2007]理想的正方形](https://www.luogu.com.cn/problem/P2216)**

有一个 $a \times b$ 的整数组成的矩阵，现请你从中找出一个 $n \times n$ 的正方形区域，使得该区域所有数中的最大值和最小值的差最小。

**输入格式：**

第一行为 $3$ 个整数，分别表示 $a,b,n$ 的值

第二行至第 $a+1$ 行每行为 $b$ 个非负整数，表示矩阵中相应位置上的数。每行相邻两数之间用一空格分隔。

**输出格式**：

仅一个整数，为 $a \times b$ 矩阵中所有 $n \times n$ 正方形区域中的最大整数和最小整数的差值的最小值。

---

$Solution:$ $st[i][j][k]$ 表示以 $(i,j)$ 为左上角，边长为 $k$ 的正方形的权值最值。

```cpp
for(int k=1;k<=log2(min(n,m));k++)
    for(int i=1;i+(1<<k)-1<=n;i++)
        for(int j=1;j+(1<<k)-1<=m;j++)
            st[i][j][k]=max(max(max(st[i][j][k-1],st[i+(1<<(k-1))][j][k-1]),st[i][j+(1<<(k-1))][k-1]),st[i+(1<<(k-1))][j+(1<<(k-1))][k-1]);

for(int k=1;k<=log2(min(n,m));k++)
    for(int i=1;i+(1<<k)-1<=n;i++)
        for(int j=1;j+(1<<k)-1<=m;j++)
            st1[i][j][k]=min(min(min(st1[i][j][k-1],st1[i+(1<<(k-1))][j][k-1]),st1[i][j+(1<<(k-1))][k-1]),st1[i+(1<<(k-1))][j+(1<<(k-1))][k-1]);
```

---

**二分：[P1462 通往奥格瑞玛的道路](https://www.luogu.com.cn/problem/P1462)**

在艾泽拉斯，有 $n$ 个城市。编号为 $1,2,3,...,n$。

城市之间有 $m$ 条双向的公路，连接着两个城市，从某个城市到另一个城市，会遭到联盟的攻击，进而损失一定的血量。

每次经过一个城市，都会被收取一定的过路费（包括起点和终点）。路上并没有收费站。

假设 $1$ 为暴风城，$n$ 为奥格瑞玛，而他的血量最多为 $b$，出发时他的血量是满的。

歪嘴哦不希望花很多钱，他想知道，在可以到达奥格瑞玛的情况下，他所经过的所有城市中最多的一次收取的费用的最小值是多少。

**输入格式：**

第一行3个正整数，$n,m,b$。分别表示有 $n$ 个城市，$m$ 条公路，歪嘴哦的血量为 $b$。

接下来有 $n$ 行，每行 $1$ 个正整数，$f_i$。表示经过城市 $i$，需要交费 $f_i$ 元。

再接下来有 $m$ 行，每行 $3$ 个正整数，$ai，bi，ci(1 \le ai，b_i \le n)$。表示城市ai和城市bi之间有一条公路，如果从城市ai到城市bi，或者从城市bi到城市ai，会损失ci的血量。

---

```cpp
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>
#include <cstring>
#include <cstdio>
using namespace std;
int n,m,b,fee[10001],head[10001],cnt,dis[10001],v[10001];
struct Edge{
	int to;
	int val;
	int nxt;
}a[200001];
priority_queue<int,vector<int>,greater<int> > q;
void add(int x,int y,int z);
bool check(int x);
int main(){
    ios::sync_with_stdio(false);
	int l=0,r=0,mid;
	cin>>n>>m>>b;
	for(int i=1;i<=n;i++){
        cin>>fee[i];
        r=max(r,fee[i]);
	}
	for(int i=1,x,y,z;i<=m;i++){
		cin>>x>>y>>z;
		if(x==y) continue;
		add(x,y,z);
		add(y,x,z);
	}
	if(!check(r+1)){
		cout<<"AFK"<<endl;
		return 0;
	}
	l=max(fee[1],fee[n]);
	while(l<=r){
        mid=l+r>>1;
        if(check(mid)) r=mid-1;
        else l=mid+1;
	}
	cout<<l<<endl;
}
void add(int x,int y,int z){
	a[++cnt].to=y;
	a[cnt].val=z;
	a[cnt].nxt=head[x];
	head[x]=cnt;
	return;
}
bool check(int x){
    if(x<fee[1]||x<fee[n]) return false;
	for(int i=1;i<=n;i++){
        if(fee[i]>x) v[i]=true;
        else v[i]=false;
	}
	for(int i=1;i<=n;i++) dis[i]=0x3f3f3f3f;
	//memset(dis,0x3f,sizeof(dis));
	dis[1]=0;
	v[1]=true;
	q.push(1);
	while(!q.empty()){
		int cur=q.top();
		q.pop();
		for(int i=head[cur];i;i=a[i].nxt){
			if(dis[a[i].to]>dis[cur]+a[i].val){
				dis[a[i].to]=dis[cur]+a[i].val;
				if(!v[a[i].to]) q.push(a[i].to);
			}
		}
	}
	if(dis[n]>b) return false;
	return true;
}
```
