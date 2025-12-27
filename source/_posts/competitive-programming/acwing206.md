---
title: "[矩阵快速幂] AcWing 206 石头游戏"
date: 2020-09-14 01:23:00
categories:
    - 算法竞赛
tags:
    - 动态规划
---

别告诉我你不知道什么是[矩阵乘法](https://baike.baidu.com/item/%E7%9F%A9%E9%98%B5%E4%B9%98%E6%B3%95) (╯▽╰)  
[题目传送门](https://www.acwing.com/problem/content/208/)

看到题目，第一反应以为是模拟水题。结果一看数据范围$1e8$...  
模拟看来是不行了，需要想复杂度更优的解法

考虑**矩阵乘法**  
把$n \times m$的矩阵映射到一个含有 $nm$ 个元素的行向量  
我们可以尝试构造新的**状态转移矩阵** $a$，其行数和列数均为 $n m$，每次操作就相当于把一个$1 \times nm$的矩阵（每个格子里石子的数量）和状态转移矩阵相乘

令$a[0][0]=1$  
如果操作是$E$，相当于把它右面的元素加上它的值，即 `a[id(i,j)][id(i-1,j)]=1` （$id(i,j)$表示第$i$行$j$列元素在映射中的位置）；$W,S,E$同理  
如果操作是数字，就转移成 `a[id(i,j)][id(i,j)]=1; a[0][id(i,j)]=x;`（相当于保留原来的数字，同时加上新的数字 $x$）  
如果操作是 $D$，不用管，置空即可

此外，题目中有一个隐含条件，如果试图把石子放到不存在的格子里的话，把它当成 $D$ 即可，所以要判断格子是否在边界处

操作序列长度不超过$6$，所以状态转移矩阵一定会有循环节且循环节不超过 $lcm(1,2,3,4,5,6)=60$  
因此我们只需要计算出前$60$个转移矩阵  
求出前$60$个矩阵的乘积，利用**矩阵快速幂**来求解乘积的$\lfloor \frac{t}{60} \rfloor$次方，剩下的$t\%60$个可以直接暴力乘上去，就可以愉快地求出最终$t$秒时的状态转移矩阵啦

注意一点，数据范围会爆$int$，注意开$long~long$

---

*Code:*

```cpp
//Ê¯Í·ÓÎÏ·--¾ØÕó¿ìËÙÃÝ 
#include <cstdio>
#include <cstring>
#include <cctype>
#include <cstdlib>
#include <algorithm>
using namespace std;
typedef long long ll;
int n,m,t,act,r[70];
char op[15][15];
struct Matrix{
    int n,m;
    ll a[70][70];
    Matrix operator*(const Matrix& rhs)const{    //矩阵乘法
        Matrix ans;
        memset(&ans,0,sizeof(ans));
        ans.n=n,ans.m=rhs.m;
        for(int i=0;i<=ans.n;i++){
            for(int j=0;j<=ans.m;j++){
                for(int k=0;k<=m;k++)
                    ans.a[i][j]+=a[i][k]*rhs.a[k][j];
            }
        }
        return ans;
    }
    Matrix() {n=m=0;memset(a,0,sizeof(a));}
    void operator*=(const Matrix& rhs) {*this=*this*rhs;}
};
Matrix qpow(Matrix x,int y){    //矩阵快速幂
    Matrix ans;
    memset(&ans,0,sizeof(ans));
    ans.n=ans.m=x.n;
    for(int i=0;i<=n*m;i++) ans.a[i][i]=1;
    while(y){
        if(y&1) ans*=x;
        x*=x; y>>=1;
    }
    return ans;
}
int id(int x,int y) {return (x-1)*m+y;}    //建立映射
int mod(int x,int y){return x%y?x%y:y;}
signed main(){
    scanf("%d%d%d%d",&n,&m,&t,&act);
    {
        char tmp[15];
        for(int i=1;i<=n;i++){
            scanf("%s",tmp+1);
            for(int j=1;j<=m;j++) r[id(i,j)]=tmp[j]-'0'+1;    //每个格子对应的操作序列编号
        }
    }
    for(int i=1;i<=act;i++) scanf("%s",op[i]+1);
    
    /* computing */
    Matrix tt[65];    //1~60的状态转移矩阵
    memset(tt,0,sizeof(tt));
    for(int i=1;i<=60;i++){            //for 1 to 60 sec(matrix)
        tt[i].n=tt[i].m=n*m;
        tt[i].a[0][0]=1;
        for(int j=1;j<=n;j++){        //for line 1 to n
            for(int k=1;k<=m;k++){    //for col 1 to n
                char cop=op[r[id(j,k)]][mod(i,strlen(op[r[id(j,k)]]+1))];
                if(cop=='N'&&j>1) tt[i].a[id(j,k)][id(j-1,k)]=1;
                else if(cop=='W'&&k>1) tt[i].a[id(j,k)][id(j,k-1)]=1;
                else if(cop=='S'&&j<n) tt[i].a[id(j,k)][id(j+1,k)]=1;
                else if(cop=='E'&&k<m) tt[i].a[id(j,k)][id(j,k+1)]=1;
                else if(isdigit(cop)){    //指令是数字
                    tt[i].a[id(j,k)][id(j,k)]=1;
                    tt[i].a[0][id(j,k)]=cop-'0';
                }
            }
        }
    }
    
    Matrix eps;        //前60个矩阵的乘积
    eps.n=eps.m=n*m;
    for(int i=0;i<=eps.n;i++) eps.a[i][i]=1;
    for(int i=1;i<=60;i++) eps*=tt[i];

    Matrix final;    //前t个矩阵的乘积
    final.n=final.m=n*m;
    final=qpow(eps,t/60);
    for(int i=1;i<=t%60;i++) final*=tt[i];
    
    Matrix original;    //石子矩阵
    original.n=1,original.m=n*m;
    original.a[1][0]=1;
    original*=final;
    
    ll ans=0;
    for(int i=1;i<=n*m;i++) ans=max(ans,original.a[1][i]);
    printf("%lld\n",ans);
    return 0;
}
```
