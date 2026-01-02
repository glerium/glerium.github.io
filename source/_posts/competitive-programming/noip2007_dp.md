---
title: "[NOIP2007] 矩阵取数游戏——区间dp+高精"
date: 2020-08-08 02:24:00
categories:
    - 算法竞赛
tags:
    - 动态规划
    - 区间DP
---

### [NOIP2007] 矩阵取数游戏

---

[题目链接](https://loj.ac/problem/10152)

##### **思路：区间dp+高精**
<!--more-->

发现每行答案分别独立，于是考虑分行做区间dp，最终把每行的答案相加。

状态转移方程（对每一行，l为区间长度）：

$$
f[i][j]=\max(f[i+1][j]+a[i+1]*2^{m-l-1},~f[i][j-1]+a[j-1]*2^{m-l-1})
$$

代码：

```cpp
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
const int maxn=82;
int n,m,a[maxn][maxn];

struct bigInt{
    int v[31],size;
    bigInt(long long x=0) {memset(v,0,sizeof(v)); size=0; while(x) v[++size]=x%10,x/=10;}
    void write(){     //输出bigInt
        if(size==0) {putchar('0'); return;}
        for(int i=size;i>=1;i--) putchar(v[i]+'0');
    }
    bigInt operator+(const bigInt& rhs) {
        bigInt ans=0;
        ans.size=max(size,rhs.size);
        int kk=0;
        for(int i=1;i<=ans.size;i++){
            ans.v[i]=v[i]+rhs.v[i]+kk;
            kk=ans.v[i]/10;
            ans.v[i]%=10;
        }
        while(kk){
            ans.v[++ans.size]=kk%10;
            kk/=10;
        }
        return ans;
    }
    bigInt operator*(const int rhs){
        bigInt ans=0;
        int kk=0;
        ans.size=size;
        for(int i=1;i<=ans.size;i++){
             ans.v[i]=rhs*v[i]+kk;
             kk=ans.v[i]/10;
             ans.v[i]%=10;
        }
        while(kk){
            ans.v[++ans.size]=kk%10;
            kk/=10;
        }
        return ans;
    }
    bool operator<(const bigInt& rhs)const{   //重载小于号运算符，用于max函数比较
        if(size!=rhs.size) return size<rhs.size;
        for(int i=size;i>=1;i--)
            if(rhs.v[i]!=v[i]) return v[i]<rhs.v[i];
        return false;
    }
} f[maxn][maxn][maxn],pow2[maxn];

int main(){
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++){
        for(int j=1;j<=m;j++)
            scanf("%d",&a[i][j]);
    }
    pow2[0]=1;
    for(int i=1;i<=m+1;i++) pow2[i]=pow2[i-1]*2;
    for(int p=1;p<=n;p++){     //对每一行分别dp
        for(int l=m;l>=0;l--){     //枚举区间长度
            for(int i=1,j=i+l;i<=m&&j<=m;i++,j++)    //枚举左右区间
                f[p][i][j]=max(f[p][i][j+1]+pow2[m-l-1]*a[p][j+1],f[p][i-1][j]+pow2[m-l-1]*a[p][i-1]);
        }
    }
    bigInt ans=0;   //最终答案
    for(int i=1;i<=n;i++){
        bigInt cur=0;   //每行的答案
        for(int j=1;j<=m;j++)
            cur=max(cur,f[i][j][j]+pow2[m]*a[i][j]);
        ans=ans+cur;
    }
    ans.write();
    putchar('\n');
    return 0;
}
```
