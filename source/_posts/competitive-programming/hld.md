---
title: "树链剖分"
date: 2020-07-25 21:57:00
categories:
    - 算法竞赛
tags:
    - 树链剖分
---

https://www.cnblogs.com/ivanovcraft/p/9019090.html

时间复杂度：
$O(n*\log_{2}^{n})$


树链剖分：
1. dfs搜索图的基本信息：dep、fa、sz
2. dfs搜有关重链的信息：dfn、id、top
3. 读入点权，初始化线段树   线段树维护tag、max、sum
4. 操作
    1. 单点修改：[l, l]处，query当前，线段树差分add
    2. 查询最大：线段树分治查询max
    3. 查询和值：线段树分治查询sum
        * 查询策略：先pushdown，再二分查询两棵子树

---

### 板子题：[HAOI2015]树上操作

https://loj.ac/problem/2125

```cpp
#include <cstdio>
#include <cstring>
#include <algorithm>
using namespace std;
typedef long long ll;
const int maxn=3e6+10;
int n,m,cnt,head[maxn],ima,kkk[maxn],origin[maxn];    //图
int fa[maxn],dep[maxn],sz[maxn],son[maxn],top[maxn],dfn[maxn],low[maxn];    //树剖相关
struct Edge{
    int to,nxt;
}a[maxn<<1];
struct Node{
    int l,r;
    ll sum,tag;
}tr[maxn<<2];
void ins(int x,int y){a[++cnt]=(Edge){y,head[x]}; head[x]=cnt;}


void dfs1(int x,int from){
    fa[x]=from;
    dep[x]=dep[fa[x]]+1;
    sz[x]=1;
    for(int i=head[x];i;i=a[i].nxt){
        if(a[i].to==fa[x]) continue;
        dfs1(a[i].to,x);
        sz[x]+=sz[a[i].to];
        if(sz[son[x]]<sz[a[i].to])
            son[x]=a[i].to;
    }
}


void dfs2(int x,int t){
    top[x]=t;
    dfn[x]=++ima;
    low[x]=max(low[x],dfn[x]);
    if(!son[x]) return;
    dfs2(son[x],t);
    low[x]=max(low[x],low[son[x]]);
    for(int i=head[x];i;i=a[i].nxt){
        if(a[i].to==fa[x]||a[i].to==son[x]) continue;
        dfs2(a[i].to,a[i].to);
        low[x]=max(low[x],low[a[i].to]);
    }
}


void update(int k) {tr[k].sum=tr[k*2].sum+tr[k*2+1].sum;}
void build(int k,int l,int r){
    tr[k].l=l,tr[k].r=r;
    if(l==r){
        tr[k].sum=origin[l];
        return;
    }
    int mid=l+r>>1;
    build(k*2,l,mid);
    build(k*2+1,mid+1,r);
    update(k);
}


void pushdown(int k){
    tr[k*2].sum+=(tr[k*2].r-tr[k*2].l+1)*tr[k].tag;
    tr[k*2].tag+=tr[k].tag;
    tr[k*2+1].sum+=(tr[k*2+1].r-tr[k*2+1].l+1)*tr[k].tag;
    tr[k*2+1].tag+=tr[k].tag;
    tr[k].tag=0;
}


void add(int k,int l,int r,ll v){
    if(l<=tr[k].l&&tr[k].r<=r){
        tr[k].tag+=v;
        tr[k].sum+=(tr[k].r-tr[k].l+1)*v;
        return;
    }
    if(tr[k].tag) pushdown(k);
    int mid=tr[k].l+tr[k].r>>1;
    if(l<=mid) add(k*2,l,r,v);
    if(r>mid) add(k*2+1,l,r,v);
    update(k);
}


ll getSum(int k,int l,int r){
    if(l<=tr[k].l&&tr[k].r<=r) {return tr[k].sum;}
    if(tr[k].tag) pushdown(k);
    ll ans=0;
    int mid=tr[k].l+tr[k].r>>1;
    if(l<=mid) ans+=getSum(k*2,l,r);
    if(r>mid) ans+=getSum(k*2+1,l,r);
    return ans;
}


void makeAdd(int l,int r,ll v){
    while(top[l]!=top[r]){
        if(dep[top[l]]<dep[top[r]])    //st. l.top.dep>=r.top.dep
            swap(l,r);
        add(1,dfn[top[l]],dfn[l],v);
        l=fa[top[l]];
    }
    /* condition: dep[top[l]]==dep[top[r]] */
    if(dep[r]<dep[l]) //st. dep[l]<=dep[r]
        swap(l,r);
    add(1,dfn[l],dfn[r],v);
}


ll query(int l,int r){
    ll ans=0;
    while(dep[top[l]]!=dep[top[r]]){
        if(dep[top[l]]<dep[top[r]]) swap(l,r);
        ans+=getSum(1,dfn[top[l]],dfn[l]);
        l=fa[top[l]];
    }
    if(dep[l]>dep[r]) swap(l,r);
    ans+=getSum(1,dfn[l],dfn[r]);
    return ans;
}


int main(){
    scanf("%d%d",&n,&m);
    for(int i=1;i<=n;i++) scanf("%d",&kkk[i]);
    for(int i=1,x,y;i<=n-1;i++){
        scanf("%d%d",&x,&y);
        ins(x,y); ins(y,x);
    }
    dfs1(1,0);
    dfs2(1,1);
    for(int i=1;i<=n;i++) origin[dfn[i]]=kkk[i];
    build(1,1,n);
    for(int i=1,op,x,y;i<=m;i++){
        scanf("%d",&op);
        if(op==1){
            scanf("%d%d",&x,&y);
            makeAdd(x,x,y);
        }else if(op==2){
            scanf("%d%d",&x,&y);
            add(1,dfn[x],dfn[x]+sz[x]-1,y);
        }else{
            scanf("%d",&x);
            printf("%lld\n",query(1,x));
        }
    }
    return 0;
}
```


### P3384 【模板】轻重链剖分

https://www.luogu.com.cn/problem/P3384

```cpp
#include <iostream>
#include <cstdio>
using namespace std;
typedef long long ll;
const int maxn=1e5+10;
int n,m,r,p,cnt,head[maxn],ori[maxn],id[maxn],ima;
int sz[maxn],fa[maxn],dep[maxn],son[maxn],dfn[maxn],top[maxn];
struct Edge{
    int to,nxt;
}e[maxn<<1];
struct Node{
    int l,r;
    ll sum,tag;
}a[maxn<<2];


template<typename T> void read(T& x){
    T v=0,w=1; char ch=getchar();
    while(ch>'9'||ch<'0') {if(ch=='-') w=-1; ch=getchar();}
    while(ch<='9'&&ch>='0') v=v*10+ch-'0', ch=getchar();
    x=v*w; return;
}
template<typename T> void write(T x){
    if(x<0) putchar('-'), x=-x;
    if(x>9) write(x/10);
    putchar(x%10+'0');
}
void ins(int x,int y){e[++cnt]=(Edge){y,head[x]}; head[x]=cnt;}


void dfs1(int x,int from){
    fa[x]=from;
    dep[x]=dep[fa[x]]+1;
    sz[x]=1;
    for(int i=head[x];i;i=e[i].nxt){
        if(e[i].to==from) continue;
        dfs1(e[i].to,x);
        sz[x]+=sz[e[i].to];
        if(sz[son[x]]<sz[e[i].to]) son[x]=e[i].to;
    }
}


void dfs2(int x,int t){
    dfn[x]=++ima;
    id[ima]=x;
    top[x]=t;
    if(!son[x]) return;
    dfs2(son[x],t);
    for(int i=head[x];i;i=e[i].nxt){
        if(e[i].to==fa[x]||e[i].to==son[x]) continue;
        dfs2(e[i].to,e[i].to);
    }
}


void update(int k) {a[k].sum=(a[k*2].sum+a[k*2+1].sum)%p;}


void build(int k,int l,int r){
    a[k].l=l,a[k].r=r;
    if(l==r){
        a[k].sum=ori[id[l]];
        return;
    }
    int mid=l+r>>1;
    build(k*2,l,mid);
    build(k*2+1,mid+1,r);
    update(k);
}


void pushdown(int k){
    a[k*2].sum=(a[k*2].sum+(a[k*2].r-a[k*2].l+1)*a[k].tag)%p;
    a[k*2].tag=(a[k*2].tag+a[k].tag)%p;
    a[k*2+1].sum=(a[k*2+1].sum+(a[k*2+1].r-a[k*2+1].l+1)*a[k].tag)%p;
    a[k*2+1].tag=(a[k*2+1].tag+a[k].tag)%p;
    a[k].tag=0;
}


void add(int k,int l,int r,ll v){
    if(l<=a[k].l&&a[k].r<=r){
        a[k].sum=(a[k].sum+(a[k].r-a[k].l+1)*v)%p;
        a[k].tag=(a[k].tag+v)%p;
        return;
    }
    if(a[k].tag) pushdown(k);
    int mid=a[k].l+a[k].r>>1;
    if(l<=mid) add(k*2,l,r,v);
    if(r>mid) add(k*2+1,l,r,v);
    update(k);
}


ll sum(int k,int l,int r){
    if(l<=a[k].l&&a[k].r<=r) return a[k].sum;
    ll ans=0;
    if(a[k].tag) pushdown(k);
    int mid=a[k].l+a[k].r>>1;
    if(l<=mid) ans=(ans+sum(k*2,l,r))%p;
    if(r>mid) ans=(ans+sum(k*2+1,l,r))%p;
    return ans;
}


void makeAdd(int x,int y,ll v){
    while(top[x]!=top[y]){
        if(dep[top[x]]<dep[top[y]]) swap(x,y);
        add(1,dfn[top[x]],dfn[x],v);
        x=fa[top[x]];
    }
    if(dep[x]>dep[y]) swap(x,y);
    add(1,dfn[x],dfn[y],v);
}


ll makeSum(int x,int y){
    ll ans=0;
    while(top[x]!=top[y]){
        if(dep[top[x]]<dep[top[y]]) swap(x,y);
        ans=(ans+sum(1,dfn[top[x]],dfn[x]))%p;
        x=fa[top[x]];
    }
    if(dep[x]>dep[y]) swap(x,y);
    ans=(ans+sum(1,dfn[x],dfn[y]))%p;
    return ans;
}


int main(){
    read(n),read(m),read(r),read(p);
    for(int i=1;i<=n;i++) read(ori[i]);
    for(int i=1,x,y;i<=n-1;i++){
        read(x),read(y);
        ins(x,y),ins(y,x);
    }
    dfs1(r,0);
    dfs2(r,r);
    build(1,1,n);
    for(int i=1,op,x,y,z;i<=m;i++){
        read(op);
        switch(op){
            case 1:
                read(x),read(y),read(z);
                makeAdd(x,y,z);
                break;
            case 2:
                read(x),read(y);
                write(makeSum(x,y)),putchar('\n');
                break;
            case 3:
                read(x),read(y);
                add(1,dfn[x],dfn[x]+sz[x]-1,y);
                break;
            case 4:
                read(x);
                write(sum(1,dfn[x],dfn[x]+sz[x]-1)),putchar('\n');
                break;
        }
    }
    return 0;
}
```

### P2486 [SDOI2011]染色

![image.png](image.webp)

```cpp
#include <iostream>
#include <cstdio>
using namespace std;
const int maxn=1e6+10;
int n,m,head[maxn],cnt,ori[maxn],ima;
int fa[maxn],dep[maxn],son[maxn],sz[maxn],dfn[maxn],top[maxn],id[maxn];
struct Edge{
    int to,nxt;
}e[maxn<<1];
void ins(int x,int y) {e[++cnt]=(Edge){y,head[x]}; head[x]=cnt;}
struct Node{
    int l,r,tag,cnt,lc,rc;
}a[maxn<<2];


template<typename T> void read(T& x){
    T v=0,w=1; char ch=getchar();
    while(ch>'9'||ch<'0') {if(ch=='-') w=-1; ch=getchar();}
    while(ch<='9'&&ch>='0') v=v*10+ch-'0', ch=getchar();
    x=v*w; return;
}
template<typename T> void write(T x){
    if(x<0) x=-x, putchar('-');
    if(x>9) write(x/10);
    putchar(x%10+'0');
}
void getc(char &x){
    x=getchar();
    while(x^'C'&&x^'Q') x=getchar();
}


void dfs1(int x,int from){
    fa[x]=from;
    dep[x]=dep[from]+1;
    sz[x]=1;
    for(int i=head[x];i;i=e[i].nxt){
        if(e[i].to==from) continue;
        dfs1(e[i].to,x);
        sz[x]+=sz[e[i].to];
        if(sz[son[x]]<sz[e[i].to]) son[x]=e[i].to;
    }
}


void dfs2(int x,int t){
    top[x]=t;
    dfn[x]=++ima;
    id[ima]=x;
    if(!son[x]) return;
    dfs2(son[x],t);
    for(int i=head[x];i;i=e[i].nxt){
        if(e[i].to==fa[x]||e[i].to==son[x]) continue;
        dfs2(e[i].to,e[i].to);
    }
}


void pushdown(int k){
    a[k*2].cnt=a[k*2+1].cnt=1;
    a[k*2].tag=a[k*2+1].tag=a[k].tag;
    a[k*2].rc=a[k*2].lc=a[k*2+1].rc=a[k*2+1].lc=a[k].tag;
    a[k].tag=0;
}


void update(int k){
    a[k].cnt=a[k*2].cnt+a[k*2+1].cnt;
    if(a[k*2].rc==a[k*2+1].lc) a[k].cnt--;
    a[k].lc=a[k*2].lc; a[k].rc=a[k*2+1].rc;
}


void build(int k,int l,int r){
    a[k].l=l,a[k].r=r;
    if(l==r){
        a[k].lc=a[k].rc=ori[id[l]];
        a[k].cnt=1;
        return;
    }
    int mid=l+r>>1;
    build(k*2,l,mid);
    build(k*2+1,mid+1,r);
    update(k);
}


void change(int k,int l,int r,int c){
    if(l<=a[k].l&&a[k].r<=r){
        a[k].cnt=1;
        a[k].lc=a[k].rc=c;
        a[k].tag=c;
        return;
    }
    if(a[k].tag) pushdown(k);
    int mid=a[k].l+a[k].r>>1;
    if(l<=mid) change(k*2,l,r,c);
    if(r>mid) change(k*2+1,l,r,c);
    update(k);
}


int sum(int k,int l,int r){
    if(l<=a[k].l&&a[k].r<=r){
        return a[k].cnt;
    }
    if(a[k].tag) pushdown(k);
    int mid=a[k].l+a[k].r>>1,ans=0,son=0;
    if(l<=mid) ans+=sum(k*2,l,r),son++;
    if(r>mid) ans+=sum(k*2+1,l,r),son++;
    if(son==2&&a[k*2].rc==a[k*2+1].lc) ans--;
    return ans;
}


void makeChange(int x,int y,int c){
    while(top[x]!=top[y]){
        if(dep[top[x]]<dep[top[y]]) swap(x,y);
        change(1,dfn[top[x]],dfn[x],c);
        x=fa[top[x]];
    }
    if(dfn[x]>dfn[y]) swap(x,y);
    change(1,dfn[x],dfn[y],c);
}


int getColor(int k,int l){
    if(a[k].tag) pushdown(k);
    if(a[k].l==l&&a[k].r==l) return a[k].lc;
    int mid=a[k].l+a[k].r>>1;
    if(l<=mid) return getColor(k*2,l);
    else return getColor(k*2+1,l);
}


int getSum(int x,int y){
    int ans=0;
    while(top[x]!=top[y]){
        if(dep[top[x]]<dep[top[y]]) swap(x,y);
        ans+=sum(1,dfn[top[x]],dfn[x]);
        if(getColor(1,dfn[top[x]])==getColor(1,dfn[fa[top[x]]])) ans--;
        x=fa[top[x]];
    }
    if(dfn[x]>dfn[y]) swap(x,y);
    ans+=sum(1,dfn[x],dfn[y]);
    return ans;
}


int main(){
    read(n),read(m);
    for(int i=1;i<=n;i++) read(ori[i]);
    for(int i=1,x,y;i<=n-1;i++){
        read(x),read(y);
        ins(x,y),ins(y,x);
    }
    dfs1(1,0);
    dfs2(1,1);
    build(1,1,n);
    char op;
    for(int i=1,x,y,z;i<=m;i++){
        getc(op);
        if(op=='C'){
            read(x),read(y),read(z);
            makeChange(x,y,z);
        }else{
            read(x),read(y);
            write(getSum(x,y));
            putchar('\n');
        }
    }
    return 0;
}
```
