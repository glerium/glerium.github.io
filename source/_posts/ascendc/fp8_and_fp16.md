---
title: "FP8与FP16浮点数的二进制表示"
date: 2026-07-14 16:56:00
categories:
    - 学习笔记
tags:
    - 浮点数
    - 量化
---

## FP8和FP16浮点数的构成

浮点数的构成分为三部分：1bit符号位S、若干bit指数位E、若干bit尾数位M。

FP16内部是e5m10的结构。BSA算子在A5芯片上采用的是 fp8e4m3fn 精度。

此外还有 fp8e8m0 精度，只能表示正二的次幂，常用于 scale factor

<!--more-->

### FP16的表示

对于一个FP16数字 `(S,E,M)=[SEEEEEMMMMMMMMMM]` 而言，其表示的数字可以按照以下流程计算：

* 正规数：E≠0且E的所有位不全为1
$$f=(-1)^S \cdot  2^{E-15} \cdot \left(1 + \frac M {2^{10}}\right)$$
* 次正规数：E=0
$$f=(-1)^S \cdot 2^{-14} \cdot \frac M {2^{10}}$$
* 特殊值：E所有位均为1
	* M所有位均为0：表示 ±inf，正负取决于符号位S
	* M不全为0：表示nan

### FP8E4M3FN的表示

FP8E4M3FN的表示与FP16类似。

对于一个FP8E4M3FN数字 `(S,E,M)=[SEEEEMMM]` 而言，其表示的数字可以按照以下流程计算：

* 正规数：E≠0且E的所有位不全为1
$$f=(-1)^S \cdot  2^{E-7} \cdot \left(1 + \frac M {2^{3}}\right)$$
* 次正规数：E=0
$$f=(-1)^S \cdot 2^{-6} \cdot \frac M {2^{3}}$$
* 特殊值：E所有位均为1
	* M所有位均为0：表示 ±inf，正负取决于符号位S
	* M不全为0：表示nan

## FP8到FP16的转换

对比两者的结构：

```
FP16: SEEEEEMMMMMMMMMM
FP8 :  SEEEEMMM
```

设FP8数字F=(S,E,M)，FP16数字F'=(S',E',M')

则可以推导出：S'=S, E'=E+8, M'=M\*128
