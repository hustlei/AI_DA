
Scipy系列(一)：常量、IO及特殊函数
==============================


[Scipy系列目录](https://blog.csdn.net/hustlei/article/details/123093966)
<hr>

@[TOC]

<hr>


# 一、 概述

<font color=#888>相比Numpy，Scipy提供了非常多的物理常量以及单位信息，提供了非常多的基础数学函数和特殊数学函数。比如光速、普朗克常量、电子质量以及各种物理量单位，贝塞尔基函数、伽玛函数、贝塔函数等等。

> <font color=#888>非常的多，对于大多数人来说都用不到。除了专业或者具体某个行业工程研究，建议了解即可，即使需要使用，也可以在用的时候，了解相应部分即可。

<font color=#888>**scipy.io**提供了一些有用的输入输出函数，如matlab文件存取，wav声音文件存取等。比较有用。

# 二、 思维导图

![在这里插入图片描述](https://img-blog.csdnimg.cn/8a86389670a240ac9239895c84263985.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 三、 Scipy常量、IO及特殊函数

## 1 Scipy常量模块(scipy.constants)

### 1.1 数学常数

+ <font color=#888>contants.pi：圆周率。与np.pi相同
+ <font color=#888>contants.golden：黄金分割比例

> <font color=#888>数学常量并不多，pi建议还是直接用numpy.pi。

### 1.2 物理常数

> <font color=#888>scipy中的物理常量非常多，建议用的时候根据专业需要查。

<font color=#888>常见的常数：

+ <font color=#888>constants.c：真空光速
+ <font color=#888>constants.h：普朗克常数
+ <font color=#888>constants.G：牛顿引力常数
+ <font color=#888>constants.g：标准重力加速度
+ <font color=#888>constants.e：基本电荷
+ <font color=#888>constants.m_e：电子质量
+ <font color=#888>constants.m_p：质子质量
+ <font color=#888>constants.m_n：中子质量

### 1.3 单位信息

> <font color=#888>scipy中还包含非常多的单位信息，同样建议用的时候根据专业需要查。

<font color=#888>常见的单位

+ <font color=#888>constants.inch：1英寸等于多少米
+ <font color=#888>constants.gram：1克等于多少千克
+ <font color=#888>constants.pound：1磅等于多少千克
+ <font color=#888>constants.gallon：一加仑（美国）是多少立方米
+ <font color=#888>constants.bbl：一桶是多少立方米。石油 计量单位
+ <font color=#888>constants.hp：1马力等于多少瓦
+ <font color=#888>constants.mach：1马赫（约15℃，一个大气压）是多少米每秒
+ <font color=#888>constants.atm：标准大气压（帕斯卡）

> 数值为1个单位是多少ISO标准单位数值

## 2. 特殊函数(scipy.special)

<font color=#888>scipy.specail模块包含了少量实用的函数，同时还包含很多基础的数学函数，比如贝塞尔基函数、椭圆函数、统计函数的基础函数、超几何函数、正交多项式、开尔文函数、伽玛相关函数等等。大多数是比较基础的函数，通常我们直接使用的较少。

**实用函数**

+ <font color=#888>special.round(x)：四舍五入为最接近的整数
+ <font color=#888>special.cbrt(x)：求立方根
+ <font color=#888>special.sindg(x)：角度x(单位为度)的正弦
+ <font color=#888>special.cosdg(x)：角度x(单位为度)的余弦
+ <font color=#888>special.tandg(x)：角度x(单位为度)的正切
+ <font color=#888>special.cotdg(x)：角度x(单位为度)的余切

<font color=#888>**特别函数**

+ <font color=#888>special.gamma(z)：伽玛函数。伽玛函数是阶乘函数在实数和复数系上的扩展
+ <font color=#888>special.gammaln(z)：伽玛函数的对数
+ <font color=#888>special.beta(a,b)：贝塔函数
+ <font color=#888>special.betaln(a,b)：贝塔函数
+ <font color=#888>special.ellipj(u,m)：雅可比椭圆函数
+ <font color=#888>special.euler(n,x)：欧拉数计算
+ <font color=#888>special.binom(n,k)：二项式系数
+ <font color=#888>special.sici(x)：正弦和余弦的积分

## 3 输入输出(scipy.io)

### 3.1 matlab数据文件

+ <font color=#888>`mditc = io.loadmat(file)`：加载.mat类型的matlab文件。返回字典mdict，key为变量名
+ <font color=#888>`io.savemat(file, mdict)`：把mdict字典数据保存到matlab格式的.mat文件
+ <font color=#888>`io.whosmat(file)`：列出matlab格式的.mat文件中的变量。

### 3.2 wav声音文件

+ <font color=#888>`rate,data = io.wavfile.read(file)`：打开wav声音文件。rate为采样频率，data为数据
+ <font color=#888>`io.wavfile.write(file, rate, data)`：写入一个简单的未压缩的wav文件



<br>
<hr>


[Scipy系列目录](https://blog.csdn.net/hustlei/article/details/123093966)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>  
>  <font color=#888>**未经允许请勿转载。**
