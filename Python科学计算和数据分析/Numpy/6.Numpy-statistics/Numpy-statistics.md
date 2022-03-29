Numpy系列（六）：函数库之3统计函数
===================================




@[TOC]

# 一、 简介

NumPy提供了很多统计函数，用于从数组中查找中位数，百分位数，标准差和方差，协方差，相关系数，
以及直方图统计等等。


# 二、 思维导图
![Numpy统计函数](https://img-blog.csdnimg.cn/90595190875b4d1d9ef66f32d7e00fbd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 三、 Numpy统计函数

## 1 顺序统计

+ `np.ptp(a,axis=None)`：最大值与最小值之差。ptp来源于peak to peak即峰间值
+ `np.median(a,axis=None)`：中位数（如果有偶数个元素，则取中间两个数的平均值）
+ `np.percentile(a,50,axis=None)`：百分位数。从小到达排序，找到X%位置的数值。（类似中位数的概念，但是可以指定任意位置）
    - `np.nanpercentile`：忽略NaN元素
+ `np.quantile(a,0.5,axis=None)`：类似percentile，但是百分比用小数指定。
    - `np.nanquantile`：忽略NaN元素

## 2 均值和方差

### 2.1 均值

+ `np.mean(a,axis=None)`：计算均值
    - `np.nanmean`：忽略NaN
+ `np.average(a,axis=None,weight=None)`：计算均值，但是可以指定权重
    - `np.nanaverage`：忽略NaN

### 2.2 方差

+ `np.var(a,axis=None)`：方差，即元素与均值差的平方和/元素数量，即E(X-E(X))
    -  np.nonvar：忽略NaN

### 2.3 标准差

+ `np.std(a,axis=None)`：标准差，方差的平方根
    - np.nanstd：忽略NaN

## 3 相关关系 

### 3.1 协方差

**3.1.1 概念和特点**

用于衡量两个变量的总体误差。而方差是协方差的一种特殊情况，即当两个变量是相同的情况。

`Cov(X,Y)=E[(X-E[X])(Y-E[Y])]=E[XY]-E[X]E[Y] =Σ(xi-mean(X))(yi-mean(Y))/(n-1)`

式中E表示期望即均值。 注意协方差最后除以的n-1，所以两个变量相同时和方差计算结果略有不同

协方差只能计算两个向量之间的cov

**3.1.2 协方差矩阵**

X,Y两个变量（向量）之间的协方差矩阵。

~~~
[[Cov(X,X),  Cov(X,Y)],
 [Cov(Y,X),  Cov(Y,Y)]]
~~~

X,Y,Z三个变量（向量）之间的协方差矩阵。

[[Cov(X,X),  Cov(X,Y), Cov(X,Z)]
 [Cov(Y,X),  Cov(Y,Y), Cov(Y,Z)]
 [Cov(Z,X),  Cov(Z,Y), Cov(Z,Z)]]

**3.1.3 Numpy协方差计算**

+ `np.cov(x)`：x为向量，计算x与x的协方差，输出为单变量数组。
+ `np.cov(x,y)`：x,y均为向量，计算x,y的协方差矩阵，输出为2x2数组
+ `np.cov(mat)`：mat为二维数组(n行)，每行作为一个变变量，计算多个变量之间的协方差矩阵。输出为nxn数组

### 3.2 皮尔逊相关系数

**3.2.1 概念**

度量两个变量之间的相关程度，其值介于-1与1之间。0表示不相关，1和-1均表示强相关，只是斜率方向不同

计算方法：r=Cov(X,Y)/(X.std()*Y.std())

**3.2.2 相关系数矩阵**

与协方差矩阵类似

**3.2.3 Numpy相关系数计算**

+ `np.corrcoef(x)`：x为向量，计算x与x的相关系数，输出为单变量数组。
+ `np.corrcoef(x,y)`：x,y均为向量，计算x,y的相关系数矩阵，输出为2x2数组
+ `np.corrcoef(mat)`：mat为二维数组(n行)，每行作为一个变变量，计算多个变量之间的相关系数矩阵。输出为nxn数组

### 3.3 信号互相关

+ `np.correlate(x,y)`：信号互相关计算，得到两个向量的内积

计算两个信号的内积：内积大小反映的是向量的共线程度


## 4 直方图

### 4.1 一维直方图

+ `hist,bin_edges = np.histogram(a,bins=10)`：hist为10个区间内统计元素数量，区间边界为bin_edges
    - `hist, bin_edges = np.histogram(a, density=True)`:hist不再是统计个数，而是用比例，即0-1之间数值
    - `hist, bin_edges = np.histogram(a, bins=[0,1,3,5])`:指定bins，即统计区间的边界

> a无论多少维度都被展开为1维

### 4.2 二维直方图

+ `hist,xedges,yedges = np.histogram2d(x,y,bins=10)`
    - 二维直方图统计，x,y都只能是一维向量，且长度相同，输出的hist为nxn方形数组
+ `hist,xedges,yedges = np.histogram2d(x, y, bins=(xedges, yedges))`
    - 用列表指定想x方向和y方向的边界

### 4.3 多维直方图

+ `H, edges = np.histogramdd(data,bins=10)`：data每一列作为一个变量，进行多维统计
+ `H, edges = np.histogramdd(r, bins = (5, 8, 4))`：三列数据，指定三个bins

### 4.4 计算元素所在区间

+ `np.digitize(x,[1,2,4])`：返回每个元素所在区间的序号，比如在1-2之间返回1，在2-4之间返回2


<br><br>



> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
