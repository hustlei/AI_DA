Numpy重点知识（四）：函数库之1基础数学函数

@[TOC]

# 一、 Numpy基础数学函数

除了基本的数学、逻辑运算，Numpy数组内置了很多函数库，常用的主要有：

+ 基本函数(最大最小、四舍五入、排序查找计数、公约数、公倍数)
+ 和差积(积加、累乘、叉乘、梯度)
+ 指数对数
+ 三角函数
+ 双曲函数
+ 复数运算

# 二、 思维导图

![Numpy基础数学函数](https://img-blog.csdnimg.cn/6ebd81319e6643709bfffd1f7d3792c4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

# 三、 函数简介

## 1. 基本函数

### 1.1 最大最小

**1.1.1 数组内部比较**

+ `max(a)`, `np.max(a)`, `a.max()`：获取数组a中最大的数值

+ `np.argmax(a)`：返回最大值的下标
  
  - `np.nanargmax(a)`：忽略NaN，返回最大值下标

+ `np.amax(a,axis=1)`：按指定维度找出最大值

+ `min(a)`, `np.min(a)`, `a.min()`：获取a中最大的数值

+ `np.argmin(a)`：返回最大值的下标
  
  - `np.nanargmin(a)`：忽略NaN，返回最大值下标

+ `np.amin(a,axis=1)`：按指定维度找出最小值

**1.1.2 数组间比较**

+ `np.maximum(x,y)`：返回x,y每个元素比较较大的值组成的数组。x,y必须形状相同
  - `np.fmax(x,y)`：类似maximum(x,y)。但是在比较的数有NaN时，返回非NaN的元素
+ `np.minimum(x,y)`：返回x,y每个元素比较较小的值组成的数组
  - `np.fmin(x,y)`：类似minimum(x,y)。但是在比较的数有NaN时，返回非NaN的元素

### 1.2 四舍五入

**1.2.1 四舍五入**

+ `np.round(x,2)`：数组元素四舍五入小数点后2两位
  - `np.round_(x,3)`：同np.round(x,3)
  - `np.around(x,4)`：同np.round(x,4)

**1.2.2 近似取整**

+ `np.floor(x)`：向下(小)取整
+ `np.ceil(x)`：向上(大)取整
+ `np.fix(x)`：向0取整(数值类型仍为浮点型)
+ `np.trunc(x)`：忽略小数部分，和np.fix(x)貌似结果相同

### 1.3 排序查找计数

**1.3.1 排序**

+ `np.sort(a, axis=-1, kind=None)`：根据指定轴、指定算法，返回升序排列副本
  - kind={‘quicksort’, ‘mergesort’, ‘heapsort’, ‘stable’}
    + quicksort:默认值，快速排序
    + mergesort:归并排序
    + heapsort:堆排序
    + stable:稳定排序
  - `a.sort()`：直接改变a内元素的顺序
  - `np.argsort(a,axis=-1,kind=None)`：返回排序后的索引(下标)
+ `np.sort_complex(a)`：复数排序，先按实部排序，然后按虚部排序

**1.3.2 查找**

+ `np.nonzero(a)`：返回非0元素的下标
+ `np.where(a>0)`：返回满足条件的元素下标
  - `np.where(a>0,x,y)`：满足条件的位置填充x，其他位置填充y。
    + `np.where(a>0,a,y)`：满足条件的位置元素值不变

**1.3.3 计数**

+ `np.count_nonzero(a)`：返回非0元素个数
+ `x,count=np.unique(a,return_counts=True)`：返回数组包含的所有数值(去除重复，类似set)及重复次数

### 1.4 公约数、公倍数

+ `np.lcm(a,b)`：a,b绝对值的最小公倍数
+ `np.gcd(a,b)`：a,b绝对值的最大公约数

### 1.5 其他简单函数

+ `np.abs(x)`：求绝对值。np.fabs也是求绝对值，但是不支持复数
+ `np.sqrt(x)`：求平方根
+ `np.sign(x)`：正数返回1，负数返回-1

## 2. 和差积(和简单 四则运算有差别)

### 2.1 和

+ `np.sum(x,axis=0)`：按指定维度所有元素相加。维度减少，比如`[1,2,3]`输出6
+ `np.nonsum(x,axis=0)`：把NaN当做0计算
+ `np.cumsum(x,axis=0)`：累加但不改变维度，比如`[1,2,3]`累加输出`[1,3,6]`

### 2.2 差

+ `np.diff(x,axis=0)`：后一个元素减前一个元素。比如`[1,2,1]`输出`[1,-1]`

### 2.3 积

+ `np.prod(a,axis=0)`：指定维度所有元素相乘。维度减少，比如`[2,2,3]`输出12
  - `np.nanprod(a,axis=0)`：把NaN当做1计算
+ `np.cumprod(a,axis=0)`：累乘但不改变维度，比如`[1,2,3]`累加输出`[1,2,6]`
  - `np.nancumprod(a,axis=0)`：把NaN当做1计算

### 2.4 叉乘

+ `np.cross(a,b)`：叉乘，a,b只能是长度为2或3的向量。输出和参数长度相同的向量。

### 2.5 梯度

+ `np.gradient(a)`：计算元素间梯度。即`(a[i+1]-a[i-1])/2`
  - `np.gradient(y,x)`：计算y对x的梯度。即y对x的导数。实际上就是`(y[i+1]-y[i-1])/(x[i+1]-x[i-1])`

## 3 指数对数

### 3.1 指数

+ `np.power(x,y)`：计算x的y次方，即x**y
+ `np.exp(x)`：计算自然常数e的x次方，即e**x

### 3.2 对数

+ `np.log(x)`：基数为e，求对数
+ `np.log2(x)`：基数为2，求对数
+ `np.log10(x)`：基数为10，求对数

## 4 三角函数

### 4.1 角度转换

+ `np.deg2rad(x)`：角度转换为弧度
+ `np.rad2deg(x)`：弧度转换为角度
+ `np.radians(x)`：角度转换为弧度
+ `np.degrees(x)`：弧度转换为角度

### 4.2 三角函数

+ `np.sin(x)`：正弦函数
+ `np.cos(x)`：余弦函数
+ `np.tan(x)`：正割函数

### 4.3 反三角函数

+ `np.arcsin(x)`：反正弦函数
+ `np.arccos(x)`：反余弦函数
+ `np.arctan(x)`：反正割函数

### 4.4 三角形斜边长度

+ `np.hypot(x1,x2)`：x1,x2为直角三角型的直角边长，输出斜边长：sqrt(x1**2+x2**2)

## 5 双曲函数

### 5.1 双曲函数

+ `np.sinh(x)`：双曲正弦函数
+ `np.cosh(x)`：双曲余弦函数
+ `np.tanh(x)`：双曲正割函数

### 5.2 反双曲函数

+ `np.arcsinh(x)`：反双曲正弦函数
+ `np.arccosh(x)`：反双曲余弦函数
+ `np.arctanh(x)`：反双曲正割函数

## 6 复数运算

### 6.1 获取实部虚部

+ `np.real(z)`, `z.real`：获取实部
+ `np.imag(z)`, `z.imag`：获取虚部

### 6.2 获取复数角度

+ `np.angle(z)`：获取复数角度(默认是弧度)。np.angle(z,deg=True)：角度为单位

### 6.3 共轭

+ `np.conj(z)`：对每个元素进行共轭运算
  - `np.conjugate(z)`：同np.conj(z)

> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
