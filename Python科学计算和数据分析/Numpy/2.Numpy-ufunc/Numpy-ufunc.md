Numpy系列（二）：数组函数ufunc




@[TOC]


# 一、 简介

Numpy数组支持很多数学运算，运算函数都被成为ufunc。
ufunc是universal function的缩写，是对数组的每个元素进行运算的函数。

NumPy内置的许多ufunc函数都是用C语言实现的，因此它们的计算速度非常快。

ufunc的特点：

+ 对数组逐个元素进行运算
+ ufunc速度比循环和列表推导式都快的多
+ ufunc函数会自动对维度不同的数组进行广播

# 二、 思维导图

![Numpy数组函数](https://img-blog.csdnimg.cn/846649849d774602a88ed501d151598f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

# 三、 Numpy数组函数

## 1. 基本运算

### 1.1	四则运算

逐个元素做运算，得到新的数组。


+ `y=x1+x2`或者`y=np.add(x1,x2)`：加法
+ `y=x1-x2`或者`y=np.subtract(x1,x2)`：减法
+ `y=-x`或者`np.negative(x)`：负值
+ `y=x1*x2`或者`y=np.multiply(x1,x2)`：乘法
+ `y=x1/x2`或者`np.divide(x1,x2)`：除法
+ `y=x1//x2`或者`np.floor_divide(x1,x2)`：整除
+ `y=x1**x2`或者`np.power(x1,x2)`：乘方
+ `y=x1%x2`或者`np.mod(x1,x2)`：求余

逐个元素运算，自动广播（广播见本文第5节），示例如下：

~~~
>>> x1=np.array([[0,1],[2,3]])
>>> x2=np.array([[2,2],[3,3]])
>>> x1+x2
array([[2, 3],
       [5, 6]])
>>> np.add(x1,x2)   #注意不能用x1.add(x2)
array([[2, 3],
       [5, 6]])

>>> x1+1            #常量广播
array([[1, 2],
       [3, 4]])

>>> x1+np.array([2,1])  #广播
array([[2, 2],
       [4, 4]])
~~~


### 1.2	比较运算

返回每个元素运算后得到的布尔数组。

+ `y = x1==x2`或者`np.equal(x1,x2)`：相等判断
+ `y = x1!=x2`或者`np.not_equal(x1,x2)`：不相等判断
+ `y = x1<x2`或者`np.less(x1,x2)`：小于判断
+ `y = x1<=x2`或者`np.less_equal(x1,x2)`：小于等于判断
+ `y = x1>x2`或者`np.greater(x1,x2)`：大于判断
+ `y = x1>=x2`或者`np.greater_eaual(x1,x2)`：大于等于判断

逐个元素运算，示例如下：

~~~
>>> x1=np.array([[0,1],[2,3]])
>>> x2=np.array([[3,1],[2,0]])
>>> x1 == x2
array([[False,  True],
       [ True, False]])

>>> x1 == 2            #常量广播
array([[False, False],
       [ True, False]])

>>> x1 == np.array([2,1])  #广播
array([[False,  True],
       [ True, False]])
~~~


### 1.3 布尔运算

python中的and，or，not等关键字无法重载，所以numpy数组不能用and，or，not进行布尔运算。

+ `np.all(a)`：是否数组所有元素是True
+ `np.any(a)`：是否任意元素是True
+ `np.logical_not(a==a)`
+ `np.logical_and(a==b,a>b)`
+ `np.logical_or(a==b,a>b)`
+ `np.logical_xor(a==b,a>b)`

示例如下：

~~~
>>> x1=np.array([[0,1],[2,3]])
>>> x2=np.array([[3,1],[2,0]])

>>> np.all(x1)
False
>>> np.any(x1)
True

>>> np.logical_and(x1,x2)  #0被认为是False，其他被认为是True
array([[False,  True],
       [ True, False]])

>>> x1==x2
array([[False,  True],
       [ True, False]])
>>> x1 > x2
array([[False, False],
       [False,  True]])
>>> np.logical_or(x1>x2,x1==x2)
array([[False,  True],
       [ True,  True]])
~~~


### 1.4 位运算

位运算符对每个元素进行运算。可以用~, &, |, ^等操作符。也可以使用bitwise_开头的ufunc。

+ `~a`或者`np.bitwise_not(a)`：按位非运算
+ `a&b`或者`np.bitwise_and(a,b)`：按位与运算
+ `a|b`或者`np.bitwise_or(a,b)`：按位或运算
+ `a^b`或者`np.bitwise_xor(a,b)`：按位异或运算

示例如下：

~~~
>>> x1=np.array([[0,1],[2,3]])
>>> x2=np.array([[3,1],[2,0]])

>>> ~x1
array([[-1, -2],
       [-3, -4]], dtype=int32)
>>> x1 & x2
array([[0, 1],
       [2, 0]], dtype=int32)
>>> x1 | x2
array([[3, 1],
       [2, 3]], dtype=int32)
>>> x1 ^ x2
array([[3, 0],
       [0, 3]], dtype=int32)
~~~

## 2. 自定义ufunc

python中也可以自定义ufunc，比直接使用python函数速度要快。方法如下：

1. 定义任意函数f()
    + 函数的参数列表数量的确定方法
        - 方法一：f()与目标ufunc参数数量相同
        - 方法二：运算时指定部分参数。yfunc1=yfunc(x, 0.1, 0.2)
2. 转换为ufunc
    + 方法1：yfun = np.frompyfunc(f, 1, 1) ：参数为输入输出参数个数
    + 方法2： yfunc = np.vectorize(f, otypes=[np.float])
3. 计算 y = yfun(x)
4. y.astype(np.float)：返回类型为object时，可以转为float


示例如下：

~~~
#1.定义函数
>>> def fsum2(x1, x2):
...    return x1 + x2/2


>>> x1 = np.array([1,2])
>>> x2 = np.array([2,2])

#2.转换ufunc方法一
>>> yfun1 = np.frompyfunc(fsum2, 2, 1)
>>>  yfun1(x1,x2)     #可以用_.astype(np.float)转换object为float类型
array([2.0, 3.0], dtype=object)

#3.转换ufunc方法二
>>> yfun2 = np.vectorize(fsum2, otypes=[np.float16])
>>> yfun2(x1,x2)
array([2., 3.], dtype=float16)
~~~

## 3. ufunc广播

ufunc计算需要计算数组形状相同。对于不同形状的数组，则会尝试转换为相同形状后再运算。
ufunc自动对计算数组形状进行扩展的过程，称为广播。

+ 最简单的广播：数组与常量的运算
+ 最经典广播案例 ：行向量与列向量形状分别为(M,) (1,N) 运算后得到(M,N)形状数组

### 3.1 广播的步骤及条件

**步骤**

+ 数组扩展为相同的维度
    - 所有数组都向维度最多的数组看齐
    - 维度不足的部分通过前面加1个维度补齐(增加的维度长度为1)
    - 如`[1,2]`增加第0维`[[1,2]]`，增加第1维`[[1],[2]]`
+ 对长度为1的维度进行repeat复制操作，使对应维度长度相同

**可广播条件**

+ 维度扩展后并可以repeat到形状相同，是数组可广播的基本要求
+ 如果两个数组某个维度上长度不同，必须有一个长度为1才能repeat。
    - 数组形状(3,1)可以repeat到(3,4)
    - 数组形状(3,2)不能repeat到(3,4)

示例：

~~~
>>> x1=np.array([1,2])           #shape为(2,)
>>> x2=np.array([[1,2],[3,4]])   #shape为(2,2)

#x1扩展为(2,2)的过程
#1. x1前面加一个维度，变为[[1,2]]    形状为(1,2)
#2. repeat复制长度为1的维度。变为[[1,2],[1,2]]   形状为(2,2)

>>> x1+x2
array([[2, 4],
       [4, 6]])
~~~


### 3.2 创建可广播数组

可广播数组即只进行广播第一步操作，数组扩展到相同的维度，但是形状不同。

#### 3.2.1 ogrid

ogrid可以根据指定范围，创建可广播的二维数组

~~~
>>> x,y = np.ogrid[:3,:3]   #注意是方括号
>>> x
array([[0],
       [1],
       [2]])
>>> y
array([[0, 1, 2]])

>>> x + y                   #运算中自动广播
array([[0, 1, 2],
       [1, 2, 3],
       [2, 3, 4]])
~~~

> 实际上就是创建"2维列向量+2维行向量"

#### 3.2.2 一维数组手工创建可广播数组

根据ogrid结果，可以看出一维数组转换为二维行、列向量就可以广播了。

**expand_dims增加维度**

~~~
>>> a=np.array([1,2,3])

>>> np.expand_dims(a,0)   #扩展为1行n列
array([[1, 2, 3]])
>>> np.expand_dims(a,1)   #扩展为n行1列
array([[1],
       [2],
       [3]])
~~~

**用None下标增加维度**

~~~
>>> a[None,:]             #扩展为1行n列
array([[1, 2, 3]])

>>> a[:,None]             #扩展为n行1列
array([[1],
       [2],
       [3]])
~~~

**用np.ix_增加维度**

~~~
>>> x,y=np.ix_(a,a)
>>> x
array([[1],
       [2],
       [3]])
>>> y
array([[1, 2, 3]])
~~~


 
#### 3.2.3 创建广播后的数组

**mgrid根据范围创建广播后的数组**

相比ogrid，mgrid更进一步，执行了广播的扩展维度和repeat两步操作。

~~~
>>> x,y = np.mgrid[:3,:3]   #创建ogrid[:3,:3]广播之后得到的数组
>>> x                       #列向量扩展列数
array([[0, 0, 0],
       [1, 1, 1],
       [2, 2, 2]])
>>> y                       #行向量扩展行数
array([[0, 1, 2],
       [0, 1, 2],
       [0, 1, 2]])
~~~

**meshgrid根据已有数组创建广播后的数组**

计算两个一维数组广播后得到的数组

~~~
>>> a=np.array([10,20])
>>> b=np.array([0,1,2])
>>> x,y=np.meshgrid(a,b)
>>> x
array([[10, 20],
       [10, 20],
       [10, 20]])
>>> y
array([[0, 0],
       [1, 1],
       [2, 2]])
~~~

> 注意meshgrid与mgrid输出的x,y是相反的。
> + meshgrid的x每行是相等的，即x是行广播，y是列广播
> + mgrid的y是每行相等的，即y是行广播，x是列广播
> 有些函数要求x必须是行广播的，比如matplotlib绘制速度场的时候。这时候可以用y,x=mgrid。
> meshgrid的indexing参数默认是'xy'，修改为'ij'则变为输出的x,y相当于'xy'参数得到的x,y交换

```
>>> x,y=np.meshgrid(a,b,indexing='ij')
>>> x
array([[10, 10, 10],
       [20, 20, 20]])
>>> y
array([[0, 1, 2],
       [0, 1, 2]])
```

## 4. ufunc的方法

函数自身的函数

### 4.1 reduce

按指定的维度**依次计算**，比如：累加，累减。指定的维度长度运算后变为1。结果比输入维度减少

~~~
>>> a=np.array([[ 0,  1,  2,  3],
... [ 4,  5,  6,  7],
... [ 8,  9, 10, 11],
... [12, 13, 14, 15]])

>>> np.add.reduce(a)
array([24, 28, 32, 36])
>>> np.add.reduce(a)
array([24, 28, 32, 36])

>>> np.subtract.reduce(a)
array([-24, -26, -28, -30])
~~~

两个输入一个输出的ufunc都可以进行reduce操作。比如：

+ np.subtract.reduce(a)：第0个元素依次减其他元素
+ np.multiply.reduce(a)：第0个元素依次乘其他元素
+ np.divide.reduce(a)：第0个元素依次除其他元素
+ ...
+ np.logical_and.reduce(a)
+ ...
+ np.bitwise_and.reduce(a)
+ ...


### 4.2 accumulate

根据指定维度，对当前位置之前的元素进行累计计算，结果和输入维度相同

~~~
>>> a=np.array([[ 0,  1,  2,  3],
... [ 4,  5,  6,  7],
... [ 8,  9, 10, 11],
... [12, 13, 14, 15]])

>>> np.add.accumulate(a)
array([[ 0,  1,  2,  3],
       [ 4,  6,  8, 10],
       [12, 15, 18, 21],
       [24, 28, 32, 36]], dtype=int32)
>>> np.add.accumulate(a,axis=1)
array([[ 0,  1,  3,  6],
       [ 4,  9, 15, 22],
       [ 8, 17, 27, 38],
       [12, 25, 39, 54]], dtype=int32)
~~~       
       
两个输入一个输出的ufunc都可以进行accumulate操作。比如：

+ np.subtract.accumulate(a)：第0个元素依次减其他元素
+ np.multiply.accumulate(a)：第0个元素依次乘其他元素
+ np.divide.accumulate(a)：第0个元素依次除其他元素
+ ...
+ np.logical_and.accumulate(a)
+ ...
+ np.bitwise_and.accumulate(a)
+ ...


### 4.3 outer

np.add.outer(a,b)：a中每一个元素，依次对b中的每个元素进行运算。
(M,N)形状数组和(J,K)形状数组运算后形状为(M,N,J,K)

~~~
>>> a=np.array([[10,20],[2,4]])
>>> b=np.array([1,2,3])
>>> np.add.outer(a,b)
array([[[11, 12, 13],
        [21, 22, 23]],

       [[ 3,  4,  5],
        [ 5,  6,  7]]])
~~~

两个输入一个输出的ufunc都可以进行outer操作。比如：

+ np.subtract.outer(a)：第0个元素依次减其他元素
+ np.multiply.outer(a)：第0个元素依次乘其他元素
+ np.divide.outer(a)：第0个元素依次除其他元素
+ ...
+ np.logical_and.outer(a)
+ ...
+ np.bitwise_and.outer(a)
+ ...


## 5. ufunc广播特例

针对一维向量，以下操作结果相同。

### 5.1 行向量和列向量自动广播

~~~
>>> a=np.array([[10],[20]])
>>> b=np.array([0,1,2])
>>> a+b
array([[10, 11, 12],
       [20, 21, 22]])
~~~

### 5.2 创建可广播数组

**根据范围用ogrid创建**

~~~
>>> x,y=np.ogrid[10:30:10,:3]
>>> x
array([[10],
       [20]])
>>> y
array([[0, 1, 2]])
>>> x+y
array([[10, 11, 12],
       [20, 21, 22]])
~~~

**expand_dims扩展维度为可广播数组**

~~~
>>> a=np.expand_dims(np.array([10,20]),1)
>>> a
array([[10],
       [20]])
>>> b=np.array([0,1,2])
>>> b
array([0, 1, 2])
>>> a+b
array([[10, 11, 12],
       [20, 21, 22]])
~~~

**None下标扩展维度为可广播数组**

~~~
>>> a=np.array([10,20])
>>> b=np.array([0,1,2])

>>> a[:,None]
array([[10],
       [20]])
>>> b[None,:]
array([[0, 1, 2]])

>>> a[:,None]+b[None,:]
array([[10, 11, 12],
       [20, 21, 22]])
~~~

**np.ix_扩展维度为可广播数组**

~~~
>>> a=np.array([10,20])
>>> b=np.array([0,1,2])

>>> x,y=np.ix_([10,20],[0,1,2])
>>> x
array([[10],
       [20]])
>>> y
array([[0, 1, 2]])

>>> x+y
array([[10, 11, 12],
       [20, 21, 22]])
~~~

### 5.3 根据范围创建广播后数组

**根据范围创建**

~~~
>>> x,y=np.mgrid[10:30:10,:3]
>>> x
array([[10, 10, 10],
       [20, 20, 20]])
>>> y
array([[0, 1, 2],
       [0, 1, 2]])
~~~

**根据已有数组创建**

~~~
>>> x,y=np.meshgrid([0,1,2],[10,20])
>>> x
array([[0, 1, 2],
       [0, 1, 2]])
>>> y
array([[10, 10, 10],
       [20, 20, 20]])
~~~

### 5.4 outer方法模拟向量广播

~~~
>>> np.add.outer([10,20],[0,1,2])
array([[10, 11, 12],
       [20, 21, 22]])
~~~

<br><br>


> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
