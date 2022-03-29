Numpy系列（一）：数组ndarray
==============================


@[TOC]
# 一、 Numpy数组简介

Numpy中所有功能都是基于N维数组数据结构**ndarray**的。

ndarray是**同类型数据**的集合，以 0 下标为开始进行元素的索引。
不同于python的List，ndarray中的每个元素在内存中都有相同存储大小的区域。

Numpy支持的数据类型与python原生类型不同，每个数据类型存储空间长度都是固定的。
Numpy常用的数据类型如下(通常用`import numpy as np`引用numpy)：

+ np.void
+ np.bool_    ：布尔类型
+ np.object_  ：任何对象
+ np.bytes_    ：字节序列
+ np.str_    ：字符
+ np.unicode_    ：字符串
+ np.int_    ：整形，自动根据系统平台自动转换为int32，int64等类型
+ np.float_    ：浮点型，自动根据系统平台自动转换为float32，float64等类型
+ np.complex_    :复数，自动根据系统平台自动转换为complex64，complex128等类型

+ 整数(Integers)类型
    - np.byte  ：字节
    - np.int8    ：8位整数
    - np.int16    ：16位整数
    - np.int32    ：32位整数
    - np.int64    ：64位整数
+ 无符号整数（Unsigned integers）类型
    - np.ubyte    ：字节
    - np.uint8    ：8位uint
    - np.uint16    ：16位uint
    - np.uint32    ：32位uint
    - np.uint64    ：64位uint
+ 浮点数字（Floating-point numbers）类型
    - np.float16    ：16位float
    - np.float32    ：32位float
    - np.float64    ：64位float
    - np.float96    ：96位float
    - np.float128    ：128位float
+ 复数（Complex floating-point numbers）类型
    - np.complex64    ：两个32位浮点数
    - np.complex128    ：两个64位浮点数
    - np.complex192    ：两个96位浮点数
    - np.complex256    ：两个128位浮点数

> numpy还有一些np.short，np.half，np.longlong等别名。不建议使用

> 在创建Numpy数组是可以指定python原生类型，但是会自动转换为对应的Numpy数据类型。
>
> Numpy实际上是用C语言编写的，其数据类型也基本上和C语言数据类型对应。

# 二、 思维导图

![NumpyN维数组](https://img-blog.csdnimg.cn/aab6786578a94b1c80780e373e8b2cce.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 三、 Numpy数组常用功能
## 1. 创建Numpy数组

### 1.1 用python序列创建数组

~~~
arr1 = np.array([1,2,3])              #list, set, tuple都可以用作参数
arr2 = np.array([[1,2],[3,4]])       #创建多维数组
~~~

> 创新的数组与参数中的list不共享内存。 

~~~
arr1 = np.array([1,2,3], np.int8)                       #指定数据类型
arr2 = np.array([1,2,3], dtype=np.float16)       #指定数据类型
~~~

### 1.2 自动生成数组

#### 1.2.1 zeros,ones,eye数组生成

**创建全0、全1、单位矩阵等特殊数组**

~~~
np.empty(2)                 #元素只分配内存，不初始化 
np.empty((2, 2))           #多维数组，参数必须为元组。list等序列会被自动转化为元组。

np.zeros(2)                  #元素全部为零 
np.zeros((2, 2))            #多维全零数组

np.ones(2)                  #元素全部为1
np.ones((2, 2))            #多维全为1数组

np.full(2, x)                 #所有元素值均为x
np.full((2,2), x)            #多维元素全部相同数组

np.identity(2)             #2×2单位矩阵，对角线为1，其余数值为0的对角方阵 

np.eye(2)                  #2×2单位矩阵
np.eye(2, 2)              #2×2单位矩阵。注意eye方法比较特殊，参数不是元组
np.eye(4, 3)              #类似单位矩阵。矩阵下标相同的位置为1 ，其余为数值为零
~~~

所有创建方法都可以用的dtype参数指定数据类型，不指定时，默认为float类型：

~~~
np.empty((2, 2), dtype=np.float_)
np.zeros((2, 2), dtype=np.float16)
np.ones((2, 2), dtype=np.float32)
np.full((2,2), 3, dtype=np.int8)
np.identity(2, dtype=np.int16)
np.eye(3, 2, dtype=np.complex_)
~~~

**根据已有数组的形状和数据类型创建数组**

~~~
np.empty_like(a)    #等同于np.empty(a的形状, dtype=a的数据类型) 
np.zeros_like(a)
np.ones_like(a)
np.full_like(a, 4) 
~~~

> 没有identity_like和eye_like

#### 1.2.2 数列数组生成

**range范围数组**

`np.arange(start=0, end, step=1)` ：生成`[start, end)`区间的数列，步长为step。

~~~
np.arange(10)    #[0, 10)步长为1的数组。注意是arange，不是range。应该是array range的意思。 
np.arange(1, 5)    #[1, 2, 3, 4]
np.arange(0.1, 0.5, 0.1)    #[0.1, 0.2, 0.3, 0.4]

np.arange(1, 5, dtype=np.float16)    #指定数据类型
~~~

**等差数列数组**

`np.linsapce(start,end,count=50,endpoint=True)`：创建`[start, end]`元素个数为count的等差数列。
可以通过endpoint参数指定是否包含终值，默认包含终值。 

~~~
np.linspace(0, 10, 11)     # [0, 1, 2, ..., 10]
np.llinspace(0, 10 , 10, endpoint=False)    #[0, 1, 2, ..., 9] 

np.linspace(1, 50)    #[1, 2, 3, ..., 50]   默认元素个数为50个。
~~~

> 注意是**lin**space不是**line**space。 

**等比数列**

`logspace(start, stop, num=50, endpoint=True, base=10.0)`：
以base为基数，用linspace(sart, stop, num)为幂的等比数列。
即通过指定开始指数、结束指数和元素个数，以及基数创建等比数列。 

~~~
np.logspace(0, 3, 4)    #[1,10,100,1000]  即[10**0, 10**1, 10**2, 10**3] 
np.logspace(0, 3, 4, base=2)    #[1,2,4,8]  即[2**0, 2**1, 2**2, 2**3]
np.logspace(0, 3, 3, endpoint=False)    #[1,10,100]  即[10**0, 10**1, 10**2]
~~~

`geomspace(start, stop, num=50, endpoint=True)`创建开始元素为start，
结束元素为end，元素个数为num的等比数列。

~~~
np.geomspace(1, 1000, 4)    #[1,10,100,1000]
np.geomspace(2, 16, 3, endpoint=False)    #[2,4,8]
~~~

> 根据首尾和个数创建等比数列。**第一个数不能是0**。

np.linspace, np.logspace, np.geomspace也都可以用dtype参数指定数据类型。

#### 1.2.3 从内存序列或文件创建

**从字符串创建**

~~~
np.fromstring('abc', dtype=np.int8)    #字符串字节数必须是数据类型长度的整数倍。dtype指定为np.int16将会出错。
np.fromstring('abcdabcd')    #获得一个float64数字(64位电脑上结果)。字符串字节数必须是64的倍数。

#字符数组不能直接使用np.str_、np.bytes_、或者np.unicode_，必须指定str的字节数。
np.fromstring('abc',dtype="S1")    # 1字节字符串数据类型。 实际类型为np.bytes_
np.fromstring('abcd',dtype="S2")    #2字节字符串数据类型。
np.fromstring('abcdabcd',dtype="U2")    #2字节Unicode字符串数据类型。实际类型为np.str_
#对于中文，通常文字的编码会被分割，所以不宜于处理中文。 
~~~

> 建议fromstring都显式指定dtype，因为默认数据类型为float。字符串字节数必须是float的倍数。

> 数据类型S1,S2,S3,S4...Sn表示n字节长度的字符串。


**从缓存创建**

np.frombuffer类似于np.fromstring，但其第一个参数为byte序列。

~~~
np.frombuffer(b'abcdabcd')    #同np.fromstring('abcdabcd')
np.frombuffer(bytes('abcd', encoding='utf8'), dtype=np.int8)
np.frombuffer(bytes('abcd', encoding='utf8'), dtype='S1')
 
np.frombuffer(buff, dtype='S1', count=5, offset=6)    #从第6个字节开始读取5个S1类型数据。 
~~~

**从迭代器创建**

~~~
np.fromiter(range(5), np.int8)    #必须指定dtype。 
~~~

**从函数创建**

`np.fromfunction(f, t)`：f为函数，t为元组。生成形状为t的数组，
数组的元素为用函数f计算的值，函数的参数为：

+ 一维数组：函数参数为元素下标。
+ 多维数组：函数参数为第0维下标。
+ n个参数函数：维度应与函数维度相同。

> 第二个参数为数组的shape，必须为元组。 

~~~
np.fromfunction(np.abs, (2,))    #[0, 1, 2]
np.fromfunction(np.abs, (2,2))    #[[0,1], [1,1]]

np.fromfunction(np.add, (2,2))   #[[0,1], [1,2]]
~~~

**从文件创建**

~~~
np.loadtxt('a.txt')    #空格和换行分割的文本文件。每一行的数据个数必须相同 
~~~

~~~
arr1.tofile('a.data')    #保存数组到文件
np.fromfile('a.data')    #只能读取x.tofile("a.data")保存的数据。
#np.fromfile读取的数据为一维数组。不管保存时是几维，读取后都是一维。 
~~~

## 2. 数组副本和视图

~~~
arr2 = arr1.copy()    #获取arr1数组的一个拷贝，arr2和arr1不共享内存。
arr3 = arr1.view()    #获取arr1数组的一个视图，arr3和arr1共享内存，修改一个，另一个也会变化。
~~~

## 3. 数组属性

### 3.1  获取属性

~~~
arr1.shape    #获取数组形状，输出为元组，形式为(2,2)
arr1.size       #获取元素总数
arr1.dtype    #获取数组元素数据类型
~~~

### 3.2 改变数组数据类型

~~~
arr2 = arr1.astype(np.int16)    #改变元素数据类型，arr2和arr1不共享内存

arr1.dtype = np.float16    #直接改变数组的数据类型，数据内存不变。
~~~

> 不建议直接给dtype赋值，对int8数据类型数组，dtype修改为int16时，会导致两个数据合并为一个。
> 数组的形状也会自动改变。

### 3.3 改变数组形状

~~~
a.resize(3,1)    #修改数组a的形状为3行1列
a.resize(3,-1)   #-1表示根据其他数据自动推导，比如9个元素则形状为(3,3)，6个元素则数组形状为(3,2)

a.shape=(3,1)    #a的形状修改为3行1列
a.shape=(1,-1)   #-1表示根据其他维度自动推导 

b=a.reshape(3,1)    #返回改变形状的数组视图。a,b形状不同，但是共享内存。修改一个另一个也会改变 
~~~

数组转置

~~~
a.T                   #返回数组a的转置视图
a.transpose()    #返回数组a的转置视图
~~~

数组扁平化

~~~
b = a.ravel()    #返回把数组展开为一维数组的视图，a,b共享内存

b = a.flatten()    #返回把数组展开为一维数组的副本，a,b不共享内存
~~~

长度为1的维度操作

~~~
b = a.squeeze()    #删除长度为1的维度，返回视图。
#若a=np.array([[[1,2,3]]])，则返回b值为[1,2,3]

b = np.expand_dims(a,0)    #指定axis=0扩展一个维度的视图，相当于squeeze反操作 
~~~

> 行向量`a=np.array([1,2,3])`转换为列向量，b=np.expand_dims(a,0)。
> 注意，只能用np.expand_dims方式调用，不同用数组直接调用。

~~~
b.swapaxes(0,2)    #返回视图shape从2,3,5转为5,3,2
a.swapaxes(0,1)    #转置视图 
 
np.moveaxis(a,0,2)    #返回视图shape从2,3,5转变为3,5,2。与swapaxes不同。 
~~~

> 注意数组没有moveaxis方法。 

### 3.4 转换为标量

~~~
b = a.item()    #把只有一个元素的数组转换为标量，即数值，[1]或者[[[1]]]形式数组都可以转换为标量1
~~~

## 4. 索引和切片

### 4.1 下标索引

~~~
b = a[5]    #同python list。获取第下标为5的元素
b = a[-1]    #同python list。获取第倒数第一个的元素

#多维数组
b = a[2][1]    #同pythonlist。先获取2行数组，在获得第1列元素，即下标为(2,1)的元素。

b = a[2,1]    #同b=a[2][1] 

b = a.flat[2]    #数组的flat属性可以得到特别的一维迭代器（可以用下标索引） 相当于a.ravel()[2]
~~~

> 可以直接赋值，如`a[5]=1,a[2][-1]=2`

### 4.2 切片

~~~
b = a[1:3]    #同python list，获取1到3下标的数据视图，不含a[3]
b = a[:-1]    #同python list，省略开始下标
b =  a[:]      #同python list，省略下标，获取整个数组的视图
b = a[1:4:2]    #指定步长，即获取下标为1, 3的数组视图

b = a[::2, ::2]    #获取偶数行列

a[1:3]=1    #赋值，所有元素赋值为1
a[1:3]=1,2    #赋值数量与元素数量相同。
~~~

**切片类型**

~~~
idx = slice(None,None,2)    #创建切片索引，None表示省略值，等同于::2
#a[idx,idx]，a[idx][idx]，a[::2,::2]  相同
~~~

**python list做为下标**

一维数字列表做下标

~~~
b = a[[1,2,1,2]]    #由a[1],a[2],a[1],a[2]组成的一个新数组，不是视图，而是元素的拷贝

#但是a[[1,2]]=3,4 确可以改变a内元素的值。
~~~

> 可以重复对同一个元素取多次值

一维布尔列表做下标（下标的形状必须和数组相同）

~~~
a[[True, False, True]]    #True对应的位置元素组成的数组，即a[0], a[2]元素组成的视图
a[[True, False, True]]=1,2
~~~

> 老版本布尔数组中True,False分别被作为1和0 

多维列表作为下标

> 多维列表做下标会自动转换为元组，元组元素对应数组的维度。
> 列表的元素个数不能大于数组的维度
    
~~~
a[[i,j,k]]   #i,j,k都可以是数组，并且都可以是多维数组
~~~


**numpy数组作为下标**

一维数组做下标（与一维列表做下相同）

~~~
a[ np.array([1,2]) ]    #同a[[1,2]]。 
~~~

多维数组做下标

+ 一维数组用多维数组做下标得到多维数组，数组元素为对应下标元素位置的数据。
+ 多维数组做下标时，每个下标表示数组的第0维索引。

> 多维数组(np.array)做下标相当于是列表作为下标时，转换为元组的第一个元素。


**过滤**

用布尔运输作为下标。

~~~
a[a>5]    #获取值大于5的元素的视图
a[a>5]=1,2,3 
~~~

## 5. 数组操作

### 5.1 数组合并

**按指定维度合并（np.concatenate）**

~~~
>>> a = np.array([[1, 2], [3, 4]])
>>> b = np.array([[5, 6]])
>>> np.concatenate((a, b), axis=0)
array([[1, 2],
       [3, 4],
       [5, 6]])
>>> np.concatenate((a, b.T), axis=1)
array([[1, 2, 5],
       [3, 4, 6]])
>>> np.concatenate((a, b), axis=None)
array([1, 2, 3, 4, 5, 6])
~~~

**按垂直方向(行)堆叠（np.vstack）**

~~~
>>> a = np.array([1, 2, 3])
>>> b = np.array([4, 5, 6])
>>> np.vstack((a,b))
array([[1, 2, 3],
          [4, 5, 6]])
~~~

> 注意一维数组合并后变为了2维数组，使用concatenate合并时，合并后仍为一维数组。

**按水平方向(列)堆叠（np.hstack）**

~~~
a = np.array([1,2,3])
b = np.array([4,5,6])
np.hstack((a,b))
array([1, 2, 3, 4, 5, 6])
~~~

~~~
a = np.array([[1],[2],[3]])
b = np.array([[4],[5],[6]])
np.hstack((a,b))
array([[1, 4],
       [2, 5],
       [3, 6]])
~~~

**将一维数组作为列堆叠（np.column_stack)**

~~~
>>> a = np.array((1,2,3))
>>> b = np.array((2,3,4))
>>> np.column_stack((a,b))
array([[1, 2],
       [2, 3],
       [3, 4]])
~~~

**按深度方向(第3轴)堆叠（np.dstack）**

~~~
>>> a = np.array([1,2,3))
>>> b = np.array([2,3,4])
>>> np.dstack((a,b))
array([[[1, 2],
        [2, 3],
        [3, 4]]])
        
        
>>> a = np.array([[1],[2],[3]])
>>> b = np.array([[2],[3],[4]])
>>> np.dstack((a,b))
array([[[1, 2]],
       [[2, 3]],
       [[3, 4]]])
~~~

> 对于不足3维的数组，在尾部增加长度为1的维度。即数组形状从(M,N)自动转换为(M,N,1)

**沿指定轴方向堆叠（np.stack）**

~~~
>>> np.stack((a, b), axis=-1)
array([[1, 4],
       [2, 5],
       [3, 6]])
~~~

> axis值不能大于维度数，所以不能像hstack那样堆叠向量

### 5.2 数组分割

**沿指定轴分隔数组得到视图（np.split）**

指定分割大小(必须能够等分）

~~~
>>> x = np.arange(9)
>>> np.split(x, 3)
[array([0,  1,  2]), array([3,  4,  5]), array([6,  7,  8])]
~~~

指定分割位置

~~~
>>> x = np.arange(8)
>>> np.split(x, [3, 5, 6, 10])
[array([0,  1,  2]),
 array([3,  4]),
 array([5]),
 array([6,  7]),
 array([], dtype=float64)]
~~~

> 可以用axis参数指定分割的轴

**沿指定轴分割数组得到视图（np.array_split）**

与np.split唯一的不同就是可以不等分

~~~
>>> x = np.arange(8.0)
>>> np.array_split(x, 3)
[array([0.,  1.,  2.]), array([3.,  4.,  5.]), array([6.,  7.])]
~~~

**vsplit， hsplit， dsplit**

分别表示在垂直(第0维）、水平（第1维）、深度（第2维）方向分割数组。
用法类似

~~~
>>> x = np.arange(8).reshape(4, 2)
array([[0, 1],
       [2, 3],
       [4, 5],
       [6, 7]])
>>> np.vsplit(x,2) #只均分
[array([[0, 1],
       [2, 3]]), 
 array([[4, 5],
       [6, 7]])]
>>> np.vsplit(x, [1,2])
[array([[0, 1]]),
 array([[2, 3]]),
 array([[4, 5],
       [6, 7]])]
~~~

### 5.3 数组平铺

**平铺（np.tile）**

简单平铺

~~~
>>> a=np.array([1,2,3])
>>> np.tile(a,2)
array([1, 2, 3, 1, 2, 3])

>>> a=np.array([[1,2],[3,4]])
array([[1, 2],
       [3, 4]])
>>> np.tile(a,2)   #按-1维度方向平铺
array([[1, 2, 1, 2],
       [3, 4, 3, 4]])
~~~

多维平铺（参数为元组）

原理：

+ 首先数组的形状和参数的形状扩展为相同的维度
    - 扩展方法，(M,N)扩展为(1,M,N)、(1,1,M,N)形式
+ 然后对应维度平铺

~~~
>>> a = np.array([0, 1, 2])
>>> np.tile(a, (2, 2))
array([[0, 1, 2, 0, 1, 2],
       [0, 1, 2, 0, 1, 2]])

>>> a = np.array([[1,2],[3,4]])
>>> np.tile(a, (2,1,2))
array([[[1, 2, 1, 2],
        [3, 4, 3, 4]],
       [[1, 2, 1, 2],
        [3, 4, 3, 4]]])
~~~

**重复（np.repeat）**

重复指定维度上的元素。

~~~
>>> np.repeat(3, 4)
array([3, 3, 3, 3])

>>> x = np.array([[1,2],[3,4]])
>>> np.repeat(x, 2)  #axis=None
array([1, 1, 2, 2, 3, 3, 4, 4])

>>> np.repeat(x, 3, axis=1)
array([[1, 1, 1, 2, 2, 2],
       [3, 3, 3, 4, 4, 4]])
~~~


### 5.4 数组增删

**尾部追加**

不指定维度，即默认值axis=None。元数组和追加数组都会被展开为一维数组，然后追加。

~~~
>>> np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]])
array([1, 2, 3, ..., 7, 8, 9])
~~~

指定维度，原数组和被追加数组维度数必须相同。

~~~
>>> np.append([[1, 2, 3], [4, 5, 6]], [[7, 8, 9]], axis=0)
array([[1, 2, 3],
       [4, 5, 6],
       [7, 8, 9]])

>>> np.append([[1, 2, 3], [4, 5, 6]], [7, 8, 9], axis=0)
Traceback (most recent call last):
    ...
ValueError
~~~

**插入元素**

不指定维度，默认axis=None

~~~
>>> a = np.array([[1, 1], [2, 2], [3, 3]])
array([[1, 1],
       [2, 2],
       [3, 3]])
>>> np.insert(a, 1, 5)
array([1, 5, 1, ..., 2, 3, 3])
~~~

插入简单元素

~~~
>>> np.insert(a, 1, 5, axis=1)
array([[1, 5, 1],
       [2, 5, 2],
       [3, 5, 3]])
~~~

插入序列元素

~~~
>>> np.insert(a, 1, [1, 2, 3], axis=1)
array([[1, 1, 1],
       [2, 2, 2],
       [3, 3, 3]])

>>> np.insert(a, 0, [5,5], axis=0)
array([[5, 5],
       [1, 1],
       [2, 2],
       [3, 3]])
~~~


**删除元素**

删除指定维度、指定位置的元素

~~~
>>> arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
array([[ 1,  2,  3,  4],
       [ 5,  6,  7,  8],
       [ 9, 10, 11, 12]])

>>> np.delete(arr, 1, axis=0)  #不改变元数组，得到的数组与原数组不共享内存
array([[ 1,  2,  3,  4],
       [ 9, 10, 11, 12]])

>>> np.delete(arr, [1,2], axis=1)
array([[ 1,  4],
       [ 5,  8],
       [ 9, 12]])

>>> np.delete(arr, slice(0,2), axis=1)  #删除切片指定位置
array([[ 3,  4],
       [ 7,  8],
       [11, 12]])
>>> np.delete(arr, np.s_[:2], axis=1)  #np.s_可以用:表示切片语法
array([[ 3,  4],
       [ 7,  8],
       [11, 12]])
~~~

**删除向量首尾部的零**

~~~
>>> a = np.array([0, 0, 0, 1, 2, 3, 0, 2, 1, 0])
>>> np.trim_zeros(a)
array([1, 2, 3, 0, 2, 1])

>>> a=np.array([[0],[1]])
>>> np.trim_zeros(a)
array([[1]])
~~~

**去除重复数字**

不指定维度

~~~
>>> np.unique([1, 1, 2, 2, 3, 3])
array([1, 2, 3])

>>> a = np.array([[1, 1], [2, 3]])
>>> np.unique(a)
array([1, 2, 3])
~~~

指定维度

~~~
>>> a = np.array([[1, 0, 0], [1, 0, 0], [2, 3, 4]])
>>> np.unique(a, axis=0)
array([[1, 0, 0], [2, 3, 4]])
~~~

返回索引（每个元素第一次出现的索引）

~~~
>>> a = np.array([0,0,1,1,2,2])
>>> b,idx = np.unique(a, return_index=True)
>>> b
array([0, 1, 2])
>>> idx      #b与a[idx]相等
array([0, 2, 4], dtype=int64)
~~~

返回反向索引。

> 返回输出的唯一值数组out的下标索引idx，使得`out[idx]`可以得到原数组。所以idx一般会有重复值

~~~
>>> a = np.array([0,0,1,1,2,2])
>>> b,idx = np.unique(a, return_inverse=True)
>>> b
array([0, 1, 2])
>>> idx     #b[idx]与a相等
array([0, 0, 1, 1, 2, 2], dtype=int64)
~~~

返回重复元素的个数

~~~
>>> a = np.array([1, 2, 6, 4, 2, 3, 2])
>>> values, counts = np.unique(a, return_counts=True)
>>> values
array([1, 2, 3, 4, 6])
>>> counts
array([1, 3, 1, 1, 1])
~~~

<br>

> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，
> 持续修订更新中。
>
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>