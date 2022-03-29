Numpy系列（七）：函数库之4线性代数
==============================



@[TOC]

# 一、 简介

NumPy 提供了线性代数和多项式函数库。包含了线性代数中行列式、向量、矩阵，相关常用操作以及解方程和多项式的功能。

# 二、 思维导图

![numpy 线性代数思维导图](https://img-blog.csdnimg.cn/19f04d7475a347cbb451e6d0dafd3a48.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 三、 Numpy线性代数

## 1 矩阵和向量乘

### 1.1 矩阵乘法

+ `np.matmul(x1, x2)`:两个矩阵的乘积。x1的行数需和x2的列数相同。
+ `@`运算符：x1 @ x2表示两个矩阵乘法。

> 其中一个数据为一维数组时，第一个参数默认被认为是行向量，后一个参数默认被认为是列向量。

### 1.2  内积外积

+ **np.inner(a,b)**：两个数组的内积。
  - a,b为向量：直接求内积
  - a,b一个多维数组，一个为向量：多维数组中每个元素分别与向量求内积
  - a,b均为多维数组，沿最后一维求内积。a为(M,N,L)形状数组，B为(J,K,L)形状数组：结果为(M,N,J,K)形状数组
+ **np.outer(a,b)**：两个数组的外积。

```python
>>> a=np.array([1,2,3])
>>> np.inner(a,a)          #向量点乘
14
>>> b=np.array([[1,1,1],[2,2,2]])
>>> np.inner(a,b)          #多维数组中每个向量与向量做点乘
array([ 6, 12])
>>> a=np.array([[1,2,3],[3,3,3]])
>>> np.inner(a,b)          #沿最后一维求内乘
array([[ 6, 12],
       [ 9, 18]])
```

```python
>>> a=np.array([1,2,3])
>>> np.outer(a, a)
array([[1, 2, 3],
     [2, 4, 6],
     [3, 6, 9]])
```

> $[x_1,x_2,...]$与$[y_1,y_2,...]$的内积为$\sum x_i y_i$。
> 
> $[x_1,x_2,...]$与$[y_1,y_2,...]$的外积为
> 
> $$
> \left[\begin{array}{ccc}
x_1*y_1 & x_2*y_1 & x_3*y_1 \\
x_1*y_2 & x_2*y_2 & x_3*y_2 \\
x_1*y_3 & x_2*y_3 & x_3*y_3
\end{array}\right]
> $$

### 1.3 点积

> 通常我们认为点积就是内积，但是在函数库里，点积积dot运算和内积有一定不同。

+ **np.dot(a,b)**数组点积
  - a,b为向量：直接求内积
  - a,b至少一个未多维数组：求np.matmul(a,b)，即矩阵乘法

```python
>>> a=np.array([1,2,3])
>>> np.dot(a,a)
14

>>> np.dot(b,a)
array([ 6, 12])

>>> b=np.array([[1,2],
... [1,2],
... [1,2]])
>>> np.dot(a,b)
array([[ 6, 12],
       [ 9, 18]])
```

+ **np.vdot(a,b)**向量点积。多维数组会被展开为一维数组后求内积

```python
>>> a=np.array([1,2,3])
>>> b=np.array([[1,2],[3,4]])
>>> np.vdot(a,a)
14
>>> np.vdot(b,b)
30
```

+ **np.tensordot(a,b)**：张量点积
  - `np.tensordot(a,b)`：张量缩并，也称张量缩约，即把a,b的-1,-2维度进行内积运算，缩减2个维度。
  - `np.tensordot(a,b,(0,1))`：沿a的0维，b的1维方向点积运算
  - `np.tensordot(a,b,([0,1],[0,1]))`：a的0维和1维展开为向量，b的0维度和1为展开为向量，做点积

张量缩并(缩约):

两个数组最后的2个维度展开为向量后内积，从而使维度减2。a为M维，b为N维，则输出为max(M,N)-2维。
所以缩并的两个数组都必须不小于2维。

张量缩约

```python
>>> a=np.arange(9).reshape(3,3)
>>> np.tensordot(a,a)
array(204)
>>> b=np.arange(27).reshape(3,3,3)
>>> np.tensordot(a,b)
array([612, 648, 684])
```

张量沿指定轴做点积

```python
>>> a=np.array([[1,2],[1,2]])
>>> np.tensordot(a,a,(1,1))
array([[5, 5],
       [5, 5]])
>>> np.tensordot(a,a,((0,1),(0,1)))
array(10)
```

### 1.4 向量叉乘

+ np.cross(a,b)：计算向量a,b的叉乘。向量长度相同，且只能是2或3。

```python
>>> a
array([1, 2, 3])
>>> np.cross(a,a)
array([0, 0, 0])
```

> a,b中存在多维数组时，通过广播，对对应的向量进行叉乘计算。

### 1.5 张量积

Kronecker积是两个任意大小的矩阵间的运算，表示为⊗，又称为直积或张量积。

$$
A=\left[\begin{array}{cc}
x_11  &  x_12 \\
x_21  & x_22
\end{array}\right]
B=\left[\begin{array}{cc}
y_11  &  y_12 \\
y_21  & y_22
\end{array}\right]
A \otimes B=\left[\begin{array}{cccc}
x_11 *y_11 &  x_11*y_12 & x_11*y_21 & x_11*y_22 \\
x_12 *y_11 &  x_12*y_12 & x_12*y_21 & x_12*y_22 \\
x_21 *y_11 &  x_21*y_12 & x_21*y_21 & x_21*y_22 \\
x_22*y_11 &  x_22*y_12 & x_22*y_21 & x_22*y_22 
\end{array}\right]
$$

+ `np.kron(a,b)`：求a,b的张量积。a形状为(M,N)，b形状为(J,K)，则结果为(M×J,N×K)

```python
>>> a=np.array([[1,2],[1,2]])
>>> b=np.array([3,4])
>>> np.kron(a,b)
array([[3, 4, 6, 8],
       [3, 4, 6, 8]])
```

## 2 分解

### 2.1 Cholesky分解

对正定方阵A，可以分解为三角方阵L与L.T（L的转置）的矩阵乘积。

+ **L=np.linalg.cholesky(a)**：求正定方阵的cholesky分解得到的三角阵
  - a必须是方阵
  - a必须是正定的，否则会出错

```python
>>> a
array([[ 1,  3,  5],
       [ 3, 13, 17],
       [ 5, 17, 42]])
>>> np.linalg.cholesky(a)
array([[1., 0., 0.],
       [3., 2., 0.],
       [5., 1., 4.]])
>>> np.dot(_,_.T)     #np.matmul(_,_.T)
array([[ 1.,  3.,  5.],
       [ 3., 13., 17.],
       [ 5., 17., 42.]])
```

> 判别对称矩阵A的正定性有两种方法：
> 
> + A的所有特征值均为正数，则A是正定的。
> + A的各阶主子式均大于零，则A是正定的。

> 对于半正定矩阵`[[1,2],[1,2]]`不出错，但是分解出来的结果也不太准。

### 2.2 QR(正交三角)分解

M×N的矩阵，M≥N可以分解为：M×M正交矩阵与非奇异上三角矩阵R（M×N）的乘积。

+ **q,r=np.linalg.qr(a)**：求矩阵a的qr分解
  - a为满秩矩阵，且行数M≥列数N
  - q为正交矩阵，即与自身转置的乘积为单位矩阵。
  - r为有正对角元的上三角矩阵

```python
>>> a
array([[ 0,  3,  1],
       [ 0,  4, -2],
       [ 2,  1,  2]])
>>> q,r=np.linalg.qr(a)
>>> q
array([[ 0. , -0.6, -0.8],
       [-0. , -0.8,  0.6],
       [-1. ,  0. ,  0. ]])
>>> r
array([[-2., -1., -2.],
       [ 0., -5.,  1.],
       [ 0.,  0., -2.]])
>>> np.dot(q,r)
array([[ 0.,  3.,  1.],
       [ 0.,  4., -2.],
       [ 2.,  1.,  2.]])
```

> 当M>N时r的最后几行均为0。

> 正交矩阵指，矩阵的转置和矩阵的逆相等

### 2.3 SVD奇异值分解

M×N矩阵被分解为，$USV^*$。U为M×M正交矩阵，V的共轭转置为N×N正交矩阵，R（M×N）的乘积。
S为的M×N对角矩阵（主对角线以外都为0），称为奇异值。

+ **u,s,vt=np.linalg.svd(a)**：求矩阵a的svd分解

```python
>>> a
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11]])
>>> u,s,vt=np.linalg.svd(a)         #奇异值分解
>>> u
array([[-0.1473065 , -0.90090739,  0.40824829],
       [-0.50027528, -0.2881978 , -0.81649658],
       [-0.85324407,  0.32451178,  0.40824829]])
>>> s
array([2.24092982e+01, 1.95534034e+00, 7.68985043e-16])
>>> v
array([[-0.39390139, -0.46087474, -0.5278481 , -0.59482145],
       [ 0.73813393,  0.29596363, -0.14620666, -0.58837696],
       [-0.50775138,  0.52390687,  0.47544042, -0.4915959 ],
       [-0.20539847,  0.65232016, -0.68844492,  0.24152322]])

>>> smat = np.append(np.diag(s),np.zeros((3,1)),axis=1)    #恢复m×n对角阵
>>> smat
array([[2.24092982e+01, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
       [0.00000000e+00, 1.95534034e+00, 0.00000000e+00, 0.00000000e+00],
       [0.00000000e+00, 0.00000000e+00, 7.68985043e-16, 0.00000000e+00]])

>>> (u @ smat @ vt).round(2)                              #核对u,s,vt乘积
array([[ 0.,  1.,  2.,  3.],
       [ 4.,  5.,  6.,  7.],
       [ 8.,  9., 10., 11.]])
```

> 注意np.linalg.svd得到的奇异值为向量。

## 3 特征值

### 3.1 方阵特征值和右特征向量

对于n阶方阵A，如果存在数$\lambda$和非零列向量X，使得

$$
AX=\lambda X
$$

则数λ称为矩阵A特征值，非零向量X称为A的对应于特征值λ的特征向量。

+ **es, vs = np.linalg.eig(a)**：求矩阵a的特征值和特征向量。
  - es为特征值数组。`es[i]`为第i个特征值
  - vs为特征向量。`vs[:,i]`vs的**第i列**，为第i个特征值对应的特征向量

```python
>>> a=np.array([[2,3],[2,1]])
>>> es,vs=np.linalg.eig(a)
>>> es
array([ 4., -1.])
>>> vs
array([[ 0.83205029, -0.70710678],
       [ 0.5547002 ,  0.70710678]])
>>>
>>> a@vs[:,0]
array([3.32820118, 2.21880078])
>>> es[0]*vs[:,0]
array([3.32820118, 2.21880078])
>>>
>>> a@vs[:,1]
array([ 0.70710678, -0.70710678])
>>> es[1]*vs[:,1]
array([ 0.70710678, -0.70710678])
```

> $AX=\lambda X$,也可以写成$(A-\lambda I)X=0$。这是n个未知数n个方程的齐次线性方程组，它有非零解的充分必要条件是系数行列式|A-λE|=0

### 3.2 Hermitian矩阵的特征值和特征向量

厄米特矩阵（Hermitian）是指其共轭转置等于自身的矩阵，对于实数矩阵，厄米特矩阵就是对称矩阵。
典型的厄尔米特矩阵：

$$
\left[\begin{array}{ccc}
    1 & 2+2j & 3-3j \\
    2-2j & 2 & -j \\
    3+3j & j &  5+j
\end{array}\right]
$$

Hermitian矩阵也可以用np.linalg.eig(a)求特征值和特征向量。numpy还提供了一个专用的计算函数
np.linalg.eigh，用法与np.linalg.eig相同。

+ **es, vs = np.linalg.eigh(a)**：求矩阵a的特征值和特征向量。
  - es为特征值数组。`es[i]`为第i个特征值
  - vs为特征向量。`vs[:,i]`vs的**第i列**，为第i个特征值对应的特征向量

```python
>>> a
array([[1, 0, 0],
       [0, 2, 0],
       [0, 0, 3]])
>>> es,vs=np.linalg.eigh(a)  #计算特征值和特征向量

>>> a @ vs[:,0]              #a和第一个特征向量乘积
array([1., 0., 0.])
>>> es[0] * vs[:,0]          #第一个特征值和第一个特征向量乘积
array([1., 0., 0.])
>>>
>>> a @ vs[:,1]              #a和第二个特征向量乘积
array([0., 2., 0.])
>>> es[1] * vs[:,1]          #第二个特征值和第二个特征向量乘积
array([0., 2., 0.])
```

> np.linalg.eigh不能用于求非Hermitian矩阵，比如：`[[2,3],[2,1]]`可以用eig求解，用eigh结果就不对，

### 3.3 方阵特征值（只求特征值）

+ **es = np.linalg.eigvals(a)**：求矩阵a的特征值。
+ **es = np.linalg.eigvalsh(a)**：求Hermitian矩阵的特征值。不能求非Hermitian矩阵特征值。

```python
>>> a
array([[2, 3],
       [2, 1]])
>>> np.linalg.eigvals(a)
array([ 4., -1.])
```

## 4. 范数等

### 4.1 对角矩阵，矩阵的迹

+ `np.diag(a)`：a是矩阵，则返回对角线元素向量。a是向量则返回对角矩阵
+ `np.tri(N,M=None,k=0)`：创建N行M列下三角矩阵，k为偏移量。不指定M则创建方阵。
+ `np.tril(a,k=0)`：对矩阵a，上三角填充0，k为偏移量
+ `np.triu(a,k=0)`：对矩阵a，下三角填充0，k为偏移量
+ `np.trace(a)`：求矩阵的迹，即对角线之和

### 4.2 求方阵的行列式

+ `np.linalg.det(a)`：求a的行列式。a为维度超过2时，最后两维必须为方阵

### 4.3 范数

+ np.linalg.norm(a,order=None)：求矩阵或向量a的范数。范数类型用order指定。
  - order参数：
    + ’fro' or None：Frobenius范数，即L2范数，即所有元素平方和的平方根
    + 'nuc'：核范数，a.T和a乘积的平方根的迹，等于奇异值的和(奇异值分解中s的和)
    + inf：无穷范数。绝对值的最大值。
    + -inf：负无穷范数。绝对值的最小值。

> 矩阵和向量的范数都是大于等于0的数值。

### 4.4 条件数

条件数指a的范数与a的逆的范数的乘积。

+ `np.linalg.cond(a,order=None)`：计算a的条件数。a可以为矩阵或向量。

### 4.5 矩阵的秩

+ `np.linalg.matrix_rank(a)`：使用SVD方法返回数组的矩阵的rank

## 5 逆矩阵和解方程

### 5.1 逆矩阵

+ `np.linalg.inv(a)`:计算方阵的逆，只有满秩矩阵可逆

非方阵或者非满秩矩阵无法计算逆矩阵。对于矩阵A，可以求其伪逆矩阵B，B满足：

ABA==A，BAB==B。B形状与A的转置形状相同

+ `B=np.linalg.pinv(A)`：计算非方阵或者非满秩矩阵的伪逆矩阵。

### 5.2 解方程

**求解线性方程组**

+ `x=np.linalg.solve(A,B)`：求Ax=b的解
  - A必须为满秩方阵，B为向量，和A行数相同

**最小二乘求线性方程系数**

已知x,y两个向量，组成的若干个点（xi,yi)。拟合方程y=wx+b

$$
\left[\begin{array}{}x_1 & 1 \\ x_2 & 1 \\ \vdots & \vdots \\ x_n & 1\end{array}\right]*
\left[\begin{array}{} w \\ b \end{array}\right]=
\left[\begin{array}{}y_1 \\ y_2 \\ \vdots \\ y_n \end{array}\right]
$$

用A表示上式中最左侧`[x,1]`形式的矩阵，z表示系数向量`[w,b]`，则上式变为Az=y。
最小二乘法求`[w,b]`方式入下：

+ `w,b = np.linalg.lstsq(A,y)`：求解线性方程系数

> 用`A = np.vstack([x, np.ones(len(x))]).T` 可以得到系数矩阵A

**求解线性方程组**

+ `x=np.linalg.solve(a,b)`：求解线性方程组ax=b的根。

```python
>>> a = np.array([[1, 2], [3, 5]])
>>> b = np.array([1, 2])
>>> x = np.linalg.solve(a, b)
>>> x
array([-1.,  1.])
```

## 6 多项式

numpy多项式有两个api：

+ np.poly1d：为了保持兼容，新版本仍可以用
+ np.polynomial：重新编写的新API。老版本numpy不支持

### 6.1 旧API：np.poly1d

#### 6.1.1 多项式求解

+ `p=np.poly1d([1,2,3])`：创建多项式类对象，多项式为1x**2+2x+3
    - p.c：多项式的系数，即[1,2,3]
    - p.order：多项式的次数，即2
    - `p(0.5)`：求解x=0.5时多项式的值
    - `p.r`：or p.roots，求多项式等于0的方程的根
    - np.polyval(p,5)：求`x**2+2x+3=5`的根
    - `np.polyval([1,2,3],5)`：求`x**2+2x+3=5`的根

#### 6.1.2 多项式拟合

最小二乘法拟合

+ `np.polyfit(x,y,deg)`：根据(x,y)数据点，拟合deg阶多项式

### 6.2 新API：np.polynomial

#### 6.2.1 多项式对象

+ `p=np.polynomial.Polynomial([1,2,3])`：创建`1+2x+3x**2`多项式。注意和poly1d相反
+ `p=np.polynomial.Chebyshev([1,2,3])`：创建T0(x)+2T1(x)+3T2(x)切比雪夫多项式
+ `p=np.polynomail.Hermite([1,2,3])`：创建H0(x)+2H1(x)+3H2(x)埃尔米特物理学多项式
+ `p=np.polynomail.HermiteE([1,2,3])`：创建He0(x)+2He1(x)+3He2(x)埃尔米特概率学多项式
+ `p=np.polynomail.Laguerre([1,2,3])`：创建L0(x)+2L1(x)+3L2(x)拉盖尔多项式
+ `p=np.polynomail.Legendre([1,2,3])`：创建P0(x)+2P1(x)+3P2(x)勒让德多项式
+ `p.fromroots()`：根据多项式等于0方程的根r，创建`(x-r[0])*(x-r[1])*...*(x-r[n-1])`多项式对象

#### 6.2.2 多项式基本操作

+ p(x)：计算多项式在x处的值
+ p.roots()：计算多项式等于0的方程的根。等于常数的根，可以通过修改多项式的常数项求。
+ p.indentity()：求p(x)=x的根
+ p.deriv()：求导，返回求导后的多项式对象
+ p.integ()：积分，返回积分后的多项式对象，求导的反操作

#### 6.2.3 多项式拟合

+ np.polynomial.polynomial.polyfit(x,y,deg)：根据(x,y)数据点，拟合deg阶多项式，使用最小二乘法

#### 6.2.4 微积分

+ np.polynomial.polynomial.polyder([1,2,3],m=1)：对多项式p进行m次求导。p用数组表示，多项式对象不行
+ np.polynomial.polynomial.polyint(p,m=1,k=0)：对p求m次积分，积分常数为k

#### 6.2.5 多项式数学运算

+ `p1+p2, np.polynomial.polynomial.polyadd(p1,p2)`：多项式和多项式或常数相加
+ `p1-p2, np.polynomial.polynomial.polysub(p1,p2)`：多项式和多项式或常数相减
+ `p1*p2, np.polynomial.polynomial.polymul(p1,p2)`：多项式和多项式或常数相乘
+ `p1**n, np.polynomial.polynomial.polypow(p1,n)`：多项式的n次方，n必须为正整数
+ `np.polynomial.polynomial.polymulx([1,2,3])`：多项式乘以x，参数不能为多项式对象
+ `np.polynomial.polynomial.polydiv(p1,p2)`：多项式和多项式或常数除
    - p1,p2可以是列表，也可以是poly1d对象，不能是Polynomial对象
    - 得到商和余数

<br><br>

> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>