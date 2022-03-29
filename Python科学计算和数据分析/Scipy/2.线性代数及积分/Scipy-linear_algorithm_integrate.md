Scipy系列(二)：线性代数及积分
==============================




[Scipy系列目录](https://blog.csdn.net/hustlei/article/details/123093966)
<hr>

@[TOC]

<hr>


# 一、 概述

<font color=#888>scipy.linalg(scipy线性代数模块)包含numpy.linalg(numpy线性代数模块)中的所有函数，并且还增加了一些高级函数。

Numpy包括

1. 矩阵和向量乘
1. 范数，行列式，秩等
1. 矩阵分解
1. 矩阵特征值和特征向量
5. 逆矩阵和解线性方程组


<font color=#888>scipy线性代数模块可以分为几个部分：

1. 基本功能
2. 解线性方程组
3. 矩阵分解
4. 特征值和特征向量
5. 矩阵函数
6. 矩阵方程求解器

<font color=#888>scipy积分模块可以主要包括：

1. 对给定函数进行积分
2. 对给定样本进行离散积分

# 二、 Scipy线性代数

> Scipy大约有200个线性代数函数，很多函数与Numpy相同，因此这里只简单介绍函数功能。
> Numpy中线性代数模块详细说明见[Numpy系列：线性代数](https://blog.csdn.net/hustlei/article/details/122094984)

## 1 Scipy基本功能

+ scipy.linalg.issymmetric(a)：判断方阵a是否对称矩阵。
+ scipy.linalg.ishermitian(a)：判断方阵a是否hermitian矩阵。
+ scipy.linalg.inv(a)：求方阵a的逆矩阵。和np.linalg.inv(a)结果相同，但是多两个可选参数，较少用，一般可以认为是一样的。
+ scipy.linalg.pinv(b)：求矩阵a的伪逆矩阵。同样比np.linalg.pinv(a)多几个可选参数。
+ scipy.linalg.pinvh(b)：Hermitian矩阵的伪逆。相当于np.linalg.pinv(a,hermitian=True)，但是有更多可选参数。
+ scipy.linalg.norm(c)：求矩阵或向量范数。比np.linalg.norm(a)多一个是否检查a包含无穷大的参数。
+ scipy.linalg.det(a)：求方阵的行列式。比np.linalg.det(a)多两个参数，一个参数表示结果覆盖a，一个参数表示检查是否包含无穷大。
+ scipy.linalg.kron(a,b)：求张量积(克罗内克积)。同np.kron(a,b)
+ scipy.linalg.khatri_rao(a,b):KR积，两个列数相等的矩阵对应列向量的kron积。


**KR积示例**

```python
>> A=np.array([[1,2],[3,4],[5,6]])
>> B=np.array([[2,6],[3,4]])
>> scipy.linalg.khatri_rao(A,B)
 
array([[ 2, 12],
       [ 3,  8],
       [ 6, 24],
       [ 9, 16],
       [10, 36],
       [15, 24]])
```


## 2 解线性方程组

+ scipy.linalg.solve(a,b)：类似np.linalg.solve(a,b)，但是支持更多参数。求解ax=b的根，a为方阵。
+ scipy.linalg.lstsq(a,b)：类似np.linalg.lstsq(a,b)，最小二乘法求解线性方程组的系数。多两个参数，是否重写覆盖参数及检查是否包含无穷大。

同时还支持几个特殊的求解线性方程的函数：

+ solve_banded(l_and_u, ab, b)：求解ax=b，假设a为带状矩阵。ab为带状矩阵，l_and_u为带状矩阵非零的上、下带数。
+ solveh_banded(ab, b)：求解a为hermitan带状矩阵的线性方程ax=b。
+ solve_circulant(c, b)：求解cx=b方程。c为循环矩阵。
+ solve_triangular(a, b)：求解ax=b方程。a为三角方阵。
+ solve_toeplitz(c_or_cr, b)：求解Tx=b方程。T为托普利兹(toeplitz)矩阵。T由c_or_cr确定。

## 3 矩阵分解

+ scipy.linalg.cholesky(a)：Cholesky分解。返回L，a=L×L.T。类似np.linalg.cholesky(a)。
+ scipy.linalg.qr(a)：QR分解。返回q,r，a=q×r。类似np.linalg.qr(a)。
+ scipy.linalg.svd(a)：SVD奇异值分解。返回u,s,vh，a=u×s×vh。类似np.linalg.svd(a)。
+ scipy.linalg.svdvals(a)：求解a的奇异值s。
+ scipy.linalg.lu(a)：LU分解，返回p,l,u，a为(M,N)形状，p为(M,M)矩阵，l为(M,K)下三角矩阵，u为(K,N)上三角或trapezoidal矩阵。


同时scipy还支持更多的矩阵分解，如ldl，polar，cossin等。

## 4 特征值和特征向量

+ scipy.linalg.eig(a):求解方阵的特征值和特征向量，类似np.linalg.eig(a)。
+ scipy.linalg.eigvals(a)：求特征值，类似np.linalg.eigvals(a)。
+ scipy.linalg.eigh(a)：求Hermitian矩阵的特征值和特征向量，类似np.linalg.eigh(a)。
+ scipy.linalg.eigvalsh(a)：求Hermitian矩阵的特征值，类似np.linalg.eigvalsh(a)。

scipy求解特征值和特征向量的函数支持更多的参数。同时scipy还增加了几个函数：

+ scipy.linalg.eig_banded(a_band):求解实对称或复厄米带矩阵特征值和特征向量。
+ scipy.linalg.eigvals_banded(a_band):求解实对称或复厄米带矩阵特征值。
+ scipy.linalg.eigh_tridiagonal(d,e):求解实对称三对角矩阵的特征值和特征向量。
+ scipy.linalg.eigvalsh_tridiagonal(d,e):求解实对称三对角矩阵的特征值。

## 5 矩阵函数

矩阵函数与numpy函数不同，比如np.sqrt(a)表示对a的每一个元素求平方根。矩阵平方根则表示求得的值x与x进行矩阵乘等于a。

> 矩阵A的平方表示A与A的矩阵乘积。
> 矩阵A的n次幂表示n个A依次进行矩阵乘。
> 矩阵A的平方根x表示：x与x的矩阵乘积等于A。
> 矩阵A作为指数，求解e的A次幂，可以用级数分解为A的n的方的函数。即可以通过A的幂求解e的A次幂。
> 同样矩阵A的三角函数也可以用e的iA次幂求解。

+ scipy.linalg.sqrtm(A)：求矩阵A的平方根。
+ scipy.linalg.expm(A)：求e的矩阵A次幂。 
+ scipy.linalg.logm(A)：求A对e的对数。
+ scipy.linalg.sinm(A)：求A的正弦。
+ scipy.linalg.cosm(A)：求A的余弦。
+ scipy.linalg.tanm(A)：求A的正切。
+ scipy.linalg.sinhm(A)：求A的双曲正弦。
+ scipy.linalg.coshm(A)：求A的双曲余弦。
+ scipy.linalg.tanhm(A)：求A的双曲正切。
+ scipy.linalg.signm(A)：求方阵符号。可以参考<https://nhigham.com/2020/12/15/what-is-the-matrix-sign-function/>
+ scipy.linalg.funm(A,fun)：对A执行fun函数。比如funm(A,lambda x:x*x)相当于x与x的矩阵乘。

## 6 矩阵方程求解器(控制理论方程)

> scipy线型模块中提供了5个控制理论中的矩阵方程求解器。

+ scipy.linalg.solve_sylverster(a,b,q)：求解控制理论中的西尔维斯特方程AX+XB=Q。a为M阶方阵，b为N阶方阵，q为(M,N)矩阵。
+ scipy.linalg.solve_continuous_are(a,b,q,r)：求解连续时间代数黎卡提方程（CARE）（最优控制的非线性方程）。
+ scipy.linalg.solve_discrete_are(a,b,q,r)：求解离散时间代数黎卡提方程（CARE）。
+ scipy.linalg.solve_continuous_lyapunov(a,q)：求解控制理论中的连续李雅普诺夫方程。
+ scipy.linalg.solve_discrete_lyapunov(a,q)：求解控制理论中的离散李雅普诺夫方程。

# 三、 Scipy积分

## 1 对给定函数积分

### 1.1 通用积分

`scipy.integrate.quad(func,a,b)`：对函数func从a到b进行积分。

> 返回值为元组，元组第一个值为积分值，第二个值为绝对误差

```python
>>> scipy.integrate.quad(np.sin,0,2*np.pi)
(2.221501482512777e-16, 4.3998892617845996e-14)
```

a,b可以取正负无穷大

```python
>>> scipy.integrate.quad(lambda x:1/x**2,1,np.inf)
(1.0, 1.1102230246251565e-14)
```

func可以为多个参数的函数，用args参数为后边的参数指定值。

```python
>>> scipy.integrate.quad(np.add,1,2,args=(1,))   #函数为x+y，y取值为1，即对x+1进行积分。
(2.5, 2.7755575615628914e-14)
```

发散积分并不会出错，而是会在命令行中返回一个警告消息。

```python
>>> scipy.integrate.quad(lambda x:1/x,0, np.inf)
<input>:1: IntegrationWarning: The maximum number of subdivisions (50) has been achieved.
(48.720960971461565, 16.30167063049395)
  If increasing the limit yields no improvement it is advised to analyze 
  the integrand in order to determine the difficulties.  If the position of a 
  local difficulty can be determined (singularity, discontinuity) one will 
  probably gain from splitting up the interval and calling the integrator 
  on the subranges.  Perhaps a special-purpose integrator should be used.
```

> 有些发散的积分，根据返回结果很难判断是否发散。当然在命令行中还是有警告信息。

```python
scipy.integrate.quad(lambda x:x,0, np.inf)
<input>:1: IntegrationWarning: The integral is probably divergent, or slowly convergent.
(0.4999999961769933, 5.7336234760563265e-06)
```

> 在非命令行中执行时，可以通过指定参数full_output=True，quad函数返回的元组会包含第三、四个数据，详细说明计算结果。

### 1.2 向量函数积分

quad函数积分中func返回值都是单个数组。当func函数为单输入返回向量的函数时，可以用quad_vec函数积分。

比如：``func=lambda x:x*np.array([1,2,3])`。对x积分后也返回一个向量。

`scipy.integrate.quad_vec(func,a,b)`：对向量函数func从a到b进行积分。

```python
>>> scipy.integrate.quad_vec(lambda x:x*np.array([1,2,3]),1,5 )
(array([12., 24., 36.]), 1.495466705179881e-12)
```

> quad_vec函数同样支持args和full_output参数。用法同quad函数。

### 1.3 多重积分

#### 1.3.1 双重积分

`scipy.integrate.dblquad(func, a, b, gfun, hfun, args=())`对函数func(y,x)，对x从a到b积分，对y从gfun到hfun积分。

$$
\int_{gfun}^{hfun}\int_a^b{func(y,x)}dxdy
$$

```python
>>> scipy.integrate.dblquad(lambda y,x:x/y, 1,2,10,20)
(1.0397207708399179, 1.5374249079577676e-14)
>>> scipy.integrate.dblquad(lambda y,x:x/y, 10,20,10,20)
(103.97207708399179, 1.154321938967493e-12)
```

gfun、hfun可以是浮点数，也可以是x的函数，即gfun(x)、hfun(x)。

```python
>>> scipy.integrate.dblquad(lambda y,x:x/y, 1,2,10,lambda a:a*10)
(0.6362943611198906, 1.535015458322731e-14)
```

> args参数用法同quad函数。

#### 1.3.2 多重积分

**三重积分**

```python
scipy.integrate.tplquad(func, a, b, gfun, hfun, qfun, rfun, args=())
```

对函数func(z, y, x)对x从a到b，对y从gfun到hfun，对z从qfun到rfun进行三重积分。

+ 用法同双重积分：
    + gfun, hfun可以为浮点数，也可以为x的函数，即gfun(x), hfun(x)
    + qfun, rfun可以为浮点数，也可以为x,y的函数，即qfun(x,y), rfun(x,y)


```python
>>> from scipy import integrate
>>> f = lambda z, y, x: x*y*z
>>> integrate.tplquad(f, 1, 2, lambda x: 2, lambda x: 3, lambda x, y: 0, lambda x, y: 1)
(1.8750000000000002, 3.324644794257407e-14)
```

**N重积分**

```python
scipy.integrate.nquad(func, ranges, args=None)
```

+ 对函数func(x0, ... xn)进行积分。
+ ranges参数表示n+1个积分变量的上下限
    + ranges每个元素可以是一个由2个数字组成的序列：[(a0,b0),(a1,b1),...(an,bn)]，x0的积分上下限分别为a0,b0，依次类推。
    + ranges每个元素也可以是一个由两个函数组成的序列。

```python
>>> from scipy import integrate
>>> func = lambda x0,x1,x2: x0**2 + x1*x2
>>> integrate.nquad(func, [[0,1], [-1,1], [1,2]])
(0.6666666666666667, 2.5784770663221333e-14)
```

### 1.4 高斯积分法计算积分

这里的高斯积分实际上是指用高斯积分的方式计算积分。

> 一维高斯积分公式为$\int_{-1}^1f(t)dt=ft(t_1)w_1+f(t_2)w_2+...+f(t_n)w_n$，式中wi为权系数。
> 高斯积分的积分区间是`[-1,1]`，普通积分可以转换为高斯积分。
> $\int_a^bf(x)dx=\frac{b-a}2\int_{-1}^1\phi(t)dt$
> 其中：$x=\frac{1}2(b+a)+\frac{1}2(b-a)t$

scipy.integrate中高斯积分包含两个函数：

1. fixed_quad：执行固定阶(即高斯积分公式中的n)高斯求积。
2. quadrature：执行多个阶的高斯正交，直到积分估计的差值低于用户提供的某个公差。

`scipy.integrate.fixed_quad(func, a, b, args=(), n=5)`：用n阶高斯积分法对func从a到b进行积分。

```python
>>> integrate.fixed_quad(np.exp,0,2)
(6.389056096688674, None)
>>> integrate.fixed_quad(np.add,0,2,args=(1,))
(4.0, None)
>>> integrate.quad(np.add,0,2,args=(1,))
(4.0, 4.440892098500626e-14)
```

`scipy.integrate.quadrature(func, a, b, args=())`：通过多个阶的高斯积分正交计算func从a到b的积分。

```python
>>> integrate.quadrature(np.add,0,2,args=(1,))
(4.0, 0.0)
```

### 1.5 Romberg方法计算积分

龙贝格方法是数值计算积分的另一种方法。也称为逐次分半加速法，是在梯形公式、辛普森公式和柯特斯公式之间关系的基础上，构造出一种加速计算积分的方法。 作为一种外推算法，在不增加计算量的前提下提高了误差的精度。

`scipy.integrate.romberg(function, a, b, args=())`：使用Romberg方法对function从a到b进行积分。

```python
>>> integrate.romberg(np.add,0,2,args=(1,))
4.0
```

## 2 对给定样本进行离散积分

### 2.1 梯形法离散积分

+ `scipy.integrate.trapezoid(y, x=None, dx=1.0, axis=-1)`：计算y对x的积分。
    + 如果x=None，则根据dx进行积分。x,y都指定了，dx无效。
    + y为多维数组，则按照axis方向，分别对y的子数据进行积分。

> 类似np.trapz函数。

```python
>>> integrate.trapezoid(np.array([2,2,3]), np.array([0,2,4]))
9.0
>>> integrate.trapezoid(np.array([1,2,3]))
4.0
>>> integrate.trapezoid(np.array([3,3,3]),dx=0.5)
3.0
```

当y为多维数组时：

```
>>> integrate.trapezoid(np.array([[2, 2, 3], [3, 3, 3]]), [1,2,3])
array([4.5, 6. ])
>>> integrate.trapezoid(np.array([[2,2,3],[3,3,3]]))
array([4.5, 6. ])
>>> integrate.trapezoid(np.array([[2,2,3],[3,3,3]]), dx=0.5)
array([2.25, 3.  ])
```

设置axis对多维数组按行积分。

```python
>>> integrate.trapezoid(np.array([[2,2,3],[3,3,3]]), dx=0.5, axis=0)
array([1.25, 1.25, 1.5 ])
```

### 2.2 梯形法累计计算离散积分

+ `scipy.integrate.cumulative_trapezoid(y, x=None, dx=1.0, axis=- 1, initial=None)`：梯形积分，输出每步积分计算结果。
    + y,x,dx,axis参数与trapezoid参数用法相同。
    + initial不为None时，在返回结果的开头插入此值。这样返回值长度就和输入长度相同了。

```python
>>> integrate.cumulative_trapezoid([3,3,3],[1,2,3])
array([3., 6.])
>>> integrate.cumulative_trapezoid([3,3,3,3])
array([3., 6., 9.])
>>> integrate.cumulative_trapezoid([3,3,3,3],initial=0)
array([0., 3., 6., 9.])
```

### 2.3 辛普森法计算离散积分

+ `scipy.integrate.simpson(y, x=None, dx=1.0, axis=- 1, even='avg')`：用辛普森法计算积分
    + y,x,dx,axis参数含义与梯形法相同。
    + even指y数值个数为偶数是处理方法。（辛普森法要求区间数为偶数，即数值点数为奇数）
        + 'avg'：取'first'和'last'的平均值。
        + 'first'：对前N-1个数用simpson法积分，最后的数用梯形法。
        + 'last'：第一个区间用梯形法，后边的数值用simpson法积分。

```python
>>> integrate.simpson([1,2,1,2,1])
6.666666666666666
>>> integrate.simpson([1,2,1,2,1],dx=0.5)
3.333333333333333
>>> integrate.simpson([1,2,1,2,1],[1,2,3,4,5])
6.666666666666666
>>> integrate.simpson([1,2,1,2],even='last')
4.166666666666666
>>> integrate.simpson([1,2,1,2],even='avg')
4.5
```

### 2.4 Romberg法计算离散积分

+ `scipy.integrate.romb(y, dx=1.0, axis=- 1, show=False)`：Romberg 法计算离散积分
    + y为长度为`2**k + 1`个元素的数组。
    + dx,axis参数与梯形法含义相同。
    + show为True则显示计算过程数据。

```python
>>> integrate.romb([1,2,1,2,1])
6.844444444444445
>>> integrate.romb([1,2,1,2,1],dx=0.5)
3.4222222222222225
>>> integrate.romb([1,2,1,2])  #长度不为2*k+1，出错。
...
    raise ValueError("Number of samples must be one plus a "
ValueError: Number of samples must be one plus a non-negative power of 2.
```

<br>
<hr>


[Scipy系列目录](https://blog.csdn.net/hustlei/article/details/123093966)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>  
>  <font color=#888>**未经允许请勿转载。**
