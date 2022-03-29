Scipy系列(三)：统计
==============================


[Scipy系列目录](https://blog.csdn.net/hustlei/article/details/123093966)
<hr>

@[TOC]

<hr>


# 一、 概述

<font color=#888>scipy.stats模块包含大量概率分布、汇总统计、频率统计、相关性函数、统计测试、核密度估计、准蒙特卡罗功能等。

<font color=#888>numpy中没有统计模块，但是实现了不少统计函数功能。主要包括概率分布，顺序统计、均值和方差、相关关系、直方图密度统计。并且功能较少，比如分布就无法计算概率密度。

# 二、 Scipy统计

## 1 概率分布

scipy.stats模块中包含103个连续分布，12个多元分布，19个离散分布。

numpy的random模块中也包含了36个分布函数，包括了常用的离散和连续分布函数。详见[Numpy系列之随机数及概率分布](https://blog.csdn.net/hustlei/article/details/122023562)

**numpy中只能生成随机样本，而scipy.stats模块中则可以计算概率密度等更多数据**。因此，如果只是想生成服从分布的随机样本，用numpy就可以，如果需要更多信息建议用scipy，并且scipy支持更多的分布类型。

### 1.1 scipy.stats离散分布

#### 1.1.1 scipy.stats离散分布函数调用(方法一)

> numpy中的分布函数在random模块中，调用函数可以根据指定分布生成一组样本数据。
> scipy.stats模块中的分布并非函数，而是一个对象，可以调用函数获取随机数、均值、方差等。

以bernoulli伯努利分布为例：

```
scipy.stats.bernoulli.rvs(p,size=10) #设置概率为p，生成size个样本。random_state参数可以设置随机种子。

scipy.stats.bernoulli.pmf(k,p)       #值k的概率质量函数。伯努利
scipy.stats.bernoulli.cdf(k,p)       #累积分布函数
scipy.stats.bernoulli.sf(k,p)        #残存函数或者可靠性函数，1-cdf
scipy.stats.bernoulli.logpmf(k,p)    #pmf的对数
scipy.stats.bernoulli.logcdf(k,p)    #cdf的对数
scipy.stats.bernoulli.logsf(k,p)     #sf的对数

scipy.stats.bernoulli.ppf(q,p)       #百分位函数(累计分布函数的逆函数，根据q=cdf求k）
scipy.stats.bernoulli.isf(q,p)       #生存函数的逆函数(1–cdf 的逆函数)根据q=1-cdf求k。q必须是[0,1)，否则返回-1或nan

scipy.stats.bernoulli.stats(p, moments='mv') #根据moment显示均值Mean('m'),方差Variance('v'),偏度Skw('s'),峰度Kurtosis('k')
scipy.stats.bernoulli.median(p)  #分布的中位数
scipy.stats.bernoulli.mean(p)  #分布的均值
scipy.stats.bernoulli.std(p)  #分布的标准差
scipy.stats.bernoulli.var(p)  #分布的方差

scipy.stats.bernoulli.interval(alpha,p) #分布包含α分位点的数据范围。即变量概率0-α的最小最大数值。alpha取值为`[0,1]`
scipy.stats.bernoulli.entropy(p)  #随机变量的差分熵
scipy.stats.bernoulli.expect(func,args=(p,))  #函数相对于分布的期望，即分布条件下func的条件期望
```

```python
>>> stats.bernoulli.rvs(0.8, size=10)  #值1概率为0.8的伯努利分布，获取10个值的样本
array([0, 1, 1, 1, 1, 1, 1, 1, 1, 1])
>>> stats.bernoulli.rvs(0.8, size=10, random_state=2022)   #设置随机种子，无论运行几次，结果都一样。
array([1, 1, 1, 1, 1, 1, 0, 1, 0, 1])

>>> stats.bernoulli.pmf(0,0.8)  #概率为0.8的伯努利分布，求值0的概率质量函数
0.19999999999999996
>>> stats.bernoulli.pmf([0,1],0.8)  #求0,1值的概率，即同时求多个值的概率质量函数
array([0.2, 0.8])
>>> stats.bernoulli.cdf(1,0.8)  #概率为0.8的伯努利分布，求小于等于1的值的概率总和(累积)
1.0

>>> stats.bernoulli.ppf([0.18,0.2,1],0.8)  #根据cdf值为0.18,0.2,1，求对应位置的变量数值。
array([0., 1., 1.])

>>> stats.bernoulli.stats(0.8)  #计算均值和方差
(array(0.8), array(0.16))
>>> stats.bernoulli.mean(0.8)  #计算分布均值
0.8
>>> stats.bernoulli.median(0.8)  #计算分布分位数
1.0
>>> stats.bernoulli.std(0.8)  #计算标准差
0.39999999999999997
>>> stats.bernoulli.var(0.8)  #计算方差
0.15999999999999998

>>> stats.bernoulli.interval(0.5,0.2)  #概率0-0.5之间的值区间为0-0
(0.0, 0.0)
>>> stats.bernoulli.interval(0.8,0.2)  #概率0-0.8之间的值区间为0-1
(0.0, 1.0)
```

其他函数用法类似，伯努利分布控制参数只有一个概率p，其他分布控制参数数量不同时，在参数p位置替换分布控制参数即可。

比如二项分布binom控制参数有n和p两个，那么：其调用方式为：

```python
scipy.stats.bernoulli.rvs(n,p,size=10) #设置概率为p，生成size个样本。random_state参数可以设置随机种子。

scipy.stats.bernoulli.pmf(k,n,p)       #值k的概率质量函数。伯努利
scipy.stats.bernoulli.cdf(k,n,p)       #累积分布函数
scipy.stats.bernoulli.sf(k,n,p)        #残存函数或者可靠性函数，1-cdf
scipy.stats.bernoulli.logpmf(k,n,p)    #pmf的对数
scipy.stats.bernoulli.logcdf(k,n,p)    #cdf的对数
scipy.stats.bernoulli.logsf(k,n,p)     #sf的对数

scipy.stats.bernoulli.ppf(q,n,p)       #百分位函数(累计分布函数的逆函数，根据q=cdf求k）
scipy.stats.bernoulli.isf(q,n,p)       #生存函数的逆函数(1–cdf 的逆函数)根据q=1-cdf求k。q必须是[0,1)，否则返回-1或nan

scipy.stats.bernoulli.stats(n,p, moments='mv') #根据moment显示均值Mean('m'),方差Variance('v'),偏度Skw('s'),峰度Kurtosis('k')
scipy.stats.bernoulli.median(n,p)  #分布的中位数
scipy.stats.bernoulli.mean(n,p)  #分布的均值
scipy.stats.bernoulli.std(n,p)  #分布的标准差
scipy.stats.bernoulli.var(n,p)  #分布的方差

scipy.stats.bernoulli.interval(alpha,n,p) 
scipy.stats.bernoulli.expect(func,args=(n,p))
```

#### 1.1.2 scipy.stats离散分布函数调用(方法二)

> 从方法一中可以看出，每个函数中都包含控制参数。不同分布控制参数数量不同，但函数调用方法还是相同的。

在scipy.stats中，可以先定下参数，创建分布对象，然后调用函数，可以省略控制参数。以伯努利和二项式分布为例

```python
distrib_bn = stats.bernoulli(0.8)     #创建值1概论为0.8的伯努利分布对象bn
distrib_bi = stats.binom(9,0.6)   #创建n=9,p=0.6的二项分布。即9次独立试验，每次试验成功概论为0.6，试验成功次数的概率分布。
```

所有离散分布对象，都可以直接调用如下函数：

```python
distrib_obj.rvs(size=10) #生成size个样本。random_state参数可以设置随机种子。

distrib_obj.pmf(k)       #值k的概率质量函数。伯努利
distrib_obj.cdf(k)       #累积分布函数
distrib_obj.sf(k)        #残存函数或者可靠性函数，1-cdf
distrib_obj.logpmf(k)    #pmf的对数
distrib_obj.logcdf(k)    #cdf的对数
distrib_obj.logsf(k)     #sf的对数

distrib_obj.ppf(q)       #百分位函数(累计分布函数的逆函数，根据q=cdf求k）
distrib_obj.isf(q)       #生存函数的逆函数(1–cdf 的逆函数)根据q=1-cdf求k。q必须是[0,1)，否则返回-1或nan

distrib_obj.stats(moments='mv') #根据moment显示均值Mean('m'),方差Variance('v'),偏度Skw('s'),峰度Kurtosis('k')
distrib_obj.median()  #分布的中位数
distrib_obj.mean()  #分布的均值
distrib_obj.std()  #分布的标准差
distrib_obj.var()  #分布的方差

distrib_obj.interval(alpha) 
distrib_obj.expect(func)
```

以具体对象为例：

```python
>>> distrib_bn.rvs(10)
array([1, 0, 1, 1, 1, 1, 1, 1, 0, 0])
>>> distrib_bn.pmf(0)
0.19999999999999996
>>> distrib_bn.cdf(0)
0.19999999999999996
>>> distrib_bn.ppf(0.5)
1.0

>>> distrib_bn.stats()
(array(0.8), array(0.16))
>>> distrib_bn.median()
1.0
>>> distrib_bn.mean()
0.8
>>> distrib_bn.std()
0.39999999999999997
>>> distrib_bn.var()
0.15999999999999998
```

#### 1.1.3 常见离散分布

常见离散分布如下（常见离散分布详细含义可以参考[numpy系列之随机数及概率分布](https://blog.csdn.net/hustlei/article/details/122023562)）：

+ randint：均布离散随机值。控制参数low,high。
+ bernoulli：伯努利分布。控制参数：p。其中：0≤p≤1。
+ binom：二项分布。控制参数：n,p。其中：n为自然数，0≤p≤1。
+ geom：几何分布。控制参数：p。其中：0≤p≤1。
+ poisson：泊松分布。控制参数：mu。其中：mu≥0。
+ dlaplace：离散拉普拉斯分布。控制参数：a。其中：a>0。

上述控制参数指确定分布的参数。比如伯努利分布由概率p确定；二项分布由试验次数n和概率p确定。



### 1.2 scipy.stats连续分布

> scipy.stats中连续分布用法与离散分布用法相似，可以通过两种方法调用函数。

以t分布为例（其他分布类似，控制参数替换t分布的df参数即可）：

```python
stats.t.rvs(df,size=10)  #自由度参数为df的t分布，计算size个随机样本。参数random_state可以设置随机种子

stats.t.pdf(x,df)       #x值的概率。
stats.t.cdf(x,df)       #x值的累积分布概率。
stats.t.sf(x,df)         #x值的残存函数或者可靠性函数，1-cdf
stats.t.logpmf(x,df)    #pmf的对数
stats.t.logcdf(x,df)    #cdf的对数
stats.t.logsf(x,df)     #sf的对数

stats.t.ppf(q,df)       #百分位函数(累计分布函数的逆函数，根据q=cdf求k）
stats.t.isf(q,df)       #生存函数的逆函数(1–cdf 的逆函数)根据q=1-cdf求k。q必须是[0,1)，否则返回-1或nan

stats.t.stats(df, moments='mv') #根据moment显示均值Mean('m'),方差Variance('v'),偏度Skw('s'),峰度Kurtosis('k')
stats.t.median(df)  #分布的中位数
stats.t.mean(df)  #分布的均值
stats.t.std(df)  #分布的标准差
stats.t.var(df)  #分布的方差

stats.t.fit(data)  #假设数据data符合t分布，拟合计算出df, loc, scale

stats.t.interval(alpha,df)  #分布包含α分位点的数据范围。即变量概率0-α的最小最大数值。alpha取值为`[0,1]`
stats.t.entropy(df)  #随机变量的差分熵
stats.t.expect(func,args=(df,),lb=None,ub=None) #函数相对于分布的期望，即分布条件下func的条件期望
```

> 上述所有函数都支持loc和scale参数，用于移动和缩放分布概率。对于t分布概率密度变为(x-loc)/scale。

同样连续分布也可以先设置分布参数创建分布对象，然后调用函数。但是要注意：

+ fit函数除外，因为fit函数不需要t分布参数。
+ loc和scale参数也是在创建分布对象时设置，用分布对象调用函数时，函数参数不能再设置loc和scale参数。

```python
distrib_t = stats.t(1)  #创建t分布对象，自由度为1
# distrib_t = stats.t(1,loc=0,scale=1)  #创建t分布对象，自由度为1，位置在loc，缩放系数为scale

distrib_t.rvs(size=10)  #计算size个随机样本。参数random_state可以设置随机种子

distrib_t.pdf(x)       #x值的概率。
distrib_t.cdf(x)       #x值的累积分布概率。
distrib_t.sf(x)         #x值的残存函数或者可靠性函数，1-cdf
distrib_t.logpdf(x)    #pmf的对数
distrib_t.logcdf(x)    #cdf的对数
distrib_t.logsf(x)     #sf的对数

distrib_t.ppf(q)       #百分位函数(累计分布函数的逆函数，根据q=cdf求k）
distrib_t.isf(q)       #生存函数的逆函数(1–cdf 的逆函数)根据q=1-cdf求k。q必须是[0,1)，否则返回-1或nan

distrib_t.stats(moments='mv') #根据moment显示均值Mean('m'),方差Variance('v'),偏度Skw('s'),峰度Kurtosis('k')
distrib_t.median()  #分布的中位数
distrib_t.mean()  #分布的均值
distrib_t.std()  #分布的标准差
distrib_t.var()  #分布的方差

distrib_t.interval(alpha)  #分布包含α分位点的数据范围。即变量概率0-α的最小最大数值。alpha取值为`[0,1]`
distrib_t.entropy()  #随机变量的差分熵
distrib_t.expect(func,lb=None,ub=None) #函数相对于分布的期望，即分布条件下func的条件期望
```

> 其他分布类似。

常见连续分布如下（常见连续分布详细含义可以参考[numpy系列之随机数及概率分布](https://blog.csdn.net/hustlei/article/details/122023562)）：

+ 连续分布
    + uniform：均匀分布。在[0,1]范围内的均匀分布。loc和scale移动分布到[loc, loc+scale]范围。
    + expon：指数分布。指数分布的λ参数相当于scale的倒数，所以scipy中expon除了loc和scale没有其他参数。
    + norm：正态分布。位置在loc，缩放系数为scale的正态分布。
    + lognorm：对数正态分布。控制参数s，s>0。
    + gamma：伽玛分布。控制参数a，a>0。
    + rayleigh：瑞利分布。无控制参数。瑞利分布变量x不小于0。
    + beta：贝塔分布。控制参数a,b。a>0,b>0。
    + laplace：拉普拉斯分布。无控制参数。
+ 三大抽样分布
    + chi2：卡方分布。控制参数df。
    + f：F分布。控制参数dfn, dfd。
    + t：t分布。控制参数：自由度df。

> 所有分布都支持loc, scale参数，通常两个参数的含义为：pdf(x)变换pdf((x-loc)/scale))/scale计算概率密度。
> loc会导致概率密度曲线中心移动到loc，曲线高度按1/scale缩放，宽度按scale缩放(scale大于1时，曲线会变矮变宽)。



scipy.stats概率密度分布曲线示例

```python
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x=np.arange(-10,10,0.1)
y1=stats.laplace.pdf(x)
plt.plot(x,y1,label="laplace")
y2=stats.laplace(scale=3).pdf(x)
plt.plot(x,y2,label="scale=3")
y3=stats.laplace(loc=3).pdf(x)
plt.plot(x,y3,label="loc=3")

plt.legend()
plt.show()
```

![distribution_loc_scale]()

### 1.3 scipy.stats多元分布

多元分布，即多变量分布，指变量x为向量(x1,x2,...xn)的概率。



> 对于x向量中的每一个变量可以认为是一个维度。比如x=(x1,x2),可以认为是平面上的横轴和纵轴。详细可以参考<https://www.cnblogs.com/yanshw/p/11898408.html>

在scipy中，x参数的向量长度根据函数x的最后一维长度确定。比如x=(x1,x2)则。x参数的数组形状应该是(...,2)

以多元正态分布为例：

```
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x0,y0=np.mgrid[-5:5:0.5,-5:5:0.5]   #x0,y0的shape都是(20,20)
x=np.dstack([x0,y0])  #x的shape为(20,20,2)
y=stats.multivariate_normal.pdf(x,[0,0], [[2,1], [1,2]])  #概率密度为数组y的形状为(20,20)

plt.figure().add_subplot(projection="3d").plot_wireframe(x0,y0,y)
plt.show()
```

![多元正态分布]()

```
>>> x=np.array([[0.05,0.01],[0.1,0.15],[0.5,0.2]])        #二元分布，x可以是(M,N,2)形状，(M,2)形状数组也可以
>>> stats.multivariate_t.pdf(x,shape=np.eye(2,2))
[0.15853625 0.15169981 0.10862644]
```


同样多元分布也可以先创建指定参数的对象，然后调用函数。

```
#不同的多元分布，参数不同。多元分布元数量不同时，参数长度也可能不同
distrib_multi=stats.multivariate_normal([0,0], [[2,1], [1,2]]) #参数指定2元正态分布的均值，和相关系数矩阵(2×2对阵半正定方阵)
distrib_multi.pdf(x)  #计算x(对于二元分布，x应为(...,2)形状数组)的概率密度
```

常见的多元分布：

+ multivariate_normal：多元正态分布。必须控制参数：均值（长度为n的向量），相关系数矩阵（n×n形状对称半正定方阵）
+ multivariate_t：多元t分布。必须的控制参数：shape(半正定n×n方阵)。参数loc默认为0，可以为每个方向分别设置，df默认为1。
+ dirichlet：狄利克雷分布。dirichlet分布的x参数比较特殊，后边单独介绍。


多元分布可用函数：

+ `rvs(size=1,random_state=None)`：随机数样本
+ `pdf(x)`：概率密度函数。
+ `logpdf(x)`：pdf的对数。
+ `cdf(x)`：累积密度函数。
+ `logcdf(x)`：cdf的对数。
+ `entropy()`：多元微分熵。

> 分布的控制参数可以在函数的x参数之后(注意entropy函数无参数)。或者在拆改那就分布对象时设置参数，在调用函数时不使用控制参数。可参考前述离散分布和连续分布。

多元分布，n元随机变量就会有n个值：

```python
>>> stats.dirichlet.rvs([1])  #一元分布，获取一个随机变量
Out[4]: array([[1.]])

>>> stats.dirichlet.rvs([1,1])  #二元分布，获取一个随机变量
Out[5]: array([[0.54441826, 0.45558174]])

>>> stats.dirichlet.rvs([1,1,2])  #三元分布，获取一个随机变量
Out[6]: array([[0.1583077 , 0.03149985, 0.81019245]])

>>> stats.dirichlet.rvs([1,1,2,3])  #四元分布，获取一个随机变量
Out[7]: array([[0.23294627, 0.30891125, 0.29961083, 0.15853165]])

```

> dirichlet分布相对特殊，与其他多元分布不同，他的x第0维长度和元数相同，而不是最后一个维度长度与元数相同(其他分布是这样的)。
> 并且，从上边示例可以看出，每个dirichlet随机数的所有分量之和都等于1。

dirichlet分布的控制参数有：alpha。
    
+ dirichlet的alpha必须＞0。且alpha只能是一维数组。数组长度表示分布元数。
+ dirichlet变量的特点，也是函数中x的特点
    + dirichlet的x中的每个数值都必须是0-1之间的。如果对应alpha小于1，值还必须大于0（即不能等于0）。
    + x的第0维长度需要和alpha长度相同，或者比alpha长度小1。x.shape[0]==alpha.shape[0]或x.shape[0]=alpha.shape[0]-1
    + dirichlet的x沿第0维相加，必须为1。np.sum(x, 0)的所有元素值都是1。

```
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

distrib_d = stats.dirichlet(alpha=[2,3])  #二元迪利克雷分布
x0=np.arange(0,1,0.01)
y0=1-x0
x=np.array([x0,y0])  #两行，两行相加所有元素都是1

y=distrib_d.pdf(x)

plt.figure().add_subplot(projection="3d").plot(x0,y0,y)
plt.show()
```

![迪利克雷分布]()



## 2. 汇总统计

### 2.1 基本统计量

很多基本统计量，在numpy、pandas等都出现过，但是在scipy中略有不同。

+ stats.tmean(a,limits=None,axis=None)：沿着axis轴计算a的截尾平均值。设置limits=(lower,upper)时，忽略超出limits的值，否则计算所有值的均值。axis=None忽略轴，对所有数据计算均值。
+ stats.tvar(a)：计算方差。设置limits参数时计算截尾方差。axis参数指定轴。ddof设置自由度，默认为1，计算方差时，除以n-ddof取代除以n，从而对样本统计，得到无偏方差。
+ stats.tstd(a)：计算标准差。设置limits参数时计算截尾标准差。axis参数指定轴。ddof设置自由度。
+ stats.tsem(a)：计算标准误。支持limits、axis、ddof参数。标准误指多次抽样得到的样本的均值的标准差，是描述对应的抽样分布的离散程度及衡量对应样本统计量抽样误差大小的尺度。因为只有一个样本，很难计算，通常用方差除以N，再开方计算。
+ stats.tmin(a,lowerlimit=None,axis=None)：计算截尾最小值(忽略比lowerlimit更小的值)。沿axis计算，默认。
+ stats.tmax(a,upperlimit=None,axis=None)：计算截尾最大值。
+ stats.mode(a)：计算众数，支持axis参数，默认对axis=0计算众数。对于多个数出现次数并列最大时，指返回第一个数。
+ stats.moment(a,moment=1,axis=0)：计算n阶中心矩（元素与均值的差的n次方和，除以n）。一阶矩恒为0，二阶是方差，三阶是偏度(非标准化偏度)，四阶是峰度(非标准化峰度)。
+ stats.skew(a,axis=0,bias=True)：计算偏度(三阶标准化矩)。bias=False则进一步矫正计算结果乘以一个系数后得到Fisher-Pearson标准化矩系数。偏度表示分布偏斜方向和程度，指出数据非对称程度，0表示对称，小于0表示均值偏左，即分布偏右，反之亦然。
+ stats.kurtosis(a,axis=0,fisher=True,bias=True):计算峰度（四阶中心矩除以方差的平方减3）。fisher=False则不减3。bias=False则进行修正。
+ stats.describe(a,axis=0,ddof=1,bias=True)：计算最小、最大，均值，方差，偏度，峰度。

> + **截尾算法**：就是去掉极值后计算。计算平均分时，去掉最高分和最低分就是截尾均值。
> + **ddof**：自由度默认为1。当ddof为0时，计算方差时除以n，即总体方差，对于样本属于有偏方差。ddof为1时，除以n-1，即对样本计算无偏方差。
> + 众数mode函数的axis参数默认值为0，其他函数axis参数默认值为None。
> + skew计算的是三阶标准化中心矩。即三阶矩与二阶矩的1.5次方之比（三阶矩与标准差之比的三次方）。bias=False的时候，结果乘以(N*(N-1))**0.5/(N-2)
> + kurtosis中减3可以让正态分布峰度为0。服从正态分布的数据峰度是3。峰度值为1~∞，值越大分布曲线约瘦高。

```python
>>> a
array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> stats.tmean(a)    #计算均值
4.5
>>> stats.tmean(a,limits=(1,None))   #排除0，计算均值
5.0
>>> stats.tvar(a)   #计算方差
9.166666666666666
>>> stats.tstd(a)   #计算标准差
3.0276503540974917
>>> stats.tstd(a,ddof=0)  #计算总体(样本有偏)标准差。即计算时，除以N而不是N-1。
2.8722813232690143
>>> stats.tsem(a)   #计算标准误，与标准差除以N的平方根相同。
0.9574271077563381

>>> a=np.array([1,1,2,2,2,3,3,3,4,5,6])
>>> stats.mode(a)
ModeResult(mode=array([2]), count=array([3]))


>>> a
array([1, 1, 2, 2, 2, 3, 3, 3, 4, 5, 6])

>>> stats.skew(a)
0.6349896455441276

>>> stats.kurtosis(a)
-0.5105493100324998

>>> a=np.random.normal(size=100)
>>> stats.kurtosis(a)
0.0985068411502068

>>> stats.describe(a)
DescribeResult(nobs=100, minmax=(-2.2358216859781384, 2.2330434331979183), mean=-0.06105246142953563, variance=0.7795496121777539, skewness=-0.14965830081799855, kurtosis=0.0985068411502068)
```

+ stats.find_repeats(arr)：找到重复值。

```
>>> stats.find_repeats([2, 1, 2, 3, 2, 2, 5])
RepeatedResults(values=array([2.]), counts=array([4]))
>>> stats.find_repeats([[10, 20, 1, 2], [5, 5, 4, 4]])
RepeatedResults(values=array([4.,  5.]), counts=array([2, 2]))
```



### 2.2 部分基本统计量另一种算法函数

+ stats.gmean(a)：几何均数，然后变量值连乘开项数次方根。注意有数值等于0或小于0时，计算可能会出错，或者得到nan。支持axis参数。
+ stats.hmean(a)：调和均数，倒数的平均值的倒数。数值都必须≥0。支持axis参数。
+ stats.trim_mean(a,proportiontocut):百分比截尾均值。proportiontocut指最小和最大均要截掉的比例，取值0-1。支持axis参数。
+ stats.gstd(a)：几何标准差，exp(std(log(a)))。支持axis和ddof。
+ stats.sem(a,axis=0,ddof=1)：计算标准误(无截尾参数)。



```python
>>> a
array([1, 2, 3, 4, 5, 6, 7, 8, 9])
>>> stats.gmean(a)
4.147166274396913
>>> stats.hmean(a)
3.181371861411138
>>> stats.trim_mean(a,0.1)
5.0

>>> stats.gstd(a)
2.052917619389798
>>> stats.sem(a)
0.9128709291752769
```


### 2.3 复杂统计量

+ stats.iqr(x, axis=None, rng=(25, 75):计算四分位数间距。即75%分位数25%分位数的范围。
+ stats.median_abs_deviation(x,axis=0):绝对中位差。median(|Xi-median(X)|)
+ stats.variation(a):变异系数。比较两组数据离散程度的量，是标准差与均值的比值。
+ stats.bayes_mvs(a)：均值、方差、标准差的贝叶斯置信区间。
+ stats.kstat(data,n=2):n阶累积量kappa_n的唯一对称无偏估计量。
+ stats.kstatvar(data,n=2):返回kstat方差的无偏估计值。
+ stats.bootstrap(data,statistic,confidence_level=0.95)：计算数据的双侧重抽样置信区间。对数据进行重采样，第每个采样计算测试统计数据，确定置信区间。data的每个元素都是服从分布的一组数据，即data至少是2维数据。statistic是要计算置信区间的统计信息的函数。默认置信区间为0.95。


## 3. 频率统计

+ stats.cumfreq(a,numbins=10):累积柱状图，类似直方图，但是返回值中的cumcount为累积值，而不是直方图中的count。
+ stats.relfreq(a,numbins=10)：直方图，相对频率统计。
+ stats.binned_statsitic(x,values,statistic='mean',bins=10):类似直方图，把x分为10个区间，计算values的mean。x,value形状必须相同。
+ stats.binned_statsitic_2d(x,y,values):二维版binned_statistic。
+ stats.binned_statsitic_dd(sample,values):多维版binned_statistic。
+ stats.percentileofscore(a,score):计算score在a中的百分位排名。
+ stats.scoreatpercentile(a,per):根据百分位排名计算score值。

```
>>> a=np.random.normal(size=100)
>>> stats.cumfreq(a)
CumfreqResult(cumcount=array([  2.,  11.,  23.,  42.,  65.,  84.,  90.,  98.,  99., 100.]), lowerlimit=-2.6127861617102033, binsize=0.5975561727163177, extrapoints=0)
>>> a=[0,1,2,3,3,3,4]
>>> stats.relfreq(a,5)
RelfreqResult(frequency=array([0.14285714, 0.14285714, 0.14285714, 0.42857143, 0.14285714]), lowerlimit=-0.5, binsize=1.0, extrapoints=0)

>>> x=np.arange(100)
>>> y=np.random.normal(size=100)
>>> stats.binned_statistic(x,y,bins=5)
BinnedStatisticResult(statistic=array([-0.29320732,  0.15116167, -0.03095417, -0.07738226, -0.09060843]),
                     bin_edges=array([ 0. , 19.8, 39.6, 59.4, 79.2, 99. ]), binnumber=...)

>>> a=[0,1,2,3,4]
>>> stats.percentileofscore(a,3)
80.0
```


## 4. 相关性函数

+ pearsonr(x, y)：皮尔逊相关系数和p值用于检验非相关性。
+ spearmanr(a[, b, axis, nan_policy, alternative])：用相关的p值计算斯皮尔曼相关系数。
+ pointbiserialr(x, y)：计算点双列相关系数及其p值。
+ kendalltau(x, y[, initial_lexsort, …])：计算Kendall's tau，这是顺序数据的相关度量。
+ weightedtau(x, y[, rank, weigher, additive])：计算肯德尔的加权版本。
+ somersd(x[, y, alternative])：计算Somers’D，这是顺序关联的非对称度量。
+ linregress(x[, y, alternative])：计算两组测量值的线性最小二乘回归。
+ siegelslopes(y[, x, method])：计算一组点（x，y）的西格尔估计量。
+ theilslopes(y[, x, alpha, method])：计算一组点（x，y）的泰尔森估计量。
+ multiscale_graphcorr(x, y[, …])：计算多尺度图相关性（MGC）测试统计量。




## 5. 统计测试


## 6. 核密度估计

scipy.stats中的gaussian_kde函数可以用于一元或多元核密度估计。高斯核密度估计方法就是把样本根据带宽h分割为n个区间，每个区间用一个高斯概率密度函数表示概率，然后叠加得到高斯核密度曲线。
带宽越小核密度估计曲线越接近直方图，波动越大。带宽越大，核密度曲线越光滑。

+ kde_obj=stats.gaussian_kde(data,bw_method=None)：创建data的核密度估计曲线对象。
	+ bw_method指用于确定带宽的方法，默认是'scott'，可以是'silverman'，也可以是数值或者函数。
		+ scott方法指带宽为n**(-1/(d+4))。n为数据点数，d为数据维度数。
		+ 数值或函数用于指定带宽值。

stats.gaussian_kde()函数返回值是一个对象




```python

```




> 实际上核密度估计，除了高斯核，还可以用tophat，epanechnikow，exponential，linear，cosine等。

## 7. 准蒙特卡罗功能


<br>
<hr>


[Scipy系列目录](https://blog.csdn.net/hustlei/article/details/123093966)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>  
>  <font color=#888>**未经允许请勿转载。**
