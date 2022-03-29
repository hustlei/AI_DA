Pandas系列(五)：可视化绘图
==============================



@[TOC]

<hr>

# 一、 简介

<font color=#888>Pandas数据提供了方便的绘图函数，可以直接调用绘图。不仅包括常用的点、线、柱形图、饼图，还包括直方图、核密度曲线图等统计图形。

> Pandas的绘图函数基于Matplotlib。

> <font color=#999>Pandas系列将Pandas的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![Pandas数据可视化](https://img-blog.csdnimg.cn/861041e7be3942eb9bb63c8ef9877dd9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



<br>

# 三、 Pandas可视化绘图

## 1. Pandas绘图基本方法

Pandas数据可视化有两种途径

方法一：通过数据（Series或DataFrame）的plot属性调用绘图函数。

```
# 导入库
import pandas as pd
import matplotlib.pyplot as plt

# 初始化数据
d=pd.DataFrame({'x':[-3,-2,-1,0,1,2,3]})
d['y']=d['x']**2

# 绘图
d.plot.line(x='x', y='y')      #调用plot模块的line方法绘制折线图

# 显示
plt.show()
```

显示图形结果如下：

![pandas plot intro](https://img-blog.csdnimg.cn/17822122f66d42abbbfb2bef852d4c2f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


方法二：直接调用plot函数


```
# 导入库
import pandas as pd
import matplotlib.pyplot as plt

# 初始化数据
d=pd.DataFrame({'x':[-3,-2,-1,0,1,2,3]})
d['y']=d['x']**2

# 绘图
d.plot(x='x', y='y', kind="line")      #调用plot函数绘图，指定图形类别为折线图

# 显示
plt.show()
```

> 效果与方法一完全相同。

## 2. plot模块

### 2.1 简单图表

#### 2.1.1 绘图函数

+ df.plot.scatter(x='col1',y='col2')：散点图（Series不支持）。
+ d.plot.line(x='col1',y='col2')：折线图。
+ d.plot.area(x='col1',y='col2')：颜色填充的折线图。
+ d.plot.bar(x='col1',y='col2')：柱状图。
+ d.plot.barh(x='col1',y='col2')：水平柱状图。
+ d.plot.pie(y='col2')：饼状图。只需要指定y值。


```python
np.random.seed(20211231)
x=np.arange(5)
y=np.random.randint(10,size=5)
d=pd.DataFrame({'x':x,'y':y})

d.plot.scatter(x='x',y='y')
d.plot.line(x='x',y='y')
d.plot.area(x='x',y='y')
d.plot.bar(x='x',y='y')
d.plot.barh(x='x',y='y')
d.plot.pie(y='y') 
plt.show()
```

显示效果如下：

![Pandas 简单绘图效果](https://img-blog.csdnimg.cn/4b24bc3c5cca45319ad633199aa8ed80.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)




#### 2.1.2 进阶用法

**1）Series数据绘图**

> Series不支持scatter函数，即不能用Series绘制散点图。英文散点图必须有x,y坐标。

Series数据调用绘图函数时，不用指定x,y数据，默认把Series数据作为y坐标数据，用自然数`[0,1,2,...]`作为x坐标数据

示例：

```python
s=pd.Series([1,2,3,4,5])

s.plot.line()
s.plot.area()
s.plot.bar()
s.plot.barh()
s.plot.pie()
plt.show()
```

![Series绘图效果](https://img-blog.csdnimg.cn/76e377bb5eb246f4940ae0fca80b9451.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



**2) scatter函数特殊性**

> scatter函数必须同时指定x,y坐标。且x,y坐标长度必须相同 



+ `df.plot(x=['col1','col2'], y=['col3','col4'])`：多个数据列绘制散点图
    - (col1,col3)和(col2,col4)分别作为x,y坐标绘制散点图。
    - x,y必须都是列表，且列表长度必须相同

> 仅scatter和hexbin函数支持这样为x,y参数赋值，其他绘图函数都不支持这样使用。
> 多个数据列绘制的散点图，仍是同一个散点序列。并不能像折线图一样，两个数据列得到两根线。
>> scatter绘制多列数据与其他函数都不同。与线条等图形相比，不同的数据列表示不同的线条，但是散点图没有这个概念，所有的散点都是同一个序列的。所以，同一个图形中可以绘制多条曲线，但是在散点图中不论多少个数据列，都合并为一个散点序列。

scatter函数支持的所有数据参数调用方式：

```
df.plot.scatter(x='col1', y='col2')
df.plot.scatter(x=['col1','col1'], y=['col2','col3'])
```



**2) pie函数特殊性**

> pie只需要指定y参数，不需要指定x参数。饼图的标签采用数据的行标签。
 
+ df.plot.pie(subplots=True)：每个列绘制一个饼图
	+ 必须设置subplots为True，这样多个饼图会分别绘制在不同的子图中。
	+ layout=(2,3)参数可以设置子图排列为2行3列(数量必须大于df列数)。
	+ y不能赋值为列表。即`y=['col1','col2']会出错`

```python
df=pd.DataFrame({'col1':[1,2,3,4],'col2':[2,3,2,4],'col3':[3,1,2,1.5]},index=['a','b','c','d'])
df.plot.pie(subplots=True,layout=(2,2))
```

![Pandas pie绘图效果](https://img-blog.csdnimg.cn/d3997da20c154d6e8c6dfef0bf56df34.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



pie函数支持的所有数据参数调用方式：

```
df.plot.pie(y='col1')
df.plot.pie(subplots=True) 
```

**3）line、area、bar、barh共同特点**

这几个函数x,y参数都支持这几种用法：

+ 单个数据序列图：x,y都指定一个列。
+ 多个数据序列图：x指定一个列，y值设置为**列表**。

> 注意x参数不能是列表。
> + x，y均可以省略
>   - x默认值为0,1,2...自然是序列
>   - y默认值为除x外，其他所有列组成的列表
>> x,y都省略时，对每列数据绘图

+ 单个数据序列图
    + df.plot.line(x='col1',y='col2')：指定x,y坐标数据，绘制一根线
        + df.plot.line(y='col2')：以0,1,2,...为x坐标，绘制一根线
+ 多个数据序列图
    + df.plot.line(x='col1',y=['col2','col3'])：绘制两根曲线。x坐标都为col1，y坐标分别为col2,col3。
        + df.plot.line(x='col')：只指定x坐标数据，y坐标数据自动取df的其他列数据，绘制多根线
        + x省略，y值设置为列表
        + df.plot.line()：每个列绘制一根曲线，x坐标值采用自然数

多个序列绘图效果如下：

![Pandas多列绘图](https://img-blog.csdnimg.cn/05d0ae8f28cc49c1aa583f7f90258a15.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


**stacked**叠加绘图参数

注意上图中area的图形和line的图形不太一样，这是为什么呢？area的stacked参数默认为True，即叠加绘图。第2列数据的y坐标会与前一列的坐标值相加后显示。效果如下：

![Pandas多列stacked绘图](https://img-blog.csdnimg.cn/3f458656b00d47cfae61d23ae5b9ce4f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


> 注意：
> + area默认stacked为True，其他函数默认为False
> + line和area的stacked设置为True时，y值必须大于等于0

**子图绘制**

默认情况下多个列的数据绘制线、柱都在一个图形中。可以通过subplots参数将各个列数据分开在不同的子图中绘图。

以`df.plot.bar(x='col1',subplots=True)`为例，效果如下：

![子图subplots](https://img-blog.csdnimg.cn/76c778bcba664cd2b111949ec9d3feb2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


> 在subplots=True时，可以用layout=(2,3)参数设置子图的行列数（2行3列）。


### 2.2 统计绘图

直方图、密度分布图、箱型图、六边形填充图。

+ 直方图：用柱形图显示数据分布密度。
+ 核密度分布图：用指定的核密度函数(数据假定或已知的分布函数)拟合直方图。
+ 箱型图：用一个箱型框显示一列数据的最小值，25%分位值，中位数，75%分位值，最大值。
+ 六边形填充图：在x,y坐标处绘制平面六边形，用六边形颜色表示x,y坐标出现的次数


+ 直方图
    + d.plot.hist(bins=5)：绘制数据的直方图。
    	+ bins表示数据分为bins段统计分布，用bins个柱形绘图。默认bins=10。
    	+ 对DataFrame，为每一列单独
    + df.plot.hist(stacked=True)：在一个图中绘制每一列的直方图。默认stacked为False，即不叠加

直方图，stacked为False和True的效果：

![pandas hist多列](https://img-blog.csdnimg.cn/18735751995949aea5d27aa3bfdda017.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



+ 密度分布图
    + s.plot.density()：绘制高斯核的概率密度曲线
    + df.plot.density()：绘制每列数据的高斯核概率密度曲线

> density、kde两个方法完全相同，用于核密度估计。采用非参数方法估算随机变量的概率密度函数(用函数使用高斯核)

```python
d=pd.Series([1,1,1,2,3,4])
d.plot.kde()
```

![pandas核密度图](https://img-blog.csdnimg.cn/5a92965e91f843d2b67782283e406b64.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



+ 箱形图（用每个列的四个分位数绘制箱型图）
    + s.plot.box()：绘制一个箱型图。
    + df.plot.box()：绘制箱型图，每个列绘制一个箱型框

```
s=pd.Series([1,2,3,4,5])
df=pd.DataFrame(np.random.random((50,4)))
s.plot.box()
df.plot.box()
```

![Pandas box图](https://img-blog.csdnimg.cn/2bd43d03d532483ca4fd769f2e655d38.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


+ 六边形填充图
    + df.hexbin(x='col1',y='col2')：在每一个(xi,yi)坐标处绘制一个六边形，六边形颜色表示(xi,yi)出现的次数
    + df.hexbin(x=['col1','col1'],y=['col2','col3'])：指定多个列作为值对

> 与scatter相似。Series不支持hexbin。x,y参数也都不能省略，x,y都为列表时，长度必须相同。

```
n = 10000
df = pd.DataFrame({'x': np.random.randn(n),
                   'y': np.random.randn(n)})
ax = df.plot.hexbin(x='x', y='y', gridsize=20)
```

![pandas hexbin](https://img-blog.csdnimg.cn/85f9f95ed5f9458d9bbfbe518a933623.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


> hexbin绘图建议设置gradsize，否则六边形图很小，且间距很大，视角效果不好

## 3. plot函数

+ df.plot(x='col1',y='col2',kind='scatter')：散点图。与df.plot.scatter效果完全相同。
+ d.plot(x='col1',y='col2',kind='line')：折线图。与d.plot.line效果完全相同。
+ d.plot(x='col1',y='col2',kind='area')：颜色填充的折线图。与d.plot.area效果完全相同。
+ d.plot(x='col1',y='col2',kind='bar')：柱状图。与d.plot.bar效果完全相同。
+ d.plot(x='col1',y='col2',kind='barh')：水平柱状图。与d.plot.barh效果完全相同。
+ d.plot(y='col2',kind='pie')：饼状图。只需要指定y值。与d.plot.pie效果完全相同。

> 特点：与plot模块等效，通过kind参数指定绘图方法。其他参数完全相同

## 4. Series和DataFrame两个特殊绘图函数

DataFrame可以直接调用hist和boxplot函数。直接调用函数，相比plot模块和plot函数，多一个column参数。

+ df.hist(column='col1')：指定列绘制直方图。df.hist(column=['col1','col2'])：为每列绘制一个直方图子图。df.plot.hist()默认将多列直方图绘制在一个子图中
+ df.boxplot(column='col1')：指定列绘图。df.boxplot(column=['col1','col2'])：指定多列绘图。

## 5. 通用图形控制参数

在调用绘图函数时，还可以设置一些其他参数，调整绘图效果，这些参数是通用的。形式如下：

```python
d.plot(x='x',y='y',kind='line',title='title',xlim=(1,5))
```

### 5.1 图形属性

+ 图形
    + `figsize=(4,3)`：图形宽高分别设置为4,3
    + `title='title1'`：设置图形标题，用列表为每个子图设置标题
    + `ax=ax1`：图形绘制到axes对象上ax1
+ 样式
	+ `style=['r:','-']`：为每列数据指定样式。(r:表示红色虚线，-表示实线)
	+ `fontsize=10`：xticks的字体大小
	+ `legend=False`：是否显示图例
+ 坐标轴设置
	+ `xlim=(1,5)`:设置x轴坐标上下限
	+ `ylim=(1,5)`：设置y轴坐标上下限
	+ `xticks=[1,2,3,4]`：设置x轴坐标
	+ `yticks=[1,2,3]`：设置y轴坐标
	+ `xlabel='x'`：设置x轴标签
	+ `ylabel='y'`：设置y轴标签
	+ `logx=True`：坐标轴是否按对数坐标轴绘制
	+ `logy=True`:坐标轴是否按对数坐标轴绘制




<br>
<hr>

> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
