Seaborn系列(四)：类别统计绘图(categorical)
==============================

[Seaborn系列目录](https://blog.csdn.net/hustlei/article/details/123087608)
<hr>

@[TOC]

<hr>


> <font color=#999>在研究数值变量之间的关系时，如果其中一个主要变量是"类别"，那么使用更类别统计绘图方法可能会更好。

# 1. 类别统计绘图API概述

seaborn中“类别”绘图函数共9个：

+ `catplot()`：通过kind参数。默认为strip。
    + 分类散点图
        + `stripplot()`：散点条图
        + `swarmplot()`：散点群图
    + 分类分布图
        + `boxplot()`：箱型图
        + `boxenplot()`：增强箱型图
        + `violinplot()`：小提琴图
    + 分类估计图
        + `barplot()`
        + `pointplot()`
        + `countplot()`

> + catplot函数为figure级函数，其他函数为axes级函数。

figure级函数与axes级函数区别见[Seaborn系列（一）：绘图基础、函数分类、长短数据类型支持及数据集](https://blog.csdn.net/hustlei/article/details/123087693)

# 2. catplot基本绘图

+ `sns.catplot(x=None,y=None,data=None,kind="strip")`：根据kind绘制分类图。

> catplot同样支持legend、height、aspect等参数。hue等参数在后续详细介绍

## 2.1 catplot绘制分类散点图

分类散点图实际上就是把x坐标相同的数据同时绘制出来。catplot共有两种方法：

+ 散点条图：数据按x分组，所有x相同的y值，在x坐标上方绘制散点，散点形成条状。

```
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

sns.catplot(data=tips, x="day", y="total_bill", kind="strip")

plt.show()
```

![catplot1_strip](https://img-blog.csdnimg.cn/1b45b9403e144475aa362beb33874b70.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


+ 散点群图：类似散点条图，但是y值相同的点显示为树状。

`sns.catplot(data=tips, x="day", y="total_bill", kind="swarm")`

![catplot2_swarm](https://img-blog.csdnimg.cn/41f59e56f9bd45e59617f1ff24718027.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 2.2 catplot绘制分类分布图

分类分布图包括：箱型图、增强箱型图、小提琴图。

箱型图把按x分组的数据，每一组数据分别统计中位数、25%位置数、75%位置数，以及高位和低位值。

```
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

sns.catplot(data=tips, x="day", y="total_bill", kind="box")

plt.show()
```

![catplot3_box](https://img-blog.csdnimg.cn/d6e4bb6188264f589d980cea02f47038.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


> 箱图上下方的点表示超出高位和低位的数值。

增强箱型图和箱型图类似，但是绘制更多的分位数。（分位数由参数k_depth确定）

`sns.catplot(data=tips, x="day", y="total_bill", kind="boxen")`

![catplot4_boxen](https://img-blog.csdnimg.cn/6adb148b38b14e01b1e3f45f0d6af1b2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


小提琴图外形像小提琴，越宽的地方表示y值密度越大，小提琴图由轮廓和内部图形组成：

+ 轮廓是kde核密度估计曲线。
+ 内部是箱型图的四分位数。

`sns.catplot(data=tips, x="day", y="total_bill", kind="violin")`

![catplot5_violin](https://img-blog.csdnimg.cn/f78dfb126d494733a9f557ff9f875a5f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 2.3 catplot绘制分类估计图

如果你希望显示值的趋势估计值，可以采用分类估计图，而不是每个类别的分布。

catplot分类估计图包括：柱形图、点图

**柱形图**应用函数获取估计值（默认采用平均值），然后绘制成柱形图。

```
import matplotlib.pyplot as plt
import seaborn as sns
tips = sns.load_dataset("tips")

sns.catplot(data=tips, x="day", y="total_bill", kind="bar")

plt.show()
```

![catplot6_barplot](https://img-blog.csdnimg.cn/f915be3bd9e24391bdfb5de769e1ba01.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


柱形图上方的短线为置信区间，和lineplot置信区间相同，把柱形图和lineplot图重叠对比如下：

```
sns.catplot(data=tips, x="day", y="total_bill", kind="bar")
sns.lineplot(data=tips, x="day", y="total_bill")
```

![catplot7_barplot_lineplot](https://img-blog.csdnimg.cn/69bfcfc854bd4c5eabc814af168f3009.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)



**点图**绘制平均值曲线和置信区间，和lineplot非常像，显示的线条和短线和lineplot也完全一致。

```
sns.catplot(data=tips, x="day", y="total_bill", kind="point")
```

![catplot8_pointplot](https://img-blog.csdnimg.cn/9eca1eb5384545bbacc9dbf4625c6549.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


**计数图**对指定x或y坐标中每个值出现的次数进行统计，绘制成柱形图。柱形高度坐标值统计个数。

```
sns.catplot(data=tips, x="day", kind="count")
```

![catplot9_countplot](https://img-blog.csdnimg.cn/9407fb95969f4ef8b8a361d0a58b31b9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 3. 分类散点图

## 3.1 strip图

+ `stripplot()`：散点条图catplot(kind="strip")
    + hue参数：分组按不同颜色绘图
    + dodge参数：是否将不同的组分开绘制
    + jitter参数：设置抖动量，False表示散点条绘制在一条线上。默认为True。

```
sns.catplot(data=tips, x="day", y="total_bill", kind="strip", hue="sex")
```

![stripplot1_hue](https://img-blog.csdnimg.cn/3907a73075d6426eb35e4a10a014a45b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.catplot(data=tips, x="day", y="total_bill", kind="strip", hue="sex", dodge=True)
```

![stripplot2_hue_dodge](https://img-blog.csdnimg.cn/2d2af06c788d4e3daa3ee4bbe13d1861.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.catplot(data=tips, x="day", y="total_bill", kind="strip", jitter=False)
```

![stripplot3_hue_jitter1](https://img-blog.csdnimg.cn/01bd5dbaaec94c8688d4d4d4c0889b3f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


> jitter也可以用数值表示抖动量，即条形图的宽度，一般可以用0-1值。大于1的值也可以，但是就看不出每条的分界了。

> linewidth, size, color等参与可以用于设置marker的样式。pallete参数可以设置hue的颜色。

## 3.2 swarm图

+ `swarmplot()`：散点群图catplot(kind="swarm") 
    + hue参数：分组按不同颜色绘图
    + dodge参数：是否将不同的组分开绘制

`sns.catplot(data=tips, x="day", y="total_bill", kind="swarm", hue="sex")`

![swarmplot4_hue](https://img-blog.csdnimg.cn/231f6307b55d4d3fa521572c2fc0afe8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


`sns.catplot(data=tips, x="day", y="total_bill", kind="swarm", hue="sex", dodge=True)`

![swarmplot5_hue_dodge](https://img-blog.csdnimg.cn/9d932dc4acb74cb5bb2621830f1af602.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


> linewidth, size, color等参与可以用于设置marker的样式。pallete参数可以设置hue的颜色。

# 4 分类分布图

## 4.1 box图

+ `boxplot()`：散点群图catplot(kind="box") 
    + hue参数：分组按不同颜色绘图
    + dodge参数：是否将不同的组分开绘制
    + orient参数："v"|"h"，设置box图方向
    + whis参数：设置低位和高位点系数，高位、低位数公式：中位数±whis*(75%位数-25%位数)
        - whis设置为np.inf时，低位和高位值取min和max。

```
sns.catplot(data=tips, x="day", y="total_bill", kind="box", hue="sex")
#sns.catplot(data=tips, x="day", y="total_bill", kind="box", hue="sex", dodge=True) #dodge参数没有效果
```

![boxplot1_hue](https://img-blog.csdnimg.cn/37462b5de6244eb294fa704e36f0b51c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


orient参数可以改变box图方向。但是有一种情况box会自动改变方向，可以省略orient参数。

```
sns.catplot(data=tips, y="day", x="total_bill", kind="box")# 省略orient="h"，因为y是非数值类型分类数据
```

![boxplot2_orient](https://img-blog.csdnimg.cn/650d969c091e4b0ebddc4f899353da23.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.catplot(data=tips, x="day", y="total_bill", kind="box", whis=np.inf)
```

![boxplot3_whis](https://img-blog.csdnimg.cn/d653423ea5a549ac9708a2a49db6c960.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 4.2 boxen图

+ `boxenplot()`：增强箱型图catplot(kind="boxen") 
    + hue参数：分组按不同颜色绘图
    + dodge参数：是否将不同的组分开绘制
    + orient参数："v"|"h"，设置boxen图方向
    + k_depth参数：箱形图箱数设置。 
        + 当为数字是表示具体箱形框数量。
        + “tukey”:默认值，还可以是“proportion”, “trustworthy”, “full”，都是内置的确定方法。在 Wickham的论文中有论述
  
boxenplot的hue、dodge和orient参数和boxplot函数相似：

```
sns.catplot(data=tips, x="day", y="total_bill", kind="boxen", hue="sex", dodge=True)
```
  
![boxen4_hue](https://img-blog.csdnimg.cn/4091b977ac774580b9332c1479eb6738.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


k_depth参数通常用默认的就ok，用proportion和full会绘制的更加详细。数值可以指定箱形框数量；

![boxen5_depth4](https://img-blog.csdnimg.cn/c21060d63ecc4d4c966db6743213a9aa.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

![boxen6_depth_port](https://img-blog.csdnimg.cn/5cc24eb1b51e4aac80893fde92653fb0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 4.3 violin图

+ `violinplot()`：增强箱型图catplot(kind="violin") 
    + hue参数：分组按不同颜色绘图
    + dodge参数：是否将不同的组分开绘制
    + orient参数："v"|"h"，设置boxen图方向
    + split参数：当绘制两组图时，设置为True，则分别绘制半个提琴
    + 轮廓核密度估计参数：bw（带宽参数），gridsize(用于计算核密度估计值的点数)
    + innner参数：内部数据点表示方法
        + None不显示内部
        + box默认值，内部显示箱型图
        + quartile，内部显示四分点
        + point，内部在中心线显示所有点位置
        + stick，内部用横线显示所有点位置

hue、dodge、orient与其他图形用法相同。split参数可以和hue结合，把2组图形分别绘制一半：

```
sns.catplot(data=tips, x="day", y="total_bill", kind="violin", hue="sex",split=True)
```

![violinplot8_hue_split](https://img-blog.csdnimg.cn/51ca447935fb4fcaaf2d14b10c4d68d6.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



核密度估计参数改变会导致提琴轮廓变化。提琴轮廓实际上是kde曲线。

bw是kde的参数，值越小曲线越精细，线条弯曲越多。

```
sns.catplot(data=tips, x="day", y="total_bill", kind="violin", bw=0.2)
```

![violinplot7_bw](https://img-blog.csdnimg.cn/260ff131fa13420db8b38c54b6586677.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)


> gridsize为整数。表示绘制kde的计算点数，实际上也是提琴轮廓点数，点越少，轮廓越不光滑。一般不建议自行设置。


```
sns.catplot(data=tips, x="sex", y="total_bill", kind="violin",inner=None)
sns.catplot(data=tips, x="sex", y="total_bill", kind="violin",inner="box")
sns.catplot(data=tips, x="sex", y="total_bill", kind="violin",inner="quartile")
sns.catplot(data=tips, x="sex", y="total_bill", kind="violin",inner="point")
sns.catplot(data=tips, x="sex", y="total_bill", kind="violin",inner="stick")
```

![violinplot9_inner](https://img-blog.csdnimg.cn/dd5fe76a71214f09a3aa0e02f6285fe3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 5. 分类估计图

## 5.1 barplot

+ `barplot()`：柱形图catplot(kind="bar") 
    + hue参数：分组按不同颜色绘图
    + dodge参数：是否将不同的组分开绘制
    + orient参数："v"|"h"，设置boxen图方向
    + errwidth参数：误差线宽度
    + capsize参数：误差线端部大小
    + ci参数：误差线置信区间0—100
    + estimator参数：估计函数，也就是计算柱形图高度值的参数。

> hue、dodge和orient参数与其他函数相同。

```
sns.catplot(data=tips, x="day", y="total_bill", kind="bar", hue="sex", dodge=True)
```

![barplot1_hue_dodge](https://img-blog.csdnimg.cn/db072444bcea49f1a5be6fd3982db93f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



errwidth、capsize和ci参数是针对柱形图顶部误差线的：

```
sns.catplot(data=tips, x="day", y="total_bill", kind="bar", capsize=0.3, errwidth=5, ci=50)
```

![barplot1_err](https://img-blog.csdnimg.cn/4428f8d895c24345abe340fc856667da.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)


默认柱形图高度为平均值，estimator参数可以设置确定柱形图高度的函数。

```
sns.catplot(data=tips, x="day", y="total_bill", kind="bar", estimator=max)
```

![barplot3_estimator](https://img-blog.csdnimg.cn/064e5e7c950645a3a3f2369a8c232545.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 5.2 pointplot

+ `pointplot()`：柱形图catplot(kind="point") 
    + hue参数：分组按不同颜色绘图
    + dodge参数：是否将不同的组分开绘制
    + orient参数："v"|"h"，设置boxen图方向
    + errwidth参数：误差线宽度
    + capsize参数：误差线端部大小
    + ci参数：误差线置信区间0—100
    + estimator参数：估计函数，也就是计算柱形图高度值的参数。
    + **join参数**：是否绘制折线，False就只绘制误差线。

> hue、dodge、orient、errwidth、capsize、ci、estimator参数与barplot相同。

```
sns.catplot(data=tips, x="day", y="total_bill", kind="point")
sns.catplot(data=tips, x="day", y="total_bill", kind="point", join=False)
```

![pointplot4_join](https://img-blog.csdnimg.cn/fabfc0ad31b94752975e3e9f2b963cf3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 5.3 countplot

+ `pointplot()`：柱形图catplot(kind="point") 
    + hue参数：分组按不同颜色绘图
    + dodge参数：是否将不同的组分开绘制
    + orient参数："v"|"h"，设置boxen图方向

```
sns.catplot(data=tips, x="day", kind="count", hue="sex")
```

![countplot5](https://img-blog.csdnimg.cn/83bb6f14dda34a129f7ad1db5253ffd5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


> palette, facecolor, linewidth, edgecolor等matplotlib基本样式设置参数也同样适用于以上函数。
> height, aspect等参数同样适用。

<br>
<hr>


[Seaborn系列目录](https://blog.csdn.net/hustlei/article/details/123087608)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>  
>  <font color=#888>**未经允许请勿转载。**
