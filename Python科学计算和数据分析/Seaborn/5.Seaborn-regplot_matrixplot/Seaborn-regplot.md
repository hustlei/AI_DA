Seaborn系列（五）：回归(Regression)及矩阵(Matrix)绘图
==============================


[Seaborn系列目录](https://blog.csdn.net/hustlei/article/details/123087608)
<hr>

@[TOC]

<hr>

> <font color=#999>Seaborn中的回归包括回归拟合曲线图以及回归误差图。
> <font color=#999>Matrix图主要是热度图。

# 1. 回归及矩阵绘图API概述

seaborn中“回归”绘图函数共3个：

lmplot（回归统计绘图）：figure级regplot函数，绘图同regplot完全相同。(lm指linear model)
    + regplot：axes级函数。绘制线性回归拟合。
    + residplot：axes级函数。绘制线性回归的误差图。（不能用lmplot绘制resid图）

seaborn中矩阵绘图函数共有2个：

+ heatmap：axes级函数。热度图，绘制一个颜色块矩阵。
+ clustermap：figure级函数。聚合热度图，绘制一个分层聚合的热度图。

figure级函数与axes级函数区别见[Seaborn系列（一）：绘图基础、函数分类、长短数据类型支持及数据集](https://blog.csdn.net/hustlei/article/details/123087693)

# 2. 回归统计绘图

## 2.1 lmplot、regplot绘图

+ `sns.lmplot(x=None,y=None,data=None)`:绘制线性回归拟合图，返回FacetGrid
+ `sns.regplot(x=None,y=None,data=None)`绘制线性回归拟合图，返回Axes
    - hue：分系列用不同的颜色绘制
    - col,row：指定参数不同值绘制到不同的行或列。
    - ci=95：置信区间的大小，取值0-100
    - order：指定拟合多项式阶数
    - scatter:是否绘制散点图
    - x_jitter,y_jitter:为x变量或y变量添加随机噪点。会导致绘制的散点移动，不会改变原始数据。
    - x_estimator：参数值为函数，如np.mean。对每个x值的所有y值用函数计算，绘制得到的点，并绘制误差线。
    - x_bins：当x不是离散值时x_estimator可以配合x_bins指定计算点和误差线数量
    - robust：对异常值降低权重
    - logistic：logistic=True时，假设y取值只有2个比如True和False，并用statsmodels中的逻辑回归模型回归。

`sns.lmplot(data=tips, x="total_bill", y="tip")`

![regplot1_lmplot](https://img-blog.csdnimg.cn/8160c4a67066497eb7ceffb8e0266791.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


hue、col、row参数与其他函数用法相同

`sns.lmplot(data=tips, x="total_bill", y="tip", hue="sex", col="smoker")`

![regplot2_hue_col](https://img-blog.csdnimg.cn/8b1d2fd2faff40349f9fc56832d8ba5e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


图中拟合直线旁边透明颜色带是回归估计的置信区间，默认置信区间为95%。ci参数可以设置置信区间，ci取None则不绘制置信区间。

`sns.lmplot(data=tips, x="total_bill", y="tip", ci=50)`

![regplot3_ci](https://img-blog.csdnimg.cn/a167292cc8e6423794e10a64ff264916.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


`sns.lmplot(data=tips, x="total_bill", y="tip", order=3)`

![regplot3_order](https://img-blog.csdnimg.cn/030982a8659c4802bb7564ea0e0e6609.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


`sns.lmplot(data=tips, x="total_bill", y="tip", scatter=False)`

![regplot3_scatter](https://img-blog.csdnimg.cn/b421f264d07f4384bbeaa49641aedcba.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


x_jitter会随机改变图中散点的x坐标，y_jitter会随机改变图中散点的y坐标。

`sns.lmplot(data=tips, x="total_bill", y="tip", y_jitter=10)`

![regplot4_jitter](https://img-blog.csdnimg.cn/ba46ccaf22e545e498645974695a2057.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


`sns.lmplot(data=tips, x="total_bill", y="tip", x_estimator=np.mean, x_bins=4)`

![regplot5_x_estimator_bins](https://img-blog.csdnimg.cn/697abc1b694847a884535c0261a7b3da.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


``

robust参数为True时，会降低异常值的权重，在需要剔除异常值时，非常有用。
但是使用robust后，计算量会比较大，通常建议取ci=None加速。
注意robust参数需要安装statsmodels模块。

```
import matplotlib.pyplot as plt
import seaborn as sns
ans = sns.load_dataset("anscombe")
dat = ans.loc[ans.dataset == "III"]

sns.lmplot(data=dat, x="x", y="y", robust=True, ci=None)

plt.show()
```

![](https://img-blog.csdnimg.cn/img_convert/e11ddf8856d0f86474b4ba3d5b590ba2.png)



## 2.2 residplot绘图

+ `sns.residplot(x=None,y=None,data=None)`绘制线性回归拟合图的残差
    - order：回归拟合阶数
    - robust：对异常值降低权重
    - dropna：忽略空值

```
sns.residplot(data=tips, x="total_bill", y="tip")
```

![residplot](https://img-blog.csdnimg.cn/a015900fdce241c6a5f3d6d5d7e1a1a4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_14,color_FFFFFF,t_70,g_se,x_16#pic_center)



# 3. 矩阵图

## 3.1 heatmap热力图

+ `sns.residplot(data)`:绘制热力图
    + annot：在单元格内显示数据。
    + fmt：设置annot参数数据显示格式。
    + cbar：是否显示颜色条。
    + cmap：设置colormap。
    + square：单元格是否方形。
    + linewidths：设置单元格线条宽度。
    + linecolor：设置单元格线条颜色。


```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data = np.random.rand(10, 10)

sns.heatmap(data=data)

plt.show()
```

![heatmap1](https://img-blog.csdnimg.cn/b1216c4bb9b04d589e05e09be8795900.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.heatmap(data=data, annot=True, fmt=".2f")
```

![heatmap2_annot](https://img-blog.csdnimg.cn/95916070d3124ba6a5a567481fe478c8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_12,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.heatmap(data=data, cmap="hsv", cbar=False, linewidths=0.5, linecolor="w")
```

![heatmap3_style](https://img-blog.csdnimg.cn/5b9fa054937e4794b6518055dcffe7fd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_13,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 3.2 clustermap分层聚合热力图

```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
data = np.random.rand(10, 10)

sns.clustermap(data=data)

plt.show()
```

![clustermap](https://img-blog.csdnimg.cn/8b13d58739d04ec68453e2b78df44e8a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


> clustermap说明详见<https://zhuanlan.zhihu.com/p/165940283>


<br>
<hr>


[Seaborn系列目录](https://blog.csdn.net/hustlei/article/details/123087608)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>  
>  <font color=#888>**未经允许请勿转载。**
