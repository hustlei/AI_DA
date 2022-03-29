Seaborn系列(六)：样式和组合图
==============================


[Seaborn系列目录](https://blog.csdn.net/hustlei/article/details/123087608)
<hr>

@[TOC]

<hr>

# 1. Seaborn样式

# 1.1 设绘seaborn主题

```
sns.set_theme()  #切换到seaborn默认主题。
#sns.set()      #set是set_theme的别称，通常建议使用set函数
```

> 在0.8版本以前默认已经调用了这个函数，高版本seborn上，必须显式调用。

**set_theme**函数设置样式


+ `sns.set_theme(context='notebook', style='darkgrid', palette='deep')`
    + context参数：上下文参数，见plotting_context()
    + style参数：样式参数，见axes_style()
    + palette参数：色盘参数，见color_palette()

**set_theme**设置字体样式

+ `sns.set_theme(font=''sans-serif', font_scale=1)`
    + font参数：设置字体
    + font_scale参数：缩放字体大小

**set_theme**设置rcParams

+ `sns.set_theme(rc=None)`：rc参数可以用rcParams字典重置set_theme设置的参数。

> set_theme函数实际上就是设置matplotlib的rcparams参数。但是把参数分为了style、palette等几部分。

## 1.2 style样式设置

样式参数控制背景颜色、示范启用网格等样式属性，实际上也是通过matplotlib的rcParams实现的。

**sns.axes_style(style=None, rc=None)**：获取样式参数，或者作为上下文管理器，临时改变样式。

+ `sns.axes_style()`：获取默认样式，返回一个样式字典。
+ `sns.axes_style("darkgrid)`：获取darkgrid样式的参数字典。
+ `sns.axes_style("darkgrid", rc={'axes.grid':True})`：用rc覆盖样式中部分参数。

```
with sns.axes_style("whitegrid"):  #样式仅对下方块中语句生效。
    sns.barplot(x=...,y=...)
```

> 注意对于多个子图用不同样式时，add_subplot需要在with语句内才能起作用。

**sns.set_style(style=None, rc=None)**：设置样式。

+ `sns.set_style("whitegrid")`：使用指定样式
+ `sns.set_style("whitegrid",{"grid.color": ".6", "grid.linestyle": ":"})`：对样式指定参数修改后应用样式。

## 1.3 可用样式style

+ darkgrid
+ whitegrid
+ dark
+ white
+ ticks

![style](https://img-blog.csdnimg.cn/425e4b7bdeb64d5c9d2572122f323d74.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_13,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 1.4 颜色盘设置

+ `sns.color_palette()`：获取颜色盘，返回一个颜色列表
+ `sns.set_palette("dark")`：设置颜色盘，参数可以是：
    - 颜色盘名称：deep，muted，bright，pastel，dark，colorblind等
    - husl或者hls
    - "light:<color>", "dark:<color>", "blend:<color>,<color>"
    - 颜色列表


## 1.5 上下文设置

上下文设置绘图元素的比例，用不同的上下文表示适用于较大尺寸绘图或较小尺寸绘图，设置上下文后，相同的绘图函数显示的图形会根据上下问自动缩放。

+ `sns.set_context('paper')`：最小的尺寸
+ `sns.set_context('notebook')`：稍大的尺寸
+ `sns.set_context('talk')`：更大的尺寸
+ `sns.set_context('poster')`：最大的尺寸

## 1.6 恢复默认样式

+ `sns.reset_defaults()`：将所有RC参数恢复为默认设置。

# 2. 组合图

seaborn中包含两个组合图函数，都是figure级绘图函数。

+ jointplot：绘制二维图，并在二维图上方绘制x的分布图，在右侧绘制y的分布图
+ paireplot：多变量配对分布图，对data中的多个变量两两配对，分别绘制变量关系图，形成子图方阵（对角线上的子图上x,y坐标变量相同，绘制单变量分布图。其余子图绘制双变量关系图）。

## 2.1 jointplot组合分布图

`sns.jointplot(x=None,y=None,data=None,kind='scatter',hue=None)`:根据kind确定图形样式绘制。
    + kind参数指定二维图形类型
        + scatter：散点图。默认值
        + kde：核密度图。
        + hist：二维直方图。
        + hex：六边形图（不能和hue同时使用）
        + reg：线性回归拟合图（不能和hue同时使用）
        + resid：线性回归拟合误差图（不能和hue同时使用）
    + hue参数，分组绘图。

> 默认情况下上方和侧方绘制直方图。当kind为scatter或kde，并且指定hue时，上方和侧方绘制kde图。


**kind="scatter"**:绘制散点图，上方和右侧绘制直方图。kind默认为scatter。

```
import seaborn as sns
import matplotlib.pyplot as plt
data = sns.load_dataset("penguins")

sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="scatter")

plt.show()
```

![jointplot1_scatter](https://img-blog.csdnimg.cn/7ec3b5d2783e41ed936f2351b591d2f4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


使用hue参数后，散点图将分颜色绘制，上方和右侧的分布图将用kde（核密度分布图）代替直方图。

```
sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, hue="species")
```

![jointplot1_scatter_hue](https://img-blog.csdnimg.cn/ee0204478e2e453b838c79eec7e396c4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="kde")
```

![jointplot2_kde](https://img-blog.csdnimg.cn/9e59679bb45242e7abaaadc581bfd02f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="kde", hue="species")
```

![jointplot2_kde_hue](https://img-blog.csdnimg.cn/e728d9e9307241a293cbdea00a656094.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="hist")
```

![jointplot3_hist](https://img-blog.csdnimg.cn/7d22669226a44f44a5087d8754bd6971.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


`sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="hist", hue="species")`

![jointplot3_hist_hue](https://img-blog.csdnimg.cn/cd5fd820a34d47398a5a2a6f236749cb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


`sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="hex")`

![jointplot4_hex](https://img-blog.csdnimg.cn/dda776c481c148ffa284a787ad8df8b5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


`sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="reg")`

![jointplot5_reg](https://img-blog.csdnimg.cn/4a65aeafec62426aaffc968d020720e4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)


`sns.jointplot(x="bill_length_mm", y="bill_depth_mm", data=data, kind="resid")`

![jointplot6_resid](https://img-blog.csdnimg.cn/34f963d490f04f66a9f9f1f9a272eeb6.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_15,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 2.2 pairplot配对关系图

**sns.pairplot(data, hue=None, kind='scatter', diag_kind='auto')**：对data的所有变量两两配对，形成方阵，kind指定绘图类型，diag_kind指定对角线上图形类型。

+ hue参数，每个图形中都分组用不同的颜色绘图
+ kind参数：图形类型（scatter’, ‘kde’, ‘hist’, ‘reg’）
+ diag_kind参数：单变量图（对角线）类型（’auto’, ‘hist’, ‘kde’, None)
+ coner参数：corner为True时，只绘制下三角。

> vars=None, x_vars=None, y_vars=None参数可以指定data中需要绘制的参数。

```
data = sns.load_dataset("penguins")
sns.pairplot(data=data)
```

![paireplot](https://img-blog.csdnimg.cn/b83545559a8e4b13ae6fcda050dbf125.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
data = sns.load_dataset("penguins")
sns.pairplot(data=data,hue="species")

```

![paireplot_hue](https://img-blog.csdnimg.cn/b8bb1e45bbcd41dd8783f7288d2ad060.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)





<br>
<hr>


[Seaborn系列目录](https://blog.csdn.net/hustlei/article/details/123087608)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>  
>  <font color=#888>**未经允许请勿转载。**
