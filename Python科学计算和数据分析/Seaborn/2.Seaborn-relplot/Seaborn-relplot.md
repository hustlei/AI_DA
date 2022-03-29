Seaborn系列(二)：关系绘图
==============================

[Seaborn系列目录](https://blog.csdn.net/hustlei/article/details/123087608)
<hr>

@[TOC]

<hr>

> <font color=#999>关系绘图、分布统计绘图、类别统计绘图是seaborn中最常用的功能。

# 1. 关系绘图API概述

统计分析就是了解数据变量之间的关系的过程。seaborn中“统计关系”绘图函数共三个：

+ `relplot()`：通过kind参数指定为"scatter"或"line"，可以绘制散点图和折线图。默认为scatter。
    + `scatterplot()`：绘制散点图。绘图效果同replot(kind="scatter")
    + `lineplot()`：绘制折线图。绘图效果同replot(kind="line")

> + relplot函数为figure级函数，返回FacetGrid对象。FacetGrid对象类似于figure。
> + scatterplot和lineplot函数为axes级函数，返回axes对象。

figure级函数与axes级函数区别见[Seaborn系列（一）：绘图基础、函数分类、长短数据类型支持及数据集]()

# 2. relplot散点图

+ `sns.relplot(x=None,y=None,data=None,kind="scatter")`：根据kind绘制折线图或散点图。
    + 常用参数简介
        + x,y参数：x,y坐标数据。x,y为数组或者是data的key名称（data为DataFrame时，x,y指列名称）。
        + data参数：data为类字典数据。当data不为空时，x,y可以用data的key引用数据。
        + kind参数：scatter指绘制散点图，line指绘制折线图。
    + 分组聚合参数
        + hue参数：用不同颜色对数组分组。hue可以是列表或者data的key。hue的每一个数据绘制一根曲线，给一个图例标签。
        + size参数：用不同点大小对数组分组。数组或者data中的key。每个大小给个图例标签。
        + style参数：用不同的点样式对数据分组。数组或者data中的key。每个样式给一个图例标签。
        + row,col：把row,col指定的数据按照行或列排列到不同的子图。
    + 分组聚合参数样式
        + palette参数：指定hue分组的每组曲线的颜色。
        + markers参数：设置True使用默认点样式。False不显示点。可以用列表为style设置对应点样式(长度必须和style分组数相同)。
        + sizes参数：把size映射到一个指定的区间。
    + 其他参数
        + legend参数：图例显示方式。False不现实图例。brief则hue和size的分组都取等间距样本作为图例。full则把分组内所有数值都显示为图例。auto则自动选择brief或full。
        + height参数：每个子图的高度(单位inch)
        + aspect参数：宽度=aspect×高度

> hue,style,size类似，都是用于分组，hue根据颜色分组，style根据点先样式分组，size根据点大小或线粗细分组。每个组在曲线图中就是绘制一根单独的曲线。

> 以tips小费数据集为例。

```python:GT
>>> tips=sns.load_dataset('tips')
>>> tips
     total_bill   tip     sex smoker   day    time  size
0         16.99  1.01  Female     No   Sun  Dinner     2
1         10.34  1.66    Male     No   Sun  Dinner     3
2         21.01  3.50    Male     No   Sun  Dinner     3
3         23.68  3.31    Male     No   Sun  Dinner     2
4         24.59  3.61  Female     No   Sun  Dinner     4
..          ...   ...     ...    ...   ...     ...   ...
239       29.03  5.92    Male     No   Sat  Dinner     3
240       27.18  2.00  Female    Yes   Sat  Dinner     2
241       22.67  2.00    Male    Yes   Sat  Dinner     2
242       17.82  1.75    Male     No   Sat  Dinner     2
243       18.78  3.00  Female     No  Thur  Dinner     2
```

> 数据集为一个餐厅的小费数据。
> total_bill为总账单、tip为小费金额；
> sex为消费者性别，smoker为是否抽烟，size为用餐人数。
> day、time分别为日期、时间。

## 2.1 relplot简单的关系绘图

```
sns.relplot(data=tips,x='total_bill',y='tip') #观察账单和小费的关系，绘制散点图
sns.relplot(data=tips,x='total_bill',y='tip',kind='line') #观察账单和小费的关系，绘制折线图
```

![relplot1](https://img-blog.csdnimg.cn/c5d2129312ef49828d9981487e5eefa7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot2](https://img-blog.csdnimg.cn/17564dbd00684d34be28e9021d22f6d5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 2.2 relplot颜色(hue)分组绘图

```
sns.relplot(data=tips,x='total_bill',y='tip',hue='day') #分别观察每天的账单和小费的关系。
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker') #分别观察抽烟和不抽烟人群的账单和小费的关系。
sns.relplot(data=tips,x='total_bill',y='tip',hue='size') #根据就餐人数，分别观察人群的账单和小费的关系。
```

![relplot-散点图1_hue](https://img-blog.csdnimg.cn/063236b1c9234bb28433bef2d0afb076.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图2_hue](https://img-blog.csdnimg.cn/9db1b0c33e744250ac683a4f1b20c1d0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图3_hue](https://img-blog.csdnimg.cn/544c484518d6434a943e2885b31697f4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


> hue分组后对不同的组用不同的颜色区分。

+ 用palette参数可以指定hue映射的颜色。palette参数可以是字符串、列表、字典或者colormap。
    + 字符串：color_palette()函数的参数。
        + seaborn的palette名称，比如deep,muted,bright,pastel,dark,colorblind等。
        + matplotlib的colormap名称，比如Oranges, coolwarm。
        + 'hsl'或者'huls'：表示hsl颜色系统或者huls颜色系统。
        + 加关键字处理颜色：‘light:<color>’, ‘dark:<color>’, ‘blend:<color>,<color>’
        + cubehelix函数参数字符串‘ch:<cubehelix arguments>’：比如"ch:s=.25,rot=-.25"，"ch:2,r=.2,l=.6"
    + 列表：比如('r','b','gray')长度必须和hue分组数相同。
    + colormap：可以用字符串名称，比如"coolwarm"。也可以为colormap实例。

```
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',palette='coolwarm')  #用colormap指定颜色
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',palette='ch:start=2,rot=.5') #用cubehelix参数指定颜色
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',palette=('r','g','b','gray')) #用列表参数指定颜色
```

![relplot-散点图4_hue_palette](https://img-blog.csdnimg.cn/b0f0b48f17b04edab2a0aa76dad887c1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图5_hue_palette](https://img-blog.csdnimg.cn/8222007b6d7d4c4ea3588a916e071edf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图6_hue_palette](https://img-blog.csdnimg.cn/d283da1ea47241e6bd2d4c5e34a5ec2a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



> palette用列表或字典时，长度必须和hue分组数相同。比如上例中，day共有4种，因此，palette=('r','b','g','gray')是ok的。


## 2.3 relplot样式(style)分组绘图

style参数和hue类似，对参数不同的数据分组到不同的样式。

> 具体是什么样式有函数自行确定，或者根据marker参数确定。

```
#style分组
sns.relplot(data=tips,x='total_bill',y='tip',style="smoker")  #抽烟的人一个样式，不抽烟的人一个样式。
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',style="smoker")  #组合样式
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',style="time") #组合样式

#style+marker
sns.relplot(data=tips,x='total_bill',y='tip',style="time",markers=["o","*"]) #用markers参数指定style分组点样式
```

> 通常style的每个分组的点样式有系统自行决定。用marker可以自行指定，但是比较麻烦，且有限制，比如长度必须和style的分组个数相同；列表中只能全部是填充点,如("o","*")，或者全部是线条点，比如("+","x")

![relplot-散点图7_style](https://img-blog.csdnimg.cn/8543241340b64150ba65226cb5da69ad.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图8_style](https://img-blog.csdnimg.cn/72b802904dcb4cf495abb147b614bdaf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图9_style](https://img-blog.csdnimg.cn/5e3b893099014a6186eff8e86ac9d273.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图10_style](https://img-blog.csdnimg.cn/74537e48562e4cdca993f23c66ee1d61.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



**size分组，区分点大小绘制**

```
#size分组
sns.relplot(data=tips,x='total_bill',y='tip',size="smoker")#size分组(根据是否抽烟，分组)
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',size="size")#hue+size
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',style="size",size="smoker")#hue+style+size

#设置大小数值
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',size="size",sizes=(10,200)) #把size映射到10-200区间，用于显示大小
```

> 对于散点图，style和size和hue相似都是分组参数，知识分组表示方法不同。hue用颜色区分，style用点样式区分，size用点大小区分。

![relplot-散点图11_size](https://img-blog.csdnimg.cn/fee0448ca24f48bc8cc49b99fd7050ba.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图12_size](https://img-blog.csdnimg.cn/fe05ba50bae64750b922eecc9db00697.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![relplot-散点图13_size](https://img-blog.csdnimg.cn/c8ebc97dc221492192e59230c9eb4159.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


> size参数可以直接用于设置点大小，但是size数据的大小不太适合显示，比如都特别小，或者差异不明显不便于观看，怎么办？
> 把size参数直接处理肯定是可以的，但是为了显示就破坏原始数据并不好。用sizes参数可以把size映射到一个指定的区间


![relplot-散点图14_size](https://img-blog.csdnimg.cn/0001b3b8214e4035a765d5c55d1a685d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 2.4 relplot多子图绘图

根据row和col参数值，可以把图形绘制到多个子图。每行表达row数据的一个值，每列表达col数据的一个值。

> col_wrap

```
sns.relplot(data=tips,x='total_bill',y='tip',col='smoker')     #不同子图绘制smoker不同的图形，绘制在不同的行 
sns.relplot(data=tips,x='total_bill',y='tip',row='smoker',col='time') #用不同的行绘制子图区分smoker，不同的列子图区分time
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',col='time') #根据time绘制多列子图，每个子图用hue分组
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',col='day',col_wrap=2) #每列最多2个子图，超过2个自动换行
```

![rowcol1](https://img-blog.csdnimg.cn/b5b0514d39a74965871951f87d4fa3de.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
![rowcol2](https://img-blog.csdnimg.cn/3d7ac85853d44d2891cf5b553eeb71b4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
![rowcol3](https://img-blog.csdnimg.cn/88dbdaa84b0c4265889a5e2ab64623b2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
![rowcol4](https://img-blog.csdnimg.cn/1f755e7c9f2b49b688578d813b3e4eb1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 3. relplot折线图

> 折线图和散点图类似，kind参数设置为line，其他大部分参数都相同。

+ `sns.relplot(x=None,y=None,data=None,kind="scatter")`：根据kind绘制折线图或散点图。
    + 常用参数简介
        + x,y参数：x,y坐标数据。x,y为数组或者是data的key名称（data为DataFrame时，x,y指列名称）。
        + data参数：data为类字典数据。当data不为空时，x,y可以用data的key引用数据。
        + kind参数：scatter指绘制散点图，line指绘制折线图。       
    + 分组聚合参数
        + hue参数：用不同颜色对数组分组。hue可以是列表或者data的key。hue的每一个数据绘制一根曲线，给一个图例标签。
        + size参数：用不同点大小对数组分组。数组或者data中的key。每个大小给个图例标签。
        + style参数：用不同的点样式对数据分组。数组或者data中的key。每个样式给一个图例标签。
        + row,col：把row,col指定的数据按照行或列排列到不同的子图。
    + 分组聚合参数样式
        + palette参数：指定hue分组的每组曲线的颜色。
        + markers参数：设置True使用默认点样式。False不显示点。可以用列表为style设置对应点样式(marker列表长度必须和style分组数相同)。
        + dashes参数：bool设置是否使用默认线样式(实线)。用列表为style分组设置对应线样式(列表长度必须和style分组数相同)。
        + sizes参数：把size映射到一个指定的区间。
    + 折线图专用参数
        + sort参数：是否对x排序。默认为True。False则按照数组中x的顺序绘图。
        + ci参数：，当存在x值对应多个y值时，用置信区间绘制线条，默认显示95%置信区间(confidence intervals)。
            + ci=None表示不显示置信区间。
            + ci='sd'表示显示标准偏差(standard deviation)而不是置信区间。
            + ci=数值：表示指定置信区间的数值。
        + estimator参数：聚合设置，默认为平均值
            + estimator=None:不使用聚合，x对应多个y就每个x坐标绘制多个y点。
            + estimator=func：聚合函数，比如mean，sum等。
    + 其他参数
        + legend参数：图例显示方式。False不现实图例。brief则hue和size的分组都取等间距样本作为图例。full则把分组内所有数值都显示为图例。auto则自动选择brief或full。
        + height参数：每个子图的高度(单位inch)
        + aspect参数：宽度=aspect×高度

## 3.1 relplot折线图(是否自动排序绘图)

```
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
df=pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])

sns.relplot(data=df,x="x",y="y",kind="line")
sns.relplot(data=df,x="x",y="y",kind="line",sort=False)
plt.show()
```

![line1](https://img-blog.csdnimg.cn/2b81c173f2154790ad9d8c7485381866.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line2](https://img-blog.csdnimg.cn/e23422d9d2fa4481a9e5c43cecb11c86.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)


> 折线图默认会把坐标按x从小到大排序。参数sort设置为False，则不排序。


## 3.2 示例数据集

以fmri数据集为例

```python:GT
>>> fmri=sns.load_dataset('fmri')
>>> fmri
fmri
     subject  timepoint event    region    signal
0        s13         18  stim  parietal -0.017552
1         s5         14  stim  parietal -0.080883
2        s12         18  stim  parietal -0.081033
3        s11         18  stim  parietal -0.046134
4        s10         18  stim  parietal -0.037970
...      ...        ...   ...       ...       ...
1059      s0          8   cue   frontal  0.018165
1060     s13          7   cue   frontal -0.029130
1061     s12          7   cue   frontal -0.004939
1062     s11          7   cue   frontal -0.025367
1063      s0          0   cue  parietal -0.006899
```

> fmri是一组事件相关功能核磁共振成像数据。
> subject为14个测试者。timpoint为时间点。event为事件，分为刺激stim和暗示cue。region为区域，分为额叶和前叶。signal为信号。


## 3.3 relplot折线图(x,y一对多，聚类绘制)

```
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.relplot(data=fmri,x="timepoint",y="signal",kind="line")
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",ci=None)

plt.show()
```

![line3](https://img-blog.csdnimg.cn/6be72bfd643d470ab0a9f47cd805f83f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line4](https://img-blog.csdnimg.cn/e89198bb051e45a4bf2d2d08d0d83991.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line5_ci](https://img-blog.csdnimg.cn/97b7ac28fd6b417397b722ed4fc7ddbc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line6_ci_50](https://img-blog.csdnimg.cn/82b99df7bbdb49e7b8d8d645c4574286.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)


> 注意时间和信号并非一一对应，而是一个x对应多个y值。所以线条显示的是y的平均值，填充色把y的95%置信区间范围显示出来了。

对于大数据集，绘制置信区间可能需要较多时间。设置ci=None可以禁用绘制置信区间。
设置ci='sd'绘制标准差，而不是置信区间，对较大数据也是要给不错的选择。


**折线图(x,y一对多，关闭聚合，不绘制均值也也不绘制置信区间)**

estimator=None可以关闭聚合设置，直接用x,y点绘图，即每个x绘制多个y，当然这样绘制的图形可能会比较奇怪。

estimator也可以设置为聚合函数(如sum，mean，max等)，用聚合函数将一个x值对应的多个y值计算成一个值。


```
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

# sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",estimator=None) #不聚合
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",estimator=sum) #聚合函数为sum

plt.show()
```

![line8_estimator](https://img-blog.csdnimg.cn/7e9597786c104c9a860cda59fe1bf7bd.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_18,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 注意y坐标值为sum。

## 3.4 relplot分组绘图(hue、style、size)

```
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event") #hue分组
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",style="event") #style分组
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",size="event") #size分组

plt.show()
```

![line9_分组hue](https://img-blog.csdnimg.cn/288bd65d702a41568e809008aadd2e8d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line10_分组style](https://img-blog.csdnimg.cn/c46b5a85dc9c443a85e4ae6952eb8292.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line11_分组size](https://img-blog.csdnimg.cn/bba158a62ee5485598247773ec90313e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event",style="region") #hue+style
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event",size="region") #hue+size
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",style="event",size="region") #hue+size
```

![line12_分组hue+style](https://img-blog.csdnimg.cn/a1a96811e50940729e9299f066c0cb54.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line13_分组hue+size](https://img-blog.csdnimg.cn/6987dabaef1c4a3aa6fc5b0e7cffed4d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line14_分组style+size](https://img-blog.csdnimg.cn/1f8c4dff95b0440289e900c10236374e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


```
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event",style="event",size="region") #hue+style+size
```

![line15_分组hue+style+size](https://img-blog.csdnimg.cn/db2116ad03ae4f6782e9d7f534c07c37.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 3.5 relplot分组绘图样式

> hue分组可以用palette设置颜色样式
> style分组可以用markers设置点样式，用dashes设置线样式
> size分组可以用sizes设置线粗细

```python
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="region",palette="hsv") #颜色样式
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",style="region",markers=["o","*"]) #style点样式
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",style="region",dashes=[[5,1],[1,0]]) #style线样式
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",size="region",sizes=[0.5,2]) #size大小

plt.show()
```

注意：

1. dashes参数用列表时，列表长度必须和style分组个数相同。
2. dashes线样式用`[线长度,空白长度,线长度,空白长度,...]`列表形式自定义线条样式

![line16_分组样式palette](https://img-blog.csdnimg.cn/eb09d8610d734141beabd3bd1add1eba.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line17_分组样式markers](https://img-blog.csdnimg.cn/9e55c503570c48529653a96af553e5ea.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line18_分组样式dashes](https://img-blog.csdnimg.cn/a40b409bc1394e3398cd69c40cda6658.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![line19_分组样式sizes](https://img-blog.csdnimg.cn/71d85e1b5b89451fa5461409b5bbcfad.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



# 4. scatterplot散点图

scatterplot和使用relplot绘制散点图类似，主要区别：

+ 少了kind参数，以及row,col参数。
+ 多了个ax参数。

```python
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset('tips')

sns.scatterplot(data=tips,x="total_bill",y="tip")

plt.show()
```

> 因为scatterplot是axes级函数
> + 返回值是一个axes 
> + 可以用ax指定axes。
>      - 不指定ax时，会在默认的axes中绘图。
>      - 不指定ax，多次调用scatterplot绘图都会在默认（同一个）的axes中绘图。

```python
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset('tips')
fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
#ax1,ax2=fig.subplots(1,2)

sns.scatterplot(data=tips,x="total_bill",y="tip",hue='smoker',ax=ax1)
sns.scatterplot(data=tips,x="total_bill",y="tip",hue='smoker',size='sex',ax=ax2)

plt.show()
```

![scatterplot_ax](https://img-blog.csdnimg.cn/edf2b713d27a4461accaf55a4b956d64.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 5. lineplot折线图

lineplot和使用relplot绘制折线图类似，主要区别：

+ 少了kind参数，以及row,col参数。
+ 多了个ax参数。

```python
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.lineplot(data=fmri,x="timepoint",y="signal")

plt.show()
```

> 因为lineplot是axes级函数
> + 返回值是一个axes 
> + 可以用ax指定axes。
>      - 不指定ax时，会在默认的axes中绘图。
>      - 不指定ax，多次调用lineplot绘图都会在默认（同一个）的axes中绘图。

```python
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')
fig=plt.figure()
(ax1,ax2),(ax3,ax4)=fig.subplots(2,2)

sns.lineplot(data=fmri,x="timepoint",y="signal",ax=ax1)
sns.lineplot(data=fmri,x="timepoint",y="signal",hue='region',ax=ax2)
sns.lineplot(data=fmri,x="timepoint",y="signal",hue='region',style='event',ax=ax3)
sns.lineplot(data=fmri,x="timepoint",y="signal",hue='region',size='event',ax=ax4)

plt.show()
```

![lineplot1_ax](https://img-blog.csdnimg.cn/ceb496de29d34a11b5917715e18f54e0.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


<br>
<hr>


[Seaborn系列目录](https://blog.csdn.net/hustlei/article/details/123087608)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>  
>  <font color=#888>**未经允许请勿转载。**
