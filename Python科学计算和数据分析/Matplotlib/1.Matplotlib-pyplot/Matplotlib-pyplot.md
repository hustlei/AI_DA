Matplotlib系列(一)：快速绘图
==============================



@[TOC]

<hr>

# 一、 简介

<font color=#888>‎pyplot‎‎是 matplotlib 的一个基于状态的接口。‎可以快速的绘制图表。通常我们绘图只需要通过pyplot的接口就可以了。

> matplotlib还有一个pylab接口，pylab接口实际上就是导入了pyplot接口以及numpy，scipy的一些函数。个人建议使用pyplot，自行导入numpy等库。

> <font color=#999>Matplotlib系列将Matplotlib的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![Matplotlib快速绘图](https://img-blog.csdnimg.cn/7d7664d3e7384b95955a6636ff003d06.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)




<br>

# 三、 Matplotlib快速绘图

## 1. 两种绘图方式

### 1.1 过程式绘图

```python
import matplotlib.pyplot as plt
plt.plot(x,y)            #绘制折线图
plt.show()               #显示图形
```

> 就这么简单，依次调用pyplot模块中的相应函数就可以完成绘图。

### 1.2 面向对象绘图

采用面向对象的方式，也非常方便，并且更加容易对图形进行修改控制。

```python
import matplotlib.pyplot as plt
fig=plt.figure()             #创建画布
ax=fig.subplots()            #在画布上创建一个图表
ax.plot(x,y)                 #图表上绘制线条
plt.show()                   #显示图形
```

> 画布上可以显示多个图表，所以用fig.subplots()创建图表，默认创建一个图表。
>> plt.add_axes()也可以创建图表，但是该函数必须指定范围参数，并且创建的图表是完全空白的，不建议使用。


## 2. matplotlib绘图对象层次组成

在Matplotlib中，一个图形对象由容器、图表、图表元素几层对象组成。

+ Figure：画布对象。是可以包含多个图表的容器。matplotlib中的顶层对象。
    + Text：suptitle，supxlabel，supylabel等画布级别的文本标签。
    + Axes：图表对象。指一个含坐标轴、点线图形、图例等的图表。
        + Text：title标题文本标签
        + Spine：边框，坐标轴线。
        + Axis：XAxis，YAxis坐标轴。
            + Text：label坐标轴标签
            + Tick：刻度元素
                - 主刻度、次刻度
                - 刻度值
                - 网格线
        + Legend：图例
        + Line2D、Markers等图形元素
        + Text：其他文本
    + Legend：图例（放在图表框之外的画布级图例）。


> + Figure对象的属性texts、axes、legends，以及get_children()函数可以获得子对象。
> + Axes对象的属性title、spines、xaxis、yaxis、legend_、lines、texts，以及get_children()函数可以获得子对象。
> + Axis坐标轴通常可以直接用Axes对象的方法直接操作
> + 所有可见的对象，比如Figure、Axes、Line2D、Text等都继承自Artist类。
>> Matplotlib中用Axes表示图表有点让人容易误解，可能是一个图表包含多个坐标轴吧，所有有人把Axes翻译为轴域。


![Matplotlib图形组成](https://img-blog.csdnimg.cn/img_convert/7b9fe8fd856ffba7ab513b1026e31c26.png)

> 如果上图看不到，请到[官网图形组成链接](https://matplotlib.org/stable/tutorials/introductory/usage.html)查看

## 3. Matplotlib面向对象绘图过程

### 3.1 典型代码示例

先看一个典型的示例

```python
#准备数据
import numpy as np
x=np.linspace(-np.pi,np.pi,100)
y=np.sin(x)


#导入matplotlib库pyplot模块
import matplotlib.pyplot as plt

#创建画布
fig=plt.figure()

#创建图表
ax=fig.subplots()

#绘制折线图，设置点线样式，设置线条名称
ax.plot(x,y,'+r--', label='line1',mec='b',ms=10)  #点为蓝色+(大小为10)，线为红色虚线


#设置坐标轴
ax.set_xlabel('X axis')            #坐标轴文本标签
ax.set_ylabel('Y axis')
ax.set_xticks([-4,-2,0,2,4])       #主刻度
ax.set_xticks(np.arange(-4,4,0.5),minor=True)  #次刻度
ax.set_yticks([-1,-0.5,0,0.5,1])
ax.set_yticks(np.arange(-1.5,1.5,0.1),minor=True)
ax.tick_params(axis='y',labelrotation=30)      #y轴主刻度文字旋转30度
ax.set_xlim(-3.5,3.5)              #设置显示刻度范围
ax.set_ylim(-1.5,1.5)

ax.grid(True)                      #显示主刻度网格

#设置图例
ax.legend()                  #注意需要绘图时，需指定label参数

#设置标题
ax.set_title("sample")

#保存显示图形
fig.savefig("sample.png")
plt.show()
```

绘图效果如下：

![matplotlib绘图效果](https://img-blog.csdnimg.cn/128e28d186864d67a78f2e68c56523ad.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 3.2 创建画布

+ `fig=plt.figure()`：创建画布
+ `fig=plt.figure(figsize=(6.4,4.8),dpi=100)`：指定画布大小和分辨率创建画布
    + figsize参数格式为(width,height)，单位为inch。
    + dpi参数：分辨率，指每inch像素数。默认dpi为100
    + > figsize=(6.4,4.8),dpi=100)最终得到的图像为640×480像素大小

### 3.3 创建图表

+ `ax=fig.subplots()`：在画布上创建一个图表。

> fig,ax=plt.subplots()可以创建画布并在画布上创建一个图表


> `ax=plt.add_axes([0,0,6.4,4.8])`也可以创建图表，但是该函数必须指定坐标范围，并且创建的图表没有坐标轴等对象，是完全空白的。

### 3.4 绘制图形（包含设置点、线样式、颜色）

常见的基本图形包括：

+ ax.scatter(x,y,label='name')：散点图
+ ax.plot(x,y,label='name')：折线图
+ ax.bar(x,y,label='name')：柱状图

matplotlib支持非常多的图形绘制，包括二维图形和三维图形。后续详细解释。

#### 3.4.1 点、线样式，颜色参数

> 不同的图形，样式也不太相同。

以最常见的点、线样式为主，简单介绍常用参数如下（详细参数见后续章节）：

+ `ax.plot(x,y,color='b')`可以简写为`ax.plot(x,y,c='b')`设置点和线颜色为blue。
+ `ax.plot(x,y,linestyle='-',linewidth=1.5)`：设置线样式为实线，线宽为1.5。
+ `ax.plot(x,y,marker='+',marckersize=1.5)`：设置点样式为+，大小为1.5。
    + 可以简写为`ax.plot(x,y,marker='o',ms=1.5)`
+ `ax.plot(x,y,fmt='+r-')`：用字符串参数，同时设置点样式、点线颜色、线样式
    + 可以简写为`ax.plot(x,y,'+r-')`
    + fmt的格式为`[marker][line][color]`


点的边缘颜色、填充颜色等也可以单独设置

+ `ax.plot(x,y,'o',markerfacecolor='b',markeredgecolor='r')`：设置点颜色参数
    + `markerfacecolor`设置点填充颜色,`markeredgecolor`设置点边缘颜色。
    + `markerfacecolor='b',markeredgecolor='r'`可以简写为`mfc='b',mec='r'`

#### 3.4.2 颜色参数取值

fmt参数中只能使用简化的颜色写法即：

+ `'r'`,`'g'`,`'b'`：'red','green','blue'的简写。
+ `'c'`,`'m'`,`'y'`,`'k'`：'cyan','magenta','yellow','black'的简写。
+ `'w'`：'white'的简写。

color参数可以用简写，也可以用全称。

##### 1) 基本颜色简写

+ 'r','g','b','c','m','y','k','w'
+ 'red','green','blue','cyan','magenta','yellow','black','white'

##### 2) 颜色名称

+ `color='lightblue'`：参数表示浅蓝颜色

+ Brown, Chocolate, Gold, Gray, Orange, Pink, Purple, Silver, Snow, Tomato, Yellow
+ ForestGreen, SandyBrown, SeaGreen, SkyBlue, SpringGreen, WhiteSmoke
+ GreenYellow, OrangeRed, YellowGreen
+ LightBlue , LightCyan, LightGrey, LightGreen, LightPink, LightYellow
+ LightSeaGreen, LightSkyBlue
+ MediumBlue, MediumPurple, MediumSeaGreen
+ DarkBlue, DarkCyan, DarkGray, DarkGreen, Darkorange, DarkRed, DarkSeaGreen
+ DeepPink, DeepSkyBlue, HotPink


> 以上为部分颜色名称
> + `matplotlib._color_data.CSS4_COLORS`可以获得常用的颜色名称。
> + `matplotlib.colors.get_named_colors_mapping()`可以获取所有的颜色名称。

##### 3) 灰度数值

+ `color=0.5`：参数表示灰色

用`[0,1]`之间浮点数表示灰度数据。0表示黑色，1表示白色，0.2表示深灰，0.8表示浅灰。


##### 4）十六进制RGB字符串

+ `color='#0F0'`：#RGB格式参数表示绿色
+ `color='#00FF00'`：#RRGGBB格式参数表示绿色
+ `color='#00FF0088'`：#RRGGBBAA表示半透明绿色

> 不分大小写。'#abc'与'#ABC'相同

##### 5）RGB，RGBA元组

+ `color=(0.2,0.1,0.5)`：(r,g,b)格式参数
+ `color=(0.1,0.2,0.5,0.3)`：(r,g,b,a)格式参数

> r,g,b取值范围为`[0,1]`

##### 6）C0,C1,...CN字符串

用C0,C1,...CN字符串，循环引用`rcParams['axes.prop_cycle']`中的颜色。

+ `color='C30'`

> + `matplotlib.rcParams['axes.prop_cycle']`可以获取颜色列表。
> + C必须大写。


#### 3.4.3 线样式参数取值

+ `'-'`：实线（solid line style）
+ `'--'`：虚线（dashed line style）
+ `'-.'`：点划线（dash-dot line style）
+ `':'`：点线（dotted line style）

#### 3.4.4 点样式参数取值

+ 点
    + `'.'`:点（point marker）
    + ','：像素点（pixel marker）
    + `'o'`：实心圆（circle marker）
+ 三角
    + 'v'：倒三角（triangle_down marker）
    + `'^'`：三角形（triangle_up marker）
    + '<'：左三角（triangle_left marker）
    + '>'：右三角（triangle_right marker）
+ 多边形
    + `'s'`：四边形（square marker）
    + 'p'：五边形（pentagon marker）
    + 'h'：六边形（hexagon1 marker）尖点朝上
    + 'H'：六边形（hexagon2 marker）平边朝上
    + '8'：八边形（octagon marker）
    + `'D'`：菱形（diamond marker）
    + `'d'`：瘦菱形（thin_diamond marker）上下长，左右窄
+ 符号形状
    + `'+'`：加号（plus marker）
    + 'P'：粗加号(plus (filled) marker)
    + `'_'`：减号（hline marker）
    + `'|'`：竖线/减号旋转90度（vline marker）
    + `'x'`：乘号（x marker）
    + 'X'：粗乘号（x (filled) marker）
    + `'*'`：五角星（star marker）


### 3.5 坐标轴设置

+ 设置坐标轴标签文本
    + `ax.set_xlabel('X axis')`
    + `ax.set_ylabel('Y axis')`
+ 设置主刻度坐标
    + `ax.set_xticks([-4,-2,0,2,4])`
    + `ax.set_yticks([-1,-0.5,0,0.5,1])`
+ 设置次刻度坐标
    + `ax.set_xticks(np.arange(-4,4,0.5),minor=True)`
    + `ax.set_yticks(np.arange(-1.5,1.5,0.1),minor=True)`
+ 设置x,y轴坐标刻度显示范围
    + `ax.set_xlim(-3.5,3.5)`
    + `ax.set_ylim(-1.5,1.5)`
    

**设置网格**

+ ax.grid(visible, which='major', axis='both')
    + visible参数：bool类型，True或者False
    + which参数：可选值{'major','minor','both'}
    + axis参数：可选值{'both','x','y'}
    + color参数：网格颜色
    + linestyle参数：网格线样式
    + linewidth参数：网格线宽度

**设置刻度样式**

+ `ax.tick_params(axis='y',labelrotation=30)`：y轴刻度文字旋转30度。
    + axis参数：可取值{'x', 'y', 'both'}，默认'both'
    + reset参数：bool类型。在更新参数前是否重置刻度到默认值。默认False
    + direction参数：可取值{'in', 'out', 'inout'}。刻度文字位于内部，外部，轴上。默认外部
    + color参数：刻度颜色。
    + labelsize参数：文本大小。
    + labelcolor参数：文本颜色。
    + grid_color参数：网格颜色。
    
    
### 3.6 图例

+ `ax.legend()`：显示图例。若plot为指定label参数，则图例无法显示。
+ `ax.legend(['line1','line2',...]`：指定图例各个曲线的label名称。如果已有label，则会覆盖

图例样式设置

+ `ax.legend(loc='upper right')`:自定义图例位置。
    + 'best'：重叠最小的位置，默认值
    + 'upper right'：右上
    + 'upper left'：左上
    + 'lower left'：左下
    + 'lower right'：右下
    + 'center left'：中左
    + 'center right'：中右
    + 'lower center'：中下
    + 'upper center'：中上
+ `ax.legend(fontsize='samll')`：自定义图例文字大小。
    + 相对大小（字符串）
        + 'xx-small','x-small','smal'：小于当前默认字体大小
        + 'medium'：中等
        + 'large','x-large','xx-large'：大于当前默认字体大小
    + 绝对大小（数值）
        + `fontsize=10`：绝对字体大小，单位为点。
+ `ax.legend(labelcolor=['r','b'])`：设置图例文本颜色。
+ `ax.legend(mode='expand')`：图例水平平铺。


### 3.7 图表标题

+ `ax.set_title("sample")`：设置图表标题。
+ `ax.set_title("title", loc='left')`：设置图表标题位置。
    + loc可选值为{'left','center','right'}，默认值为'center'

### 3.8 保存图形

+ `fig.savefig("sample.png")`:保存图形到文件。

支持如下格式：

+ **jpg**, jpeg：jpg图
+ **png**：png图
+ **svg**, svgz：svg图
+ tif, tiff：tiff图
+ pgf：pgf位图
+ **pdf**, eps,  ps：pdf或postscript文件
+ raw, rgba

### 3.9 显示图形

plt.show()

> 必须调用plt.show()才能显示。

## 4. 过程式绘图过程

```python
#准备数据
import numpy as np
x=np.linspace(-np.pi,np.pi,100)
y=np.sin(x)


#导入matplotlib库pyplot模块
import matplotlib.pyplot as plt

#绘制折线图，设置点线样式，设置线条名称
plt.plot(x,y,'+r-.', label='line1')   #点为蓝色+，线为红色点划线


#设置坐标轴
plt.xlabel('X axis')                  #坐标轴文本标签
plt.ylabel('Y axis')
plt.xticks([-4,-2,0,2,4])             #主刻度，不支持次刻度设置
plt.yticks([-1,-0.5,0,0.5,1], rotation=30)
plt.xlim(-3.5,3.5)                    #设置显示刻度范围
plt.ylim(-1.5,1.5)

plt.grid(True,c='gray',linestyle=':') #显示主刻度网格

#设置图例
plt.legend()                          #注意需要绘图时，指定label参数

#设置标题
plt.title("sample")

#保存显示图形
plt.savefig("sample.png")

plt.show()
```

方法参数基本上与面向对象方式一致。

## 5. 绘图数据和多子图绘图

### 5.1 字典数据绘图

```
import numpy as np
import matplotlib.pyplot as plt

data = {'x': np.arange(50),
        'y': np.random.randint(0, 50, 50),
        'color': np.random.randn(50)}

plt.scatter('x', 'y', c='color', data=data)

plt.show()
```

效果如下：

![字典数据绘图](https://img-blog.csdnimg.cn/80b51a5ee6ce4c88a788bb54f6ae6dd5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 5.2 多子图绘图

可以用如下方式创建多个子图

+ fig,axarray=plot.subplots(m,n)：一次生成m行n列子图
+ axarray=fig.subplots(m,n)：一次生成m行n列子图，返回m×n个axes对象
+ fig.add_subplot(m,n,i)：增加一个子图，m行n列放在第i个
    + 也可以用fig.add_subplots(mni)方式调用


```python
axarray=plt.subplots(2,2)
ax1=axarray[0]
ax2=axarray[1]
ax3=axarray[2]
ax4=axarray[3]
ax1.plot(x,y)
```

> 建议用fig.add_subplots()函数创建多子图。

```python
fig=plt.figure()
ax1=fig.add_subplot(2,2,1)  #2行2列，第1个子图
ax1.plot(x,y,'r-')
ax2=fig.add_subplot(223)    #2行2列，第3个子图
ax2.plot(x,y,'b:')
ax3=fig.add_subplot(1,2,2)  #跨行子图
ax3.plot(x,y,'Dg--')
plt.show()
```

![matplotlib subplot 子图跨行跨列示意图](https://img-blog.csdnimg.cn/251465141f09411d9fdab032725abd66.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


**画布级别标题和标签**

+ fig.suptitle("figtitle")：设置总标题（与所有子图平级）
+ fig.supxlabel("figxlabel")：设置总x标签（如果所有子图标签相同，可以只设置一个总标签）
+ fig.supylabel("figxlabel")：设置总y标签


**子图间距**

+ fig.subplots_adjust(wspace=0.5,hspace=0.5)：调整子图之间的间距
    + wspace：表示子图间宽度方向间隔系数
    + hspace：表示子图间高度方向间隔系数
+ fig.tight_layout(pad=1)：调整子图四周空白宽度
    + pad：四周空白宽度系数
    + w_pad：宽度方向空白宽度系数
    + h_pad：高度方向空白宽度系数

```python
fig.suptitle("figtitle", x=0.5, y=0.98)
fig.supxlabel("figxlabel", x=0.5, y=0.02)
fig.supylabel("figylabel", x=0.02, y=0.5)
fig.tight_layout(pad=2)
```

显示效果如下：


![matplotlib 子图suptitle](https://img-blog.csdnimg.cn/5fd23f4677b84ab087184b09438eb646.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


<br>
<hr>

> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
