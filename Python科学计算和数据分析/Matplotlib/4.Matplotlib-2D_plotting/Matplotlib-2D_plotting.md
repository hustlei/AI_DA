Matplotlib系列(四)：二维绘图
==============================

[Matplotlib系列目录](https://blog.csdn.net/hustlei/article/details/122408179)
<hr>


@[TOC]

<hr>

# 一、 简介

<font color=#888>‎matplotlib对在二维图绘图方面非常强大，除了散点图、曲线图、柱状图、饼图等基本图形外，还支持辅助直线、二维标量场、矢量场、统计绘图、非结构三角网格绘图以及信号谱分析。


> <font color=#999>Matplotlib系列将Matplotlib的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![matplotlib二维绘图](https://img-blog.csdnimg.cn/93369c8529cc4451a8fa4d0b27e331a2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)





<br>

# 三、 Matplotlib二维图形

## 1. 基本图形

> 每个绘图函数都有很多设置线条、点、颜色的参数，建议日常绘图时，更多的使用`plt.style.use('...')`设置样式。而不是自行设置所有参数。

### 1.1 折线图

将x,y绘制为“线”和/或“点”。可以只显示点、只显示线，或者点、线都显示。

+ 绘图函数常用调用方式
    + `ax.plot([x], y, [fmt])`
    + `ax.plot([x], y, [fmt],[x2],y2,[fmt],...)`
+ plot函数常用参数
    + c='r'：设置颜色(color)
    + ls='-'，w=1.5：设置线样式(linestyle)和线宽(linewidth)
    + marker='o'，ms=1.5：设置点样式和点大小(markersize)
    + 点颜色
        + mfc='r'：点填充颜色(markerfacecolor)
        + mec='g'：点边缘颜色(markeredgecolor)
        + mew=0.5：点边缘宽度(markeredgewidth)
    + fmt参数同时设置点和线
        + `fmt = '[marker][line][color]'`：fmt为字符串由点样式、颜色、线样式组成
        + `'+r--'`：点样式为+，颜色为红色，线样式为虚线

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.linspace(0, 4*np.pi,50)
y1=np.sin(x)
y2=np.cos(x)
y3=np.cos(x+np.pi)

#绘制'line1'。线条为灰色，线型为点划线('-.')，线宽为1
ax.plot(x,y1,c='gray', linestyle='-.', linewidth=1, label='line1')

#绘制'line2'。线条为红色，线型为虚线('--')
ax.plot(x,y2,'r--',label='line2')

#绘制'line3'。同时显示点和线。
#点大小为10，填充颜色为绿色，边缘颜色为红色，边缘宽度为1
#线为蓝色实线，线条宽度为2
ax.plot(x,y3,'ob-', mfc='g', mec='r', ms=10, linewidth=2 ,label='line3')

ax.legend(loc="upper right")#设置图例
plt.show()
```

![plot折线图](https://img-blog.csdnimg.cn/6149b2a792674bc089648354c2314bab.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 1.2 散点图

将x,y绘制为不同大小和颜色的散点图。

+ 绘图函数常用调用方式
    + `ax.scatter(x,y)`
    + `ax.scatter(x,y,s=s,c=c,alpha=0.5)`
        + s：点大小设置参数。浮点数或者列表。单位为points^2
        + c：点颜色。浮点数、颜色或者颜色列表。 
        + aplpha：点的透明度，值为0-1。

> 比较特别的，用浮点数表示颜色，表示从当前colormap映射得到的颜色。并非所有颜色参数都可以浮点数表示（比如plot就不行）。

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.random.randn(50)*10
y=np.random.randn(50)*10
s=(np.random.randn(50)*10)**2
c=np.random.randn(50)*10

#绘制散点图。颜色为c，大小为s（单位points^2），透明度为0.5
ax.scatter(x,y,s=s,c=c,alpha=0.5)

plt.show()
```

![scatter散点图](https://img-blog.csdnimg.cn/7c5b22b2f5ad4fdb8320fca1b0bbafde.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 1.3 柱状图

将y绘制成柱状图（bar函数绘制垂直柱状图，barh函数绘制水平柱状图）。

+ 绘图函数常用调用方式
    + `ax.bar(x, height)`, `ax.barh(y, width)`
    + `ax.bar((x, height ,tick_label=[...], color=c, lw=1, edgecolor=c1))`
        + x参数，bar坐标刻度参数。也是默认的坐标刻度文本。barh为y参数。
        + height参数，柱状长度参数。barh为width参数。
        + tick_label参数：坐标刻度文本，不再使用x,y作为柱标签
        + color：颜色参数
        + lw=1,edgecolor=c1：柱形图边缘宽度和颜色

> 比较特别的，用浮点数表示颜色，表示从当前colormap映射得到的颜色。并非所有颜色参数都可以浮点数表示（比如plot就不行）。

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.arange(6)
y=np.random.uniform(0,10,6)

#ax.bar(x,y) #绘制简单柱状图。用x坐标文本标签
#制定文本作为标签，填充灰色，描边宽度为1，颜色为橙色
ax.bar(x,y,tick_label=list('abcdef'),color='gray',lw=1,ec='orange') 

plt.show()
```

![scatter柱状图](https://img-blog.csdnimg.cn/940f9a8b8765419db5913e64a35c4532.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 1.4 雷达图

网上非常流行的雷达图，实际上就是极坐标下的柱状图。

```
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(202201)
ax = plt.subplot(projection='polar')

theta = np.linspace(0, 2 * np.pi, 20)
r = 10 * np.random.rand(20)
width = np.pi / 4 * np.random.rand(20)
colors = plt.get_cmap('hsv')(r/10)

ax.bar(theta, r, width=width, color=colors, alpha=0.5)

plt.show()
```

![雷达图](https://img-blog.csdnimg.cn/0edc26f9ccee46ff941d74dc17f400df.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 1.5 饼图

将y绘制成柱状图（bar函数绘制垂直柱状图，barh函数绘制水平柱状图）。

+ 绘图函数常用调用方式
    + `ax.pie(x)`:绘制简单饼状图。无标签，默认颜色设置
    + `ax.bar(x, labels=[...], colors=[...])`
        + x参数：确定饼状图比例。
        + labels参数：柱状图文本标签。
        + colors参数：饼状图的颜色。
        + 炸开设置
            + `explode=[0,0.2,0,0]`：第i个数字表示第i个饼炸开距离（数值×半价=距离）
                + explode参数的长度必须和x参数相同，explode第0个数值貌似没有用
        + 边线
            + `wedgeprops={'lw':2,'ec':'lightblue'}`:设置边线参数，lw、ec是linewidth，edgecolor的缩写。
            + `wedgeprops={'width':0.2}`：环形饼图，环宽度与半径比例为0.2

> frame参数设置是否显示坐标系，默认False，与plt.axis('off')相同。

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.arange(6)

#ax.pie(x) #绘制简单饼状图。无标签，默认颜色设置
ax.pie(x,
       labels=list('abcdef'),             #设置标签
       explode=[0,0,0.1,0,0,0],           #第2个饼图炸开
       wedgeprops={'lw':2,'ec':'lightblue'})  #描边
#plt.axis('off')  #不显示坐标轴

plt.show()
```

![饼图](https://img-blog.csdnimg.cn/67d9d5bc634646fca2a85a5491ff541a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

### 1.6 环状饼图及嵌套饼图

通过在pie函数的wedgeprops(字典)参数中设置`'width':0.2`，可以创建环形饼图，环宽度与半径比例为0.2

```
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(202201)
fig, ax = plt.subplots()

val1 = np.abs(np.random.randn(3))
val2 = np.abs(np.random.randn(5))
cmap = plt.colormaps["tab20c"]
outer_colors = cmap(np.arange(3)*4)
inner_colors = cmap([9,1,3,5,2])

ax.pie(val1, radius=1, colors=outer_colors,wedgeprops=dict(width=0.3, edgecolor='w'))
ax.pie(val2, radius=0.7, colors=inner_colors,wedgeprops=dict(width=0.3, edgecolor='w'))

ax.set_aspect(1)
plt.show()
```

![环状饼图](https://img-blog.csdnimg.cn/98ac2af04f504be9b8386d8870d5d679.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 1.7 火柴图

将x,y绘制成火柴图。

+ 绘图函数常用调用方式
    + `ax.stem(x,y)`
    + `ax.stem(x,y,linefmt='-',markerfmt='C0o')`
        + linefmt参数：设置线型，可以选'-','--','-.',':'
        + markerfmt参数：设置点样式和颜色。字符串，颜色智能是简写，或者C0,C1,...CN 

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.arange(6)
y=np.random.uniform(1,10,6)

#ax.stem(x,y)
ax.stem(x,y,
        linefmt="--",
        markerfmt="C2o")

plt.show()
```

![火柴图](https://img-blog.csdnimg.cn/19032a58b965494fb1f599eca95e116e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 1.8 曲线填充图

fill_between在(x,y1)和(x,y2)曲线之间填充颜色。
fill_betweenx在(x1,y)和(x2,y)曲线之间填充颜色。

+ `ax.fillbetween(x,y1,y2)`:在y方向y1,y2之间填充
+ `ax.fill_betweenx(y,x1,x2)`:在x方向x1,x2之间填充
    + y2,x2可以省略，默认为0
    + ax.fillbetween(x,y1,y2,where=y2>y1)：根据条件填充，只填充y2>y1的区域

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.linspace(0,2*np.pi,200)
y1=np.sin(x)
y2=np.cos(x)

ax.fill_between(x,y1,y2,color='C2')  #填充y1,y2之间的区域
#ax.fill_between(x,y1,y2,where=y2>y1) #只填充y2>y1的区域

plt.show()
```

![](https://img-blog.csdnimg.cn/050d6a08721948659bc6439f0608fe05.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 1.9 面积图

对多个曲线(x,y1,y2,y3...)依次叠加绘制面积填充图。

+ `ax.stackplot(x,y1,y2,...)`:y1,y2,y3...长度和x相同
+ `ax.stackplot(x,y)`:y为m行长度为len(x)的数组。即y相当于`[y1,y2,y3...]`
    + colors参数：颜色列表，填充颜色循环使用colors列表中的颜色
    + labels参数：设置各个曲线标签，用于图例
    + baseline参数：基线‘zero','sym',or 'wiggle'(斜率平方和最小)

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

#准备数据
x=np.linspace(0,2,30)
y1=np.full(30,0.2)
y2=np.full(30,0.2)
y3=np.full(30,0.3)
y4=(x+0.5)**1.1

ax.stackplot(x,y1,y2,y3,y4)
#ax.stackplot(x,y1,y2,y3,y4,baseline="sym",colors=['r','b'])

plt.show()
```

![面积图](https://img-blog.csdnimg.cn/f178500bfbcb43e38df09ac75de2ad8c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 2. 辅助直线

### 2.1 直线

+ `axhline(y=0,xmin=0,xmax=1)`：水平线，起始位置(xmin,xmax)
+ `axvline(x=0,ymin=0,ymax=1)`：垂直线，起始位置(ymin,ymax)
+ `axline((0,0),(1,1))`：通过两点的无限长直线

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

ax.axhline(y=1)
ax.axvline(x=1)
ax.axhline(y=3,xmin=0.1,xmax=0.8,c='r')
ax.axvline(x=3,ymin=0.1,ymax=0.8,c='r')
ax.axline((0,0),(1,1))

ax.set_xlim(0,5)
ax.set_ylim(0,5)

plt.show()
```

![直线](https://img-blog.csdnimg.cn/09e626f5c54d42bebbdb5cd712cb4255.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 2.2 一组直线

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

ax.hlines([1,2,3],0,4)
ax.vlines([3,2,1],0.5,4)

ax.set_xlim(0,5)
ax.set_ylim(0,5)

plt.show()
```

### 2.3 填充线

+ `ax.axhspan(ymin, ymax, xmin=0, xmax=1)`：绘制高度范围ymin到ymax的竖线
+ `ax.axvspan(xmin, xmax, ymin=0, ymax=1)`：绘制宽度范围xmin到xmax的水平线

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

ax.axhspan(1,2)
ax.axvspan(2,3)

ax.set_xlim(0,5)
ax.set_ylim(0,5)

plt.show()
```

![填充线](https://img-blog.csdnimg.cn/ae50b9daa70a41b7b8ca7410e9a726bf.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 3. 二维标量场

### 3.1 imshow

数据显示为一个二维图片。(颜色填充二维网格)

+ `ax.imshow(X)`：X数据绘制为图像
    + X为(M,N,3)数据时，M行N列栅格图像，每个格子用RGB颜色填充
    + X为(M,N,4)数据时，M行N列栅格图像，每个格子用RGBA颜色填充
    + X为(M,N)数据时，M行N列栅格图像，每个格子用colormap颜色绘制
    + interpolation='bicubic'：插值方式，可用'nearest','bilinear','bicubic','spline16','spline36','hermite','bessel','gaussian'等。
    + aplha=0.5：设置透明度
    + origin='lower'：默认最上方为第一行数据。lower则第一行数据在最下方显示。默认为upper

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

X=np.random.randn(9,9)
ax.imshow(X)

ax.axis('off')
plt.show()
```

![imshow](https://img-blog.csdnimg.cn/ab4d5275865e4d9fac39f8f3acb869f9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 3.2 pcolor

伪彩色图片显示。功能类似imshow，但是不需要x,y坐标等间距。

+ `ax.pcolor([X,Y],Z)`：X,Y省略时，与C参数与imshow的参数用法接近
    + `ec='w'`：设置edgecolor参数。色块之间显示边框线。
    + `alpha=0.5`：设置色块透明度

pcolor与imshow的不同：

+ pcolor相比imshow更加灵活，不需要x,y等间距，因为可以设置x,y值。
+ pcolor没有origin参数，第一行数据显示在在下边。相当于imshow的origin='lower'
+ pcolor不支持interpolation参数

> pcolormesh与pcolorfast比pcolor更快，参数一致


```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x,y=np.mgrid[0:9,0:9]
c=np.random.randn(9,9)
ax.pcolor(x,y,c)
#ax.pcolor(c) #注意，因为mgrid的顺序问题，与不省略x,y绘制的图颜色顺序不同。

ax.axis('off')
plt.show()
```

![pcolor](https://img-blog.csdnimg.cn/29cf74bcedc24fd29730e7055bdff67e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)




### 3.3 contour、contourf等值线

绘制等值线，或者等高线。

+ `ax.contour([X,Y],Z)`：根据Z值绘制等值线，也就是等高线。
    + `levels=5`：设置等高线数量，也可以用列表直接指定，比如`levels=[1,2,3]`。
    + origin='lower'：设置`z[0,0]`在左下角。'upper'设置`z[0,0]`在左上角。
    + `colors和cmaps参数`：两个参数不能同时设置，会出错。
        + 默认根据cmaps的colormap指定颜色。
        + colors指定后cmaps失效。colors为颜色值或者颜色列表

> contourf与contour用法完全一致，contourf会在等高线间填充颜色。

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x,y=np.mgrid[-1:1:0.1,-1:1:0.1]
z=x**2+y**2
#ax.contour(x,y,z)
#ax.contourf(x,y,z)
#ax.contour(x,y,z,levels=15,origin='upper')
ax.contourf(x,y,z,levels=15,origin='upper')

plt.show()
```

![contour](https://img-blog.csdnimg.cn/a9bc4c9e46154a538d258fd367080dcc.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 4. 二维矢量场

### 4.1 quiver箭头(速度)图

绘制表示矢量场的箭头图。比如速度场就是典型的矢量场

+ `ax.quiver([X,Y],U,V)`：在x,y位置绘制箭头，箭头x,y方向的大小分量分别为u,v。。
    + U,V：分别为x,y轴方向的速度场大小分量。
    + C参数：映射colormap颜色的数组。C的值数量必须与箭头数相同。
    + color参数：与C不同，只能设置颜色或颜色列表，不能用数值表示。
    + 箭头参数：
        + `angle='uv'`：箭头方向参数。默认u表示x轴分量，v表示y轴分量。'xy'表示箭头方向为(x,y)指向(x+u,y+v)
        + `units='width'`：箭头大小单位，默认取系数×width。
            + 'width','height'：参数取这几个值分别表示宽度、高度作为系数基准。
            + 'x','y','xy'：表示x,y轴单位及x,y平方根作为基准。

> headwidth,headlength等参数可以调整箭头头部大小。

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)
plt.style.use('ggplot')

x,y=np.mgrid[-1:1:0.1,-1:1:0.1]
u=x+y
v=y-x
#ax.quiver(x,y,u,v)
ax.quiver(x,y,u,v,units='xy')

plt.show()
```

![速度图](https://img-blog.csdnimg.cn/1a18660a767c4b1db48de64dc6ff9931.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 4.1 streamplot流线图

绘制表示矢量场的流线图。

+ `ax.streamplot(x,y,u,v)`：x,y为坐标参数，u,v为矢量方向分量参数。
    + density=0.5参数：流线密度，默认为1。用desity=(0.5,1)表示不同方向密度不同。
    + color参数：可以为颜色值，也可以为浮点数数组。浮点数数组表示映射colormap的值。
    
> linewidth, arrowsize, arrowstyle等参数可以设置流线细节，需要用的时候再查API就ok

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

y,x=np.mgrid[-1:1:0.1,-1:1:0.1]  #streamplot要求x每行必须相同
u=x+y
v=y-x
#ax.streamplot(x,y,u,v)
ax.streamplot(x,y,u,v,density=0.5, color=np.random.randn(20,20))

plt.show()
```

![流线图](https://img-blog.csdnimg.cn/0de4c1b911804260bc03264d36e28448.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 5. 统计绘图

### 5.1 直方图

统计数据分布绘制直方图。

+ `ax.hist(x)`：数据x按若干个区间统计分布个数，绘制成直方图。
    + bins=10：参数设置x分成10个区间
    + density=True：默认统计区间内数值个数，density=True统计数值个数占len(x)的比例。
    + histtype='bar'：直方图样式，可取'bar','barstacked','step','stepfilled'
    + orientation='horizontal'：水平直方图或者垂直直方图
    + color="C1"：设置颜色，多个直方图绘制时，可以为颜色列表。
    + 多个直方图绘制在同一个图中
        + stacked=True：多个直方图叠加方式绘制
        + label：为每个直方图设置标签名称

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(1000)
#ax.hist(x)
ax.hist(x,bins=8,ec='w',lw=1,color="C1") #设置分组数，设置edgecolor和linewidth

plt.show()
```

![直方图](https://img-blog.csdnimg.cn/6d067284b972480abd6a0c406c8078c2.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 5.2 hist2d二维直方图

统计二维数据分布绘制在平面上，用颜色表示分布数量。

+ `ax.hist2d(x,y)`：数据x,y分布按若干个区间统计数据分布个数，绘制成平面图，用颜色表示分布数量。
    + bins=10：x,y轴都分成10个区间，`bins=[5,5]`分别设定x,y轴区间数。
    + ec='w',lw=1,density=True：同hist
    + cmap：设置colormap

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(1000)
y=np.random.randn(1000)
#ax.hist2d(x,y)
h=ax.hist2d(x,y,bins=20,cmap="Oranges") #设置分组数
#ec='w',lw=1,density=True)。设置edgecolor和linewidth

plt.colorbar(h[3],ax=ax)
plt.show()
```

![hist2d](https://img-blog.csdnimg.cn/77d51ad504ec4dfcb5d202cbe0a655c1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 5.3 boxplot箱型图

根据数据x的每一列数据的四分位数（最小、25%，中位数、75%、最大，分别称为Q0-5）值绘制箱型图。

> boxplot函数与标准的箱型图略有不同，Q1-1.5(Q3-Q1)，Q5用Q3+1.5(Q3-Q1)。超出Q0-Q5范围的值单独用o表示。

+ `ax.boxplot(x)`：绘制x每一列的箱型图
    + sym="r+"：超出Q0-Q5范围的值的绘制颜色和符号。
    + whis=1.5：`Q0=Q1-whis*(Q3-Q1)`，`Q5=Q3+whis*(Q3-Q1)`
    + showfliers=False：不显示超出Q0-Q5范围的点。


```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(100,5)  #5列，所以绘制5个箱型图
#ax.boxplot(x)
ax.boxplot(x,sym="r+",whis=1.25) #设置分组数

plt.show()
```

![boxplot](https://img-blog.csdnimg.cn/2412f174b9c0475e960e3bca8a262b8c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 5.4 errorbar误差图

在坐标x,y处，根据yerr绘制上下误差。

+ `ax.errorbar(x,y,yerr)`：类似plot绘制x,y折线或散点图，并根据yerr绘制误差。
    + fmt="+"：设置线条及点样式。格式同plot函数，即`"[markerstyle][color][linestyle"`
    + lw=2：设置linewidth
    + c="C2"：设置color
    + capsize=4：误差线，上下端短线长度。

```
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x = np.arange(-5,5,0.5)
y = np.sin(x/2)
yerr = np.random.randn(20)/5
#ax.errorbar(x,y,yerr)
ax.errorbar(x,y,yerr,fmt='o-',lw=2,capsize=4,c="C2") #设置x,y样式及线宽，设置误差样式

plt.show()
```

![errorbar](https://img-blog.csdnimg.cn/9e8cbeeb186f41548d14c3919b0c4f53.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 5.5 hexbin填充六边形统计图

在坐标x,y处，根据yerr绘制上下误差。

+ `ax.hexbin(x,y)`：类似hist2D，对x,y坐标轴分为若干个区间，统计区间内数值出现次数，绘制六边形颜色表示。
    + gridsize=20：x,y坐标轴分为20份。gridsize=(5,10)分别设置x,y坐标轴六边形数量。
    + cmap：设置colormap
    + edgecolors='w'：指定六边形边缘颜色。


```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)
plt.style.use('_mpl-gallery-nogrid')

x = np.random.randn(1000)
y = x+np.random.randn(1000)/5
#ax.hexbin(x,y)
h=ax.hexbin(x,y,gridsize=20,edgecolors='w')

plt.colorbar(h)
plt.show()
```

![hexbin](https://img-blog.csdnimg.cn/ed9333c1a08c49e6a771fa9235fa72f5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 6. 非结构化三角网格坐标

‎对于(x,y)坐标处的高度z，可以使用本类函数绘制等值线，并填充三角面。而不需要先网格化数据，然后使用contour函数绘制‎‎等值线。

> contour函数需要x,y坐标需要像meshgrid输出一样，结构化的。

### 6.1 tricontour非结构三角网格等值线
 
‎在非结构化三角形网格上绘制等值线。‎

+ ax.tricontour(x,y,z)：为x,y,z组成的非结构网格绘制等值线
+ ax.tricontourf(x,y,z)：绘制等值线并且填充颜色
    + levels=8：等值线数量
    + cmap="Oranges"：colormap参数。
    + colors:颜色参数，不能和cmap同时使用

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(100)
y=np.random.randn(100)
z=x**2+y**2
ax.plot(x, y, 'o', markersize=2, color='lightgrey')
#ax.tricontour(x,y,z)
ax.tricontourf(x,y,z,levels=8,cmap="Oranges")#colors和cmap不能同时设置

plt.show()
```
 
![tricontour](https://img-blog.csdnimg.cn/5ee49b54accd4403a70cf147c030a7e1.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 6.2 tripcolor三角网格伪彩色图

为非结构化三角网格绘制伪彩色图。

+ ax.tripcolor(x,y,z)

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(100)
y=np.random.randn(100)
z=x**2+y**2
ax.plot(x, y, 'o', markersize=2, color='lightgrey')
ax.tripcolor(x,y,z)

plt.show()
```

![tripcolor](https://img-blog.csdnimg.cn/6abc40e7f212487696493036ff2e7077.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 6.3 triplot非结构三角网格plot

把非结构三角网格绘制成曲线或者散点图。

+ ax.triplot(x,y)：所有参数和ax.plot完全相同

> 不需要z参数

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x=np.random.randn(100)
y=np.random.randn(100)
#ax.triplot(x,y)
ax.triplot(x,y, 'o-', markersize=2, color="C0", mfc='lightgrey',mec='lightgray')

plt.show()
```

![triplot](https://img-blog.csdnimg.cn/4bb99c2f0efb4e88813b8654052198f5.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 7. 信号谱分析

> 频谱相关

### 7.1 acorr自相关

绘制x的自相关图。

> 自相关指，序列x，前后自己相关。即{Xt}和{Xt+k}之间的相关系数。
> 从两个随机变量之间相关系数，变为了一个随机变量时间前后（滞后k期）相关系数。

+ `ax.acorr(x)`：绘制x的自相关图，默认最大滞后期为10
    + maxlags=20：设置最大滞后期数值

自相关图横坐标为滞后期，即k，竖轴为相关系数。

> 相关系数为-1~1之间浮点数。-1为负相关，1为正相关，0为不相关。

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

t=np.linspace(0,8*np.pi,100) #4个周期
x=np.sin(t)
#ax.acorr(x)
ax.acorr(x,maxlags=25)

plt.show()
```

![acorr](https://img-blog.csdnimg.cn/1bc573b5272b4f3b88ad80a7e9b18ddb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 7.2 xcorr互相关图

‎绘制‎‎x‎‎和‎‎y‎‎之间的互相关图。‎

> 相关系数`x[n+k]`与`y[n]`计算，k为滞后期

+ `ax.xcorr(x,y)`：绘制x,y的相关图，默认最大滞后期为10
    + maxlags=20：设置最大滞后期数值

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

t=np.linspace(0,8*np.pi,100) #4个周期
x1=np.sin(t)
x2=np.cos(t)

#ax.xcorr(x1,x2)
ax.xcorr(x1,x2,maxlags=25)

plt.show()
```

![xcorr](https://img-blog.csdnimg.cn/e63ad9a85b3a4b3aaa491ecd48473b39.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 7.3 psd功率谱密度

绘制功率谱密度。

PSD——Power Spectral Density 是表征信号的功率能量与频率的关系的物理量，单位频带的信号功率就被称之为功率谱。。PSD经常用来研究随机振动信号。


> 功率谱与频谱不太一样。
> + 功率谱的计算需要信号先做自相关，然后再进行FFT运算。
> + 频谱的计算则是将信号直接进行FFT就行了。
> + 功率谱是对信号研究，不过它是从能量的方面来对信号研究的。

+ `ax.psd(x)`

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

t=np.linspace(0,8*np.pi,100) #4个周期
x1=np.sin(t)

#ax.xcorr(x1,x2)
ax.psd(x1)

plt.show()
```

![psd](https://img-blog.csdnimg.cn/65f09d5df110474a82064900c5108348.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 7.4 CSD两个信号的交叉频谱密度

‎计算两个信号的交叉频谱密度‎（cross spectral density）

交叉谱，或者交叉频谱用来表征两个序列（时间相关）在某个（某些）频率组分上的相关程度，取值范围是0-1。



+ `ax.csd(x,y)`

```
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

t=np.linspace(0,8*np.pi,100) #4个周期
x1=np.sin(t)
x2=np.cos(t)

ax.csd(x1,x2)

plt.show()
```

> 功率谱密度是功率沿频率轴的分布。信号的PSD是信号的傅立叶变换的自相关

互谱密度相同，但是使用互相关，因此您可以使用其平方模块找到两个信号在给定频率下共享的功率，并使用其自变量找到该频率下两个信号之间的相移。

互谱密度可以用于识别嘈杂的LTI系统的频率响应：如果噪声与系统的输入或输出不相关，则可以从输入和输出的CSD中找到其频率响应。


![csd](https://img-blog.csdnimg.cn/c54c1b352d684369acda6af6016fef33.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



<br>
<hr>

[Matplotlib系列目录](https://blog.csdn.net/hustlei/article/details/122408179)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>
> <font color=#888>**未经允许请勿转载。**
