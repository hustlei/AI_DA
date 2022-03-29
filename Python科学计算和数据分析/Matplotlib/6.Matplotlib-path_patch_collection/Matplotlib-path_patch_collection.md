Matplotlib系列(六)：路径、块和集合
==============================



[Matplotlib系列目录](https://blog.csdn.net/hustlei/article/details/122408179)
<hr>


@[TOC]

<hr>

# 一、 简介

<font color=#888>‎matplotlib中路径(Path)、块(Patch)和集合(Collection)可以实现简单几何图形的绘制，并且可以组合成为更加高级的图形，matplotlib二维、三维中很多图形就是基于路径、块和集合实现的，比如箭头、非结构三角网格面等。

> <font color=#999>通常只需要指到预定义的块形状（圆、矩形、多边形等）的用法即可，其他了解一下，需要用的时候再查看帮助就ok。

> <font color=#999>Matplotlib系列将Matplotlib的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![matplotlib路径、块、集合](https://img-blog.csdnimg.cn/2c815c9bd09e4e2ab2f1ab58d9617d15.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


<br>

# 三、 Matplotlib路径、块、集合

## 1. 路径(Path)和块(Patch)

在Matplotlib中所有图形都是Artist对象，事实上连Figure和Axes也是Artist对象。

+ 路径(Path)是描述图形形状的线条，也就是图形的边线。
+ 块(Patch)是一种拥有边线(Path实现)和填充的Artist对象。多边形、椭圆等都是Patch对象。

> Path主要用于组成Patch。Path不能直接在图形中显示。可以用ax.add_patch()函数把Patch添加到坐标系中显示。

### 1.1 预定义基本图形patch

matplotlib预定义一部分Patch。包括

+ 圆、椭圆
+ 圆弧
+ 矩形
+ 多边形
+ 箭头等

所有预定义patch类都在`mtplotlib.patches`中。比如mtplotlib.patches.Circle。

#### 1.1.1 圆形

+ Circle(xy,radius=5)：创建圆心在xy=(x, y)处,半径为radius的圆形(正圆)。

```
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

fig = plt.figure()
ax = fig.subplots()

#circle1=Circle((2,2),1)
circle1=Circle((2,2),1,fc='C2',ec='0.5',lw=2,alpha=0.5)

ax.add_patch(circle1)
ax.axis([0,4,0,4])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

> matplotlib中很多函数同时有color,facecolor和edgecolor参数。color参数表示同时设置facecolor和edgecolor。facecolor和edgecolor优先级更高，通常设置这两个参数就ok了。

![circle patch](https://img-blog.csdnimg.cn/01507f83759540038f35f044e84a533a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.1.2 矩形

+ Rectangle(xy, width, height, angle=0.0)：创建起始角点在xy=(x, y)处,宽为width，高为height矩形，逆时针旋转angle度。

```
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

fig = plt.figure()
ax = fig.subplots()

#rect1=Rectangle((1,1),2,1)
rect1=Rectangle((1,1),2,1,fc='C3',ec='0.5',lw=2,alpha=0.5)

ax.add_patch(rect1)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```
![rectangle patch](https://img-blog.csdnimg.cn/d02ca3feae2d43eb82bd31ef5b455cd9.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

#### 1.1.3 椭圆形

+ Ellipse(xy, width, height, angle=0.0)：创建起始角点在xy=(x, y)处,宽为width，高为height矩形，逆时针旋转angle度。

```
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

fig = plt.figure()
ax = fig.subplots()

#e1=Ellipse((2,1.5),2,1)
e1=Ellipse((1,2),1.5,1,45)
e2=Ellipse((2,1.5),2,1,fc='C8',ec='C0',lw=2,alpha=0.5)

ax.add_patch(e1)
ax.add_patch(e2)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

![ellipse patch](https://img-blog.csdnimg.cn/ae66643318be4cfd90c96bf50069abf3.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.1.4 圆弧

圆弧实际上就是椭圆函数增加了theta1和theta2参数，即圆弧起止角度。

+ Arc(xy, width, height, angle=0.0)：和椭圆函数一样，绘制椭圆轮廓，不填充颜色。
+ Arc(xy, width, height, angle=0.0, theta1=30, theta2=90)：设置圆弧起止角度。

```
import matplotlib.pyplot as plt
from matplotlib.patches import Arc
fig = plt.figure()
ax = fig.subplots()

arc1=Arc((1,2),1.5,1)
arc2=Arc((2,1.5),2,1,0,30,120,ec='r')
arc3=Arc((2,1.5),2,1,theta1=200,theta2=300,color='g') #color不能缩写为c

ax.add_patch(arc1)
ax.add_patch(arc2)
ax.add_patch(arc3)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

![圆弧](https://img-blog.csdnimg.cn/274794609b1f4af3912bf718c1b217eb.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.1.5 多边形

+ Polygon(xy)：xy是n行2列的数组，即n个(x,y)点坐标。根据点绘制多边形。

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
fig = plt.figure()
ax = fig.subplots()

pts=np.array([[0,0],[1,1],[1,0]])
p1=Polygon(pts)
p2=Polygon(np.array([[0,2],[1,2],[2,3]]),color="g")
p3=Polygon(np.array([[2,2],[3,2],[3,1]]),fc="C4",ec='C1')

ax.add_patch(p1)
ax.add_patch(p2)
ax.add_patch(p3)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

![多边形](https://img-blog.csdnimg.cn/3f6db2149d1a4699953d128012b5282b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.1.6 正多边形

+ `RegularPolygon(xy, numVertices, radius=5, orientation=0)`：绘制正多边形
    + xy为多边形中心，numVertices为多边形边数
    + radius为外径半径
    + orientation为旋转角度(弧度)。0度表示顶点在正上方。

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import RegularPolygon
fig = plt.figure()
ax = fig.subplots()

p1=RegularPolygon([1,1.5],4,0.8)
p2=RegularPolygon([3,1.5],5,0.8,np.pi/2,color="g",ec='r',alpha=0.5)

ax.add_patch(p1)
ax.add_patch(p2)
ax.axis([0,4,0,3])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

![多边形](https://img-blog.csdnimg.cn/c024e41639ca4603bfb4e028205ee9ea.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.1.7 圆环

+ Annulus(xy, r, width)：中心在xy即(x,y)坐标，外径为r，宽度为width的圆环

> 内径为r-width，外径为r。

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Annulus
fig = plt.figure()
ax = fig.subplots()

a1=Annulus([1,2],0.8,0.2)
a2=Annulus([3,2],1,0.2,50, fc="0.5",ec='r',alpha=0.5)

ax.add_patch(a1)
ax.add_patch(a2)
ax.axis([0,4,0,4])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

![圆环](https://img-blog.csdnimg.cn/bcb7bc374695467483bc97ecea2e9b5b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

#### 1.1.8 楔形、扇形

+ Wedge(center, r, theta1, theta2)：圆心在center，外径为r，起止角度为theta1,theta2的扇形
    + width=None：参数width为圆环宽度，扇形内径为r-width,外径为r。

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge
fig = plt.figure()
ax = fig.subplots()

a1=Wedge([1,2],0.8,90,120)
a2=Wedge([3,2],1,45,180, 0.5, fc="0.5",ec='r',alpha=0.5)

ax.add_patch(a1)
ax.add_patch(a2)
ax.axis([0,4,0,4])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

![扇形](https://img-blog.csdnimg.cn/bce82f2d906b42bb9e9ccc615a808397.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.1.9 阴影

+ Shadow(patch, ox, oy)：为patch绘制阴影，阴影位置从shadow偏移ox,oy。

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle,Shadow
fig = plt.figure()
ax = fig.subplots()

c1=Circle([1,2],0.8)
c2=Circle([3,2],0.8,color='g')
s1=Shadow(c2,0.05,-0.05,lw=0)

ax.add_patch(c1)
ax.add_patch(c2)
ax.add_patch(s1)
ax.axis([0,4,0,4])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

![阴影](https://img-blog.csdnimg.cn/2b0d28ca21aa4c95bb4e05dc010f1b8e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 1.2 自定义patch和path

#### 1.2.1 用预定义path创建patch

patch由path和填充组成，创建path后，用matplotlib.patches中的PathPatch可以生成patch。

```
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
fig = plt.figure()
ax = fig.add_subplot()

path = Path.circle([2,2],1) #创建path
patch = PathPatch(path, facecolor='orange', lw=2) #根据path创建patch

ax.add_patch(patch)
ax.axis([0,4,0,4])
ax.set_aspect(1)
plt.show()
```

Path类支持以下方法创建预定义path：

+ Path.arc(theta1, theta2, n=None):单位圆弧
+ Path.wedge(theta1, theta2)：单位圆扇形
+ Path.unit_circle():只读单位圆
+ Path.unit_circle_righthalf()：单位圆的右半边
+ Path.unit_rectangle()：单位矩形
+ Path.unit_regular_asterisk(numVertices)：正星形，半径为1
+ Path.unit_regular_polygon(numVertices)：单位正多边形
+ Path.unit_regular_star(numVertices, innerCircle=0.5)：半径为1的星形。

> 注意上述预定义path都在坐标原点，没法直接修改，可以作为plot函数marker等参数的值。

#### 1.2.3 文本path

文本也可以创建path。文本path相比文本可以实现更多功能，比如特效、蒙版等。

+ `matplotlib.textpath.TextPath([1,40],"textpath")`：在指定位置创建文本path

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.textpath import TextPath
from matplotlib.patches import PathPatch
fig = plt.figure()
ax = fig.subplots()

t1=TextPath([1,40],"textpath")
t2=TextPath([1,25],"textpath2",8)  #指定文本大小，单位和坐标系相同
t3=TextPath([1,10],"$e^{i\pi}+1=0$",usetex=True)  #指定是否使用latex公式。

ax.add_patch(PathPatch(t1,fc='r'))
ax.add_patch(PathPatch(t2,fc='g'))
ax.add_patch(PathPatch(t3,fc='b'))
ax.axis([0,50,0,50])  #设置x,y轴范围，坐标系不会自动根据patch调整。
ax.set_aspect(1)    #x,y轴显示比例不随窗口改变。
plt.show()
```

![文本路径](https://img-blog.csdnimg.cn/8d23f713da2e4582ba759d657f111360.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.2.4 自定义path

路径Path实例实际上是一组(N,2)数组表示的(x,y)点和一个长度为N的包含命令的数组组成。

路径支持一组标准的命令，用于绘制曲线段和样条曲线，可以组成简单和复合轮廓。命令包括：moveto、lineto、curveto等。指定点坐标和命令就可以创建自定义路径。

+ path = Path(verts, codes, closed=False)：根据点序列verts，命令序列codes创建path

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch
fig = plt.figure()
ax = fig.subplots()

verts = [(0., 0.),  # left, bottom
   (0., 1.),  # left, top
   (1., 1.),  # right, top
   (1., 0.),  # right, bottom
   (0., 0.),  # ignored
]
codes = [
    Path.MOVETO,
    Path.LINETO,
    Path.LINETO,
    Path.LINETO,
    Path.CLOSEPOLY,
]
path = Path(verts, codes)
patch = PathPatch(path,fc='C4')

ax.add_patch(patch)
ax.axis([-1,2,-1,2])
ax.set_aspect(1)
plt.show()
```

![自定义path](https://img-blog.csdnimg.cn/fd549fc3238d4d35aadf7635c84d4a89.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.2.3 path特效

+ 文本可以用set_path_effects函数设置特效
+ 二维、三维绘图函数都有path_effects参数可以设置特效
+ Patch在创建的时候也可以用path_effects参数可以设置特效

**文本特效**

```
import matplotlib.pyplot as plt
from matplotlib import patheffects
fig = plt.figure()
ax = fig.subplots()

text = ax.text(0.1, 0.5, 'Hello world!', size=20)
text.set_path_effects([patheffects.Normal()])
plt.show()
```

**绘图特效**

```
import matplotlib.pyplot as plt
from matplotlib import patheffects
fig = plt.figure()
ax = fig.subplots()

ax.plot([0,3,2,4], path_effects=[patheffects.Normal()])
plt.show()
```

> Normal特效表示没有任何效果。

**常见特效**

在matplotlib.patheffects中有定义好的特效类：


+ Normal(offset=(0.0,0.0)):无任何效果绘制原图，通常用于和其他特效一起，用于显示原图。
    + offset参数表示绘制位置偏移像素个数。
+ SimpleLineShadow(offset=(2,-2),shadow_color='k',alpha=0.3,rho=0.3)：简单线条阴影
    + shadow_color参数设置阴影颜色，alpha设置阴影透明度
    + rho只有在shadow_color未设置时起作用，表示对rgb颜色用系数rho缩放得到阴影颜色。
+ PathPatchEffect(offset=(0.0,0.0))
+ SimplePatchShadow(offset=(2,-2),shadow_rgbFace=None,alpha=None,rho=0.3)：简单Patch阴影。
    + 参数同SimpleLineShadow
+ Stroke(offset=(0, 0):描边
+ TickedStroke(offset=(0,0),spacing=10.0,angle=45.0,length=1.4)：斜线填充

```
import matplotlib.pyplot as plt
from matplotlib import patheffects
from matplotlib.patheffects import Normal,PathPatchEffect, SimpleLineShadow,Stroke,TickedStroke
fig = plt.figure()
ax = fig.subplots()

#线条阴影
ax.plot([0,10,20,30],[0,5,2,10],lw=3,path_effects=[SimpleLineShadow(),Normal()])

#文本轮廓
text1 = ax.text(1,15,'text stands out',size=30,c='0.8')
text1.set_path_effects([Stroke(linewidth=1, foreground='C0')]) #描边特效

#文本填充阴影
text2 = ax.text(1, 30, 'Hatch shadow',size=40,weight=800)
text2.set_path_effects([PathPatchEffect((4,-4),hatch='xxxx'),PathPatchEffect(fc='w',lw=1)])

ax.axis([0,50,0,50])
plt.show()
```

![文本特效](https://img-blog.csdnimg.cn/7f61933525f14f22959a0dcb117c76ce.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


#### 1.2.4 path蒙版

把指定路径设置为蒙版，只显示路径内的图形。

+ image对象可以通过set_clip_path()函数，把path设置为蒙版
+ plot等绘图函数可以通过clip_path参数，把path设置为蒙版

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.text import TextPath
fig = plt.figure()
ax = fig.subplots()

textpath=TextPath((1,20),'textpath',prop={'weight':1000})
img=ax.imshow(np.random.randn(50,50),interpolation="bilinear")
img.set_clip_path(textpath,img.get_transform())

ax.axis([0,50,0,50])
plt.show()
```

![蒙版](https://img-blog.csdnimg.cn/ce8cbd9cd87342089b0ce0d08e4591df.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 2. 集合Collection

绘制大量图形时，可以使用集合Collection。

Collection类派生类的各种集合对象可以用于绘制图形集合。

+ Collection
    + LineCollection
    + CircleCollection
    + EllipseCollection
    + PolyCollection
    + RegularPolyCollection
        + AsteriskPolygonCollection
        + StarolygonCollection
    + PathCollection
    + PatchCollection

> 用线条集合可以同时绘制大量颜色位置长度不同的线条，可以绘制出非常复杂的图形，比如蝴蝶、动物等。

```
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import EllipseCollection
fig=plt.figure()
ax = fig.subplots()

ws = np.full(3,15)
hs = np.full(3,10)
angles = np.arange(3)*30
offsets= np.array([[10,10],[20,20],[30,30]])

ec = EllipseCollection(ws, hs, angles,offsets=offsets,transOffset=ax.transData)
#ws为椭圆宽度，hs为椭圆高度，angles为椭圆旋转角度，offsets为椭圆位置
ax.add_collection(ec)

ax.axis([0,50,0,50])
plt.show()
```

![集合](https://img-blog.csdnimg.cn/2c255a32561a448eaef8f850a515636e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


# 参考文章

+ [官方示例](https://matplotlib.org/stable/gallery/index.html#shapes-and-collections)
+ [官方教程](https://matplotlib.org/stable/tutorials/index.html#advanced)
+ [预定义图形api](https://matplotlib.org/stable/api/patches_api.html)

<br>
<hr>

[Matplotlib系列目录](https://blog.csdn.net/hustlei/article/details/122408179)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>
> <font color=#888>**未经允许请勿转载。**

