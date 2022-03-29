Matplotlib系列(五)：三维绘图
==============================


[Matplotlib系列目录](https://blog.csdn.net/hustlei/article/details/122408179)
<hr>


@[TOC]

<hr>

# 一、 简介

<font color=#888>‎matplotlib现在已经支持很多3D绘图功能了，并且也非常好用。

> 弥补了早期版本不支持3D绘图的缺憾。


> <font color=#999>Matplotlib系列将Matplotlib的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![matplotlib三维绘图](https://img-blog.csdnimg.cn/bec06940238e483ea28be52db0923f7c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


<br>

# 三、 Matplotlib三维图形

## 1. 绘制3d图形

**方法1：子图设置projection为3d**

```
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax = fig.add_subplot(projection='3d')
```

**方法2：自行创建Axes3D对象**

```
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()

ax = Axes3D(fig)
```

**三维绘图示例**

```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()

ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
# from mpl_toolkits.mplot3d import Axes3D
# ax = Axes3D(fig)   #创建3d坐标系的第二种方法

theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
x = np.sin(theta)
y = np.cos(theta)
z = np.linspace(-2, 2, 100)

ax3d.plot(x,y,z)  #绘制3d螺旋线

plt.show()
```

![简单示例](https://img-blog.csdnimg.cn/2e08be97187c486bb02a03832ce1923b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 2. 基本三维图像

### 2.1 3d折线图

在三维坐标系可以用plot函数绘制三维的线条，还可以绘制平面曲线。

+ ax3d.plot(x,y,z)：绘制三维曲线。
    + zdir参数绘制平面图
        + ax3d.plot(x,y,zdir='z')：在z=0的xy平面绘制曲线
        + ax3d.plot(x,y,2,zdir='z')：在z=2的xy平面绘制曲线
        + ax3d.plot(y,z,zdir='x')：在x=0的yz平面绘制曲线。zdir也可以为'y'
        + ax3d.plot(y,z,2,zdir='x')：在x=2的yz平面绘制曲线
    + 其他参数与二维坐标系ax.plot函数一致。注意三维plot不支持fmt参数。

> ax3d.plot3D()与ax3d.plot完全相同。

```python
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

theta = np.linspace(-2 * np.pi, 2 * np.pi, 100)
x = np.sin(theta)
y = np.cos(theta)
z = np.linspace(0.5, 1.5, 100)

ax3d.plot(x,y,z)           #绘制3d螺旋线
ax3d.plot(x,y,zdir='z')    #绘制x,y平面图形
ax3d.plot(x,y,2,zdir='z')  #绘制x,y平面图形指定高度z为2
ax3d.plot(y,z,zdir='x')    #绘制y,z平面图
ax3d.plot(y,z,-2,zdir='x') #绘制y,z平面图,指定x坐标值为-2

plt.show()
```

![3d折线图](https://img-blog.csdnimg.cn/af53b8ed59c14059bfd1deb00274ff7e.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



### 2.2  3d散点图

在三维坐标系可以用scatter函数绘制三维的散点图，还可以绘制平面散点图。

+ ax3d.scatter(x,y,z)：绘制三维散点图。
    + 平面绘制散点图
        + ax3d.scatter(x,y,zdir='z')：在z=0的xy平面绘制散点图
        + ax3d.scatter(x,y,2,zdir='z')：在z=2的xy平面绘制散点图
        + ax3d.scatter(y,z,zdir='x')：在x=0的yz平面绘制散点图。zdir也可以为'y'
        + ax3d.scatter(y,z,2,zdir='x')：在x=2的yz平面绘制散点图
    + s,c,marker,ms等参数与二维坐标系参数相同。并且都可以是列表

> 其他参数与二维plot一致。注意三维plot不支持fmt参数。

> ax3d.scatter3D()与ax3d.scatter完全相同。


```python
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

x = np.random.randn(50)
y = np.random.randn(50)
z = np.random.randn(50)
s = np.random.randn(50)*100

#ax3d.scatter(x,y,z)  #绘制3d散点图
#ax3d.scatter(x,y,z,marker=['*','o',...]) #设置不同的点样式
ax3d.scatter(x,y,z,s=s,c=s)  #绘制3d散点图
ax3d.scatter(x,y,-3,zdir='z',c='r') #3d坐标系绘制平面散点

plt.show()
```

![3d散点图](https://img-blog.csdnimg.cn/0197bcb06a05439ca203503a7fc48c0a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 2.3 3d柱形图

在三维坐标系，绘制三尾柱形图。

+ `ax3d.bar3d(x,y,z,dx,dy,dz) `：在x,y,z点绘制长、宽、高分别为dx,dy,dz的三维柱形图。
    + color参数：指定颜色；长度为N的颜色列表为每个柱形指定颜色；长度为6N的颜色列表为每个柱形的六个面(下,上,-Y,+Y,-X,+X)分别制定颜色。

> ax3d.bar函数可以在三维坐标系里不同平面上绘制一系列二维柱形图。
> 注意ax3d.bar3d是小写3d，与ax3d.bar功能是不同的。

```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

x = np.arange(5)
y = np.arange(5)
z = np.zeros(5)  #柱子底部坐标
dx=1    #柱子平面宽度
dy=1    #柱子平面深度
dz=np.random.randint(1,15,5)    #柱子高度

ax3d.bar3d(x,y,z,dx,dy,dz)  #绘制3d柱形图

plt.show()
```

![bar3d](https://img-blog.csdnimg.cn/d1625a779acf44a99c932aeed3548257.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 2.4 3d火柴图

绘制三维坐标的火柴图。

+ ax3d.stem(x,y,z)：绘制火柴图，在x,y,z坐标处会火柴头，火柴根在x,y平面，z=0坐标
+ ax3d.stem(x,y,z,orientation='x',bottom=0)：绘制火柴图，火柴根在y,z平面，x=0坐标

> 其他参数与二维stem一致。

> ax3d.stem3D()与ax3d.stem完全相同。


```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

t=np.linspace(-np.pi,np.pi,50)
x = np.sin(t)
y = np.cos(t)
z = np.linspace(-2,2,50)

ax3d.stem(x,y,z)  #绘制3d火柴图
#ax3d.stem(x,y,z,orientation="x", bottom=-2) #火柴根在yz平面

plt.show()
```

![stem3d](https://img-blog.csdnimg.cn/ae58f8d0683048f8b3786529a46867b4.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)

![stem3d-x](https://img-blog.csdnimg.cn/de0eeb69231d41bc9aafef72354d6723.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 2.5 3d误差图

+ ax3d.errorbar(x,y,z,zerr)：根据x,y,z绘制曲线，并在x,y,z处根据zerr绘制误差线。zerr可以是数值也可以是数组
+ ax3d.errorbar(x,y,z,zerr,yerr,xerr)：同时绘制x,y,z三个方向的误差线。注意是三根误差线分别绘制。
+ ax3d.errorbar(x,y,z,zerr,capsize=2,errorevery=2) ：errorevery表示每两个数据点绘制一个误差线。

> capsize, ecolor,fmt,elinewidth等参数与二维误差图相同。

> 注意，没有errorbar3D函数。

```python
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

t=np.linspace(-np.pi,np.pi,50)
x = np.sin(t)
y = np.cos(t)
z = np.linspace(-4,4,50)
zerr=np.random.randn(50)

#ax3d.errorbar(x,y,z,zerr,capsize=2)  #只有z方向误差
#ax3d.errorbar(x,y,z,zerr,0.2,0.1,capsize=2)  #同时显示zerr,yerr,xerr，注意是三个误差线
ax3d.errorbar(x,y,z,zerr,capsize=2,errorevery=2) #每两个数据点绘制一个误差线。

plt.show()
```
![errorbar3d](https://img-blog.csdnimg.cn/26b4542a1b5342a4bbd9c1b4f3909d47.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 3. 三维曲面

### 3.1  3d网格面

绘制三维网格曲面

+ ax3d.plot_wireframe(x,y,z)：x,y,z均为二维数组，根据数据绘制网格面
+ ax3d.plot_wireframe(x,y,z,rstride=2,cstride=2)：两行/列数据显示为一条线
+ ax3d.plot_wireframe(x,y,z,rcount=10,ccount=12)：设置最大显示线条数

> edgecolor, facecolor, linewidths, linestyles, capstyle, cmap等参数与二维绘图函数相同

```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

x,y=np.mgrid[-2:2:0.2,-2:2:0.2]
z = x*np.exp(-x**2-y**2)

#ax3d.plot_wireframe(x,y,z)
#ax3d.plot_wireframe(x,y,z,rstride=2,cstride=2)# 两条线合并为一条线
ax3d.plot_wireframe(x,y,z,rcount=10,ccount=12)#设置最大显示线条数

plt.show()
```

![wireframe](https://img-blog.csdnimg.cn/8c76c4e2a5b34e9da6f4f817f5dc4954.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 3.2 3d曲面

绘制三维曲面。‎默认情况下，它将以纯色的阴影着色，但它也通过提供‎‎cmap‎‎参数来支持颜色映射。‎

+ ax3d.plot_surface(x,y,z)：x,y,z均为二维数组，根据数据绘制曲面

> rcount,ccount,rstride,cstride参数和plot_wireframe相同。
> color,cmap,facecolors,edgecolor, linewidths, linestyles, capstyle等参数与二维绘图函数相同。

```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

x,y=np.mgrid[-3:3:0.2,-3:3:0.2]
z = x*np.exp(-x**2-y**2)

#ax3d.plot_surface(x,y,z)
#ax3d.plot_surface(x,y,z,rstride=2,cstride=2)# 两条线合并为一条线
#ax3d.plot_surface(x,y,z,rcount=16,ccount=18)#设置最大显示线条数
#ax3d.plot_surface(x,y,z,cmap="YlOrRd")
ax3d.plot_surface(x,y,z,cmap="YlOrRd")

plt.show()
```

![surface](https://img-blog.csdnimg.cn/a432f027a38b4e7cb5343d4c872e5614.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 3.3 3d非结构化三角网格

+ ax3d.plot_trisurf(x,y,z)：x,y,z均为一维数组，根据数据绘制三角曲面


> cmap,facecolors,edgecolor, linewidths, linestyles, capstyle等参数与二维绘图函数相同。

```python
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

x=np.random.randn(200)*2
y=np.random.randn(200)*2
z = x*np.exp(-x**2-y**2)

#ax3d.plot_trisurf(x,y,z)
ax3d.plot_trisurf(x,y,z,cmap="YlOrRd")

plt.show()
```

![trisurf](https://img-blog.csdnimg.cn/b6cb2e91f18e457cb2389ed106cae220.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 3.4 3d非结构化网格等值线

+ ax3d.tricontour(x,y,z)：x,y,z为一维数组，根据x,y,z形成非结构化网格，绘制等高线
+ ax3d.tricontour(x,y,z,zdir='x',levels=10)  #x方向等值线
    + levels, cmap等参数与二维绘图函数相同
    + offset=0参数把等值线投影到指定坐标

> ax3d.tricontourf为填充等值线。

```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系
np.random.seed(202201)

x=np.random.randn(200)*2
y=np.random.randn(200)*2
z = x*np.exp(-x**2-y**2)

#ax3d.tricontour(x,y,z)
ax3d.tricontour(x,y,z,levels=10,cmap="coolwarm")
#ax3d.tricontour(x,y,z,zdir='x',levels=10,cmap="coolwarm") #绘制x方向等值线

plt.show()
```

![tricontour](https://img-blog.csdnimg.cn/242004adcbde440c89f612c37fbb333c.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


## 4. 3d标量矢量场

### 4.1 3d等高线

+ ax3d.contour(x,y,z)：x,y,z均为二维数组。显示等高线。
+ ax3d.contour(x,y,z,zdir='x',levels=10)  #x方向等值线
    + levels, cmap等参数与二维绘图函数相同
    + offset=0参数把等值线投影到指定坐标

> ax3d.contourf(x,y,z)绘制填充等高线，用法与contour相同，填充方式仅仅是在等高线附近显示面，offset投影会比较有用。

> ax3d.contour3D()与ax3d.contour完全相同。
> ax3d.contourf3D()与ax3d.contourf完全相同。

```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

x,y=np.mgrid[-3:3:0.2,-3:3:0.2]
z=x*np.exp(-x**2-y**2)

#ax3d.contour(x,y,z)
ax3d.contour(x,y,z,levels=10,cmap="coolwarm")  #指定等高线数和颜色
#ax3d.contourf(x,y,z,levels=10,cmap="coolwarm") #填充等高线
#ax3d.contour(x,y,z,zdir='x',levels=10)  #x方向等高线

#投影
#ax3d.contour(x,y,z,levels=10,zdir='x',offset=-3)
#ax3d.contour(x,y,z,levels=10,zdir='y',offset=3)
#ax3d.contour(x,y,z,levels=10,zdir='z',offset=-0.4)

plt.show()
```

![contour3d](https://img-blog.csdnimg.cn/9bac2e63245d44d99744941126ad068d.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
3d等高线
![contour3d-x](https://img-blog.csdnimg.cn/860e6375ea2346099f3b47d53662977f.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
![contour-offset](https://img-blog.csdnimg.cn/66d254aff979479caa2ceec2dd81c144.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
等值线投影到指定位置平面

x方向等值线
![contourf3d](https://img-blog.csdnimg.cn/a37e9490d9674d32b3dee4841d01ac10.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
填充等值线，没什么意义，但是投影到x,y,z轴平面会比较好看



### 4.2 3d矢量图

+ ax3d.quiver(x,y,z,u,v,w)：在每一个x,y,z坐标绘制矢量方向为u,v,w的箭头
    + x,y,z参数可以为一维、二维、三维数组。
    + length,colors,linewidths,facecolors等参数同二维绘图函数。

> matplotlib的3d矢量图绘制功能略弱。比如不能设置uvw为箭头终点等等。


```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

t=np.linspace(-np.pi,np.pi,20)
x=np.sin(t)
y=np.cos(t)
z=np.linspace(-1,1,20)
u=np.sin(t+0.1)-x
v=np.cos(t+0.1)-y
w=0.1
ax3d.quiver(x,y,z,u,v,w)  #在每一个x,y,z坐标绘制矢量方向为u,v,w的箭头

plt.show()
```

![矢量图](https://img-blog.csdnimg.cn/47be4b4372764030bf84718232b8160a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 5. 其他

### 5.1 3d文本

+ ax3d.text(0.5,0.5,0.5,'3dtext')：在指定坐标处绘制文本
    + zdir='z'：参数可以设置文本打印方向
    + color,fontsize,fontstyle,fontweight等参数也可以设置

```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax3d = fig.add_subplot(projection='3d')  #创建3d坐标系

ax3d.text(0.5,0.5,0.5,'3dtext',c='r')   #在指定坐标处绘制文本。文本永远朝向用户
ax3d.text(0.1,0.1,0.5,'3dtextz',c='r',zdir='z')  #文本沿z轴方向打印
#ax3d.text2D(0.1,0.1,'2dtext',c='b') #好像效果和官方文档不一致。

plt.show()
```

![text](https://img-blog.csdnimg.cn/e06fc436f34e47caa6e48a2efdc5e4aa.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


### 5.2 图形旋转

+ ax3d.view_init(angle1, angle2)：以度为单位设置视角的高度角和方位角。

```
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
X, Y, Z = axes3d.get_test_data(0.1)

ax.plot_wireframe(X, Y, Z, rstride=5, cstride=5)

# rotate the axes and update
for angle in range(0, 360):
    ax.view_init(30, angle)  #设置视角
    plt.draw()  #重绘图
    plt.pause(.1)   #暂停
```

> 显示效果为360度旋转看绘制的图形，即动画。

### 5.3 三维体元素

在指定位置绘制三维体元素（通常为六面体，六面体并非必须标准形状，六个面坐标可以指定）。

+ ax3d.voxels(filled)：#filled为True的位置绘制六面体
+ ax3d.voxels(filled,facecolors=colors) #filled为True的位置绘制六面体,并设置颜色
    + facecolors：设置体元素表面颜色。
    + edgecolors：设置体元素表边颜色。
    + 注意facecolors和edgecolors颜色列表，颜色个数必须和filled数组一样。filled形状为(m,n,k)则颜色形状为(m,n,k,4)。
    + 参数color和cmap貌似都不起作用。颜色参数都不能赋值为浮点数映射colormap。

```
import numpy as np
import matplotlib.pyplot as plt

fig = plt.figure()
ax3d = fig.add_subplot(121,projection='3d')

#filled为bool类型数组，在True的元素下标位置绘制体元素
i,j,k=np.indices((3,3,3))
filled= (i==j) & (j==k)  #3行3列3层，对角线为True
c=plt.get_cmap('RdBu')(np.linspace(0,1,27)).reshape(3,3,3,4)

#ax3d.voxels(filled)             #filled为True的位置绘制六面体
ax3d.voxels(filled,facecolors=c) #filled为True的位置绘制六面体,并设置颜色

#
ax3d = fig.add_subplot(122,projection='3d')
#x,y,z=np.indices((3,4,5))
#ax3d.voxels(x,y,z,filled)

plt.show()
```

![voxels](https://img-blog.csdnimg.cn/adb1d34bc254422a983ea8252580b455.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)




+ ax3d.voxels(x,y,z,filled)：在filled为True的位置绘制六面体，用x,y,z指定各个面的位置坐标。
    + x,y,z指定六面体的起始位置，所以x,y,z的3个维度长度都必须比filled大1。


```
import matplotlib.pyplot as plt
import numpy as np
fig = plt.figure()
ax=fig.add_subplot(projection='3d')

filled = np.ones((3,3,3))
filled[0,0,0]=False
x, y, z = np.indices((4,4,4))**1.2  #x,y,z的三个维度都必须比filled大1.
ax.voxels(x, y, z, filled, edgecolors='C1')

plt.show()
```

![voxels2](https://img-blog.csdnimg.cn/e5ba8880d6634e9da7ff1ccc8f04a5a7.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)
> 注意，因为设置了x,y,z坐标位置，所以六面体的尺寸并不相同。实际上还可以更加复杂的绘制多个六面体拼成圆环、球等形状。

# 参考文章

+ [官方示例](https://matplotlib.org/stable/gallery/index.html#d-plotting)
+ [官方教程](https://matplotlib.org/stable/tutorials/toolkits/mplot3d.html)
+ [官方API](https://matplotlib.org/stable/api/_as_gen/mpl_toolkits.mplot3d.axes3d.Axes3D.html#mpl_toolkits.mplot3d.axes3d.Axes3D)

<br>
<hr>

[Matplotlib系列目录](https://blog.csdn.net/hustlei/article/details/122408179)
<hr>

> <font color=#888>个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
> 
> <font color=#888>修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>
> <font color=#888>**未经允许请勿转载。**


