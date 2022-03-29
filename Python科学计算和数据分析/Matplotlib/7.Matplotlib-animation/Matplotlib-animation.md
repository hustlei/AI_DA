Matplotlib系列(七)：动画及图形修改操作
==============================


Matplotlib系列(七)：动画
==============================



[Matplotlib系列目录](https://blog.csdn.net/hustlei/article/details/122408179)
<hr>


@[TOC]

<hr>

# 一、 简介

<font color=#888>‎matplotlib的animation模块可以实现高效的动画绘制，并能够保持到gif或者视频文件中。
<font color=#888>‎matplotlib中的图形，如线条、点、坐标系、柱形图等等都可以通过代码修改，为控制图像显示，以及实现动画提供支持。


> <font color=#999>Matplotlib系列将Matplotlib的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![matplotlib动画及图形修改操作](https://img-blog.csdnimg.cn/2c815c9bd09e4e2ab2f1ab58d9617d15.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


<br>

# 三、 Matplotlib动画及图形修改操作

## 1.  手写代码更新图形实现动画

自己写代码，循环重绘图形可以实现简单的动画。

```
import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
ax = fig.subplots()

t=np.linspace(0,10,100)
y=np.sin(t)
ax.axis([0,10,0,2])
ax.set_aspect(3)

while True:
    ax.plot(t,y)
    plt.pause(0.1)  #显示图形并暂停。‎注意用‎‎time.sleep‎‎‎‎将不起作用。‎‎
    ax.cla()        #清除图形
    t+=np.pi/30     #更新数据
    y=np.sin(t)
```

> 该方法仅适用于简单、低性能的使用。
> 对于要求更高的程序，应该使用‎‎动画‎‎模块。‎

## 2. animation模块动画

### 2.1 Animation类简介

Animation类是matplotlib.animation模块中所有动画类的父类。其子类集成关系如下：

+ Animation：动画类的基类
    + TimedAnimation：继承自Animation。指定时间间隔，绘制一帧图形，实现动画
        + FuncAnimation：继承自TimedAnimation。通过重复调用fun()方法来绘制动画
        + ArtistAnimation：继承自TimedAnimation。使用一组不变的Artist对象绘制动画。

> 最常用的方法是使用FuncAnimation创建动画

### 2.2 FuncAnimation动画

+ FuncAnimation(fig, func, frames=None, init_func=None, fargs=None)
    + **fig**:用于显示动画的figure对象
    + **func**:用于更新每帧动画的函数。func函数的第一个参数为帧序号。返回被更新后的图形对象列表。
    + frames：动画长度，帧序号组成的列表
        + 依次将列表中数值传入func函数
        + frames是数值时，相当于range(frames)
        + 默认值为itertools.count，即无限递归序列，从0开始，每次加1。
        + 实际上也可以传递用户数据（类似fargs），用于更新帧。
    + init_func：自定义开始帧，即绘制初始化图形的初始化函数
    + fargs：额外的需要传递给func函数的参数。
    + **interval**：更新频率，单位是毫秒。
    + repeat：布尔值，默认为True。是否是循环动画。
    + repeat_delayint：当repeat为True时，动画延迟多少毫秒再循环。默认为0。
    + blit：选择更新所有点，还是仅更新产生变化的点。

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
fig = plt.figure()
ax = fig.subplots()

t=np.linspace(0,10,100)
y=np.sin(t)
ax.set_aspect(3)
ax.plot(t,y,'--',c='gray')
line=ax.plot(t,y,c='C2')

def update(i):  #帧更新函数
    global t    #直接引用全局变量，也可以通过函数的frames或fargs参数传递。
    t+=0.1
    y=np.sin(t)
    line[0].set_ydata(y)
    return line

ani=FuncAnimation(fig,update,interval=100) #绘制动画
plt.show() #显示动画
```

![动画](https://img-blog.csdnimg.cn/c2bcef49830c43b88994b834942df184.gif#pic_center)


> 在编写时通常会用到set_data,set_xdata,set_ydata等类似的方法更新图形数据

### 2.3 ArtistAnimation动画

ArtistAnimation(fig, artists)
    + **fig**:用于显示动画的figure对象
    + **artists**:每帧需要显示的artists列表。
    + **interval**：更新频率，单位是毫秒。
    + repeat：布尔值，默认为True。是否是循环动画。
    + repeat_delayint：当repeat为True时，动画延迟多少毫秒再循环。默认为0。
    + blit：选择更新所有点，还是仅更新产生变化的点。

```
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import ArtistAnimation
fig = plt.figure()
ax = fig.subplots()

arts=[]
t=np.linspace(0,np.pi*2,20)
for i in range(20):
    t+=np.pi*2/20
    y=np.sin(t)
    lines=ax.plot(y,'--',c='gray')  #绘制一帧图形
    arts.append(lines)              #每帧图形都保存到列表中

ani=ArtistAnimation(fig,arts,interval=200) #绘制动画
#ani.save("animate_artists_basic.gif")  #保存动画
plt.show() #显示动画
```

![动画](https://img-blog.csdnimg.cn/629439f725ea4e3d93a32bbb4a08545d.gif#pic_center)


### 2.4 保存动画

+ animation.save(filename)：‎通过绘制每一帧将动画存为影片文件。‎
    + filename参数：保存的动画文件名称，如'mov.gif','mov.mp4'。
        + 
    + writer参数：保持动画的库。MoviewWriter对象或者字符串。默认值'ffmpeg'。
        + "pillow":PillowWriter，用pillow库写如动画文件。
        + "ffmpeg":FFMpegWriter，‎基于ffmpeg库写动画。
        + "ffmpeg_file":FFMpegFileWriter，基于文件的FFMpegWriter，用ffmpeg库把帧写入临时文件，然后拼接成动画。
        + "imagemagick":ImageMagickWriter，‎基于管道的动画GIF。‎帧通过管道传输到ImageMagick并写入文件。
        + "imagemagick_file"：基于文件的imagemagick写动画。
        + "hmtl":HTMLWriter，基于javascript html的动画。
    + fps：每秒帧数，默认根据动画的interval确定
    + dpi：每英寸点数，默认和figure相同。可以控制动画大小尺寸。
    + codec：编码格式，默认'h264'
+ animation.to_html5_video()：返回html5 <video>标签，用base64文本编码直接保持。
    + embed_limit参数：动画文件大小限制，单位为MB。默认为20MB，超出限制则不创建动画。
+ animation.to_jshtml()：返回js动画，用base64文本编码。
    + fps：每秒帧数，默认根据动画的interval确定。
    + embed_frames：布尔类型，是否嵌入帧。
    + default_mode：'loop','once'或者'reflect'

```
# ani.save("movie.mp4")

writer = animation.FFMpegWriter(fps=15, bitrate=1800)
ani.save("movie.mp4", writer=writer)
```

## 3. 常用图形更新函数
所有可见图形都继承自Artist类。并且具有如下函数

+ Artist
    + draw()：重绘图形
    + get_visible(),set_visible()
    + get_alpha(),set_alpha()
    + get_zorder(),set_zorder()
    + remove():从图形中移除
    + get_children()：获取子对象列表

坐标轴具有如下常用操作

+ Axes
    + cla()：清除坐标系中的图形
    + clear()：同cla
    + set_axis_on(), set_axis_off()
    + autoscale_view()：自动缩放视窗
    + set_aspect()：设置坐标系比例

绘图函数返回的图形对象通常是Line2D、Patch、Text。常用的更新函数有：

+ set_xdata(x)：修改x数据
+ set_ydata(y)：修改y数据
+ set_data()：同时修改x,y数据。参数为(2,N)数组，或者两个一维数组。
+ plot函数中每个参数都可以用get_, set_函数操作。比如：get_linestyle(),set_linestyle()
+ set(xdata=,ydata=...)方式也可以设置参数。
    

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
