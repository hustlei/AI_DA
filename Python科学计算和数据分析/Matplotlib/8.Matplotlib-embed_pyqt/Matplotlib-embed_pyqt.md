Matplotlib系列(八)：嵌入Python Qt界面
==============================




[Matplotlib系列目录](https://blog.csdn.net/hustlei/article/details/122408179)
<hr>


@[TOC]

<hr>

# 一、 简介

<font color=#888>‎matplotlib可以很容易嵌入PyQt、GTK、Tk、wxPython以及Web中。这里只介绍嵌入PyQt(PySide)。


> <font color=#999>Matplotlib系列将Matplotlib的知识和重点API，编制成思维导图和重点笔记形式，方便记忆和回顾，也方便应用时参考，初学者也可以参考逐步深入学习。

# 二、 思维导图

![matplotlib嵌入qt](https://img-blog.csdnimg.cn/f59c4eecc956483898625d30300b228b.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



<br>

# 三、 Matplotlib嵌入PyQt界面

## 1. Matplotlib嵌入GUI基本方法

**matplotlib嵌入GUI时，首先需要确定界面渲染引擎，并导入渲染引擎对应的画布**

以Qt(PyQt或PySide)为例：

```
import matplotlib
matplotlib.use('QtAgg')  #指定渲染后端。QtAgg后端指用Agg二维图形库在Qt控件上绘图。
from matplotlib.backends.backend_qtagg import FigureCanvas
#使用matplotlib中的FigureCanvas(继承自QtWidgets.QWidget)绘制图形可以嵌入QT GUI
```

> 创建FigureCanvas时需要传入一个figure作为参数用于绘图：`canvas=FigureCanvas(figure)`

## 2. matplotlib界面渲染引擎

matplotlib显示图形实际上通过第三方库作为后端实现的，即调用后端的接口渲染图形。

matplotlib有多种后端，并且很多后端还有多种可选的UI引擎：

### 2.1 可用的渲染引擎后端

可用的后端简介如下（后端名称不区分大小写）

+ agg（Anti-Grain Geometry）：一个C++高保真2D渲染引擎
    + GTK3Agg, GTK4Agg：用agg在GTK Canvas上渲染
    + QtAgg, Qt5Agg：用agg在qt Canvas上渲染
    + TkAgg：用Cairo在Tk Canvas上渲染
    + WXAgg：用agg在wxWidgets Canvas上渲染
    + WebAgg：用html渲染
    + nbAgg(Ipython notebook)
+ cairo：一款开源的2d矢量图形库。Python中用cairocffi或pycairo调用
    + GTK3Cairo，GTK4Cairo：：用Cairo在GTK Canvas上渲染
    + QtCairo, Qt4Cairo, Qt5Cairo：：用Cairo在Qt Canvas上渲染
    + TkCairo：用Cairo在Tk Canvas上渲染
    + WXCairo：用Cairo在wxWidgets Canvas上渲染
+ WX：用wxWidgets在wxWidgets Canvas上渲染
+ MacOSX：用Cocoa在OSX windows上渲染
+ 用于生成图片及文件的后端
    + pdf
    + pgf
    + ps
    + svg
    + template
    
> 以上后端名称都可以用于matplotlib.use函数

### 2.2.渲染引擎后端设置

有三种设置后端引擎的方法：

+ `matplotlib.use('agg')`
+ `rcParams["backend"] = 'agg'`
+ `MPLBACKEND`环境变量
 
> 默认后端为'agg'。后端名称字符串不分大小写。

对于qt。QtAgg和QtCairo后端同时支持Qt5和Qt6。Qt5Agg同时支持PyQt5和Pyside2。
那么matplotlib会使用pyqt还是pyside作为引擎呢？

+ ‎如果已经加载了任何绑定（PyQt5,PyQt6,PySide2,PySide6），则它将用于Qt后端。‎
+ ‎如果还没有加载了任何绑定，则从`QT_API环境变量`确定。
    + `QT_API环境变量`可以设置为PyQt6,PySide6,PyQt5,PySide2,PyQt4或PyQt4v2。不分大小写
+ 如果`QT_API环境变量`的后端不存在，则自动尝试导入
    + PyQt6
    + PySide6
    + PyQt5
    + PySide2

## 3. 可嵌入GUI的画布

### 2.3.1 画布及backend模块

Matplotlib还将后端渲染器（renderer）和画布（canvas）分离开来，以实现更灵活的定制功能。画布就是一个GUI的控件，可以直接嵌入GUI中。

在Matplotlib安装目录的“backends”子目录里是这些后端的模块文件，例如有

+ backend_agg
    + backend_qtagg
    + backend_qt5agg（实际上和backend_qtagg一样，只是为了兼容旧API。）
    + backend_webagg
    + backend_tkagg
    + backend_gtk3agg
    + backend_gtk4agg
    + backend_wxagg
+ backend_cairo
    + backend_qtcairo
    + backend_tkcairo
    + backend_gtk3cairo
    + backend_gtk4cairo
    + backend_wxcairo
+ backend_wx
+ web_backend 
+ backend_macosx
+ backend_nbagg
+ backend_mixed
+ 其他
    + backend_pdf
    + backend_pgf
    + backend_ps
    + backend_svg
    + backend_template

### 2.3.2 qt创建画布的方法

推荐做法：

```
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvas #FigureCanvas是FigureCanvasQTAgg的别名。

fig = plt.figure()
cav = FigureCanvas(fig)  #创建matplotlib画布
```

> from matplotlib.backends.backend_qt5agg import FigureCanvas实际上和上边代码效果一样，同样能用于qt6。只是为了兼容旧API。

FigureCanvas类继承自QWidget，所以，FigureCanvas就是个Widget组件，可以直接嵌入GUI窗体上。

> 基于qt的后端还有backend_qtcairo、backend_qt5agg、backend_qt、backend_agg。通常建议使用backend_qtagg
> + backend_qtcairo需要安装pycairo或cairocffi
> + backend_qt5agg实际上就是从backend_qtagg导入的，和backend_qtagg完全相同，并非针对qt5的，主要为了兼容老版本
> + backend_qtagg同时继承了backend_qt和backend_agg。单独使用backend_qt和backend_agg会有很多功能缺失，甚至不能绘图，需要自行写很多代码

## 3. Matplotlib嵌入Qt

### 3.1 Matplotlib嵌入Qt GUI示例


```
import sys
import numpy as np
from qtpy.QtWidgets import QApplication #qtpy会以PyQt5,PyQt6,PySide2,PySide6的顺序依次尝试import。
from matplotlib.backends.backend_qtagg import FigureCanvas
import matplotlib.pyplot as plt

app=QApplication(sys.argv)     #创建QApplication

fig = plt.figure()                        #创建figure
ax = fig.subplots()
t = np.linspace(0,np.pi,50)
ax.plot(t,np.sin(t))                     #画曲线（在窗口显示之后画也可以）
win=FigureCanvas(fig)             #创建画布控件
win.show()                              #画布控件作为窗口显示

sys.exit(app.exec())                  #启动App事件循环
```

注意，示例中并没有使用matplotlib.use指定qt绑定库版本。因为我们从backend_qtagg导入了FigureCanvas，所以matplotlib自动将后端设置为qtagg，所以不需要再重复设置了。

> 一般情况下，matplotlib嵌入GUI的时候都不需要再调用matplotlib.use。
> 在不需要嵌入GUI，安装了多个GUI库，又想指定后端的时候可以显示调用matplotlib.use。

![在这里插入图片描述](https://img-blog.csdnimg.cn/537c20fba91c4e3a9f1735aa36e59fd8.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


> qtpy是一个把PyQt和PySide多个版本api抽象为统一接口的小型库。
> qtpy导入Python Qt绑定库的规则为：
> + 如果已经导入了PyQt6,PyQt5,PySide6或PySide2则直接使用。
>     - 如果FORCE_QT_API环境变量为True，则优先用`QT_API环境变量`确定。
> + 如果没有已经导入的库，则从`QT_API环境变量`确定。
>     - 环境变量可以是PyQt6,PyQt5,PySide6或PySide2（不区分大小写）。
>     - 如果环境变量指定的库没有安装则出错。
> + 如果没有没有设置环境变量，则以PyQt5,PyQt6,PySide2,PySide6的顺序依次尝试import。

> 使用from matplotlib.backends.compat import QtWidgets能达到from qtpy import QtWidgets类似的效果。




### 3.2 Matplotlib画布工具栏

每个backend后端都定义了一个NavigationToolbar类，可以用于显示matplotlib图形的工具栏。

```python
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar 
...
toolbar=NavigationToolbar(figurecanvas, parent)
...
```

> 从matplotlib.backends.backend_qtagg模块中导入了NavigationToolbar2QT类，并重命名为NavigationToolbar。

NavigationToolbar也是一个QWidget控件，可以直接嵌入GUI窗体上。

```
import sys
import numpy as np
from qtpy.QtWidgets import QApplication, QWidget, QVBoxLayout
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt

class MainWin(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("matplotlib embeded in Python Qt with figure toolbar")
        self.initUI()
        self.plotfig()
        
    def initUI(self): 
        self.fig = plt.figure()                                                   #创建figure对象
        self.canvas=FigureCanvas(self.fig)                              #创建figure画布
        self.figtoolbar=NavigationToolbar(self.canvas, self)     #创建figure工具栏
       
        vlayout=QVBoxLayout()
        vlayout.addWidget(self.canvas)                                 #画布添加到窗口布局中
        vlayout.addWidget(self.figtoolbar)                             #工具栏添加到窗口布局中
        self.setLayout(vlayout)
        
    def plotfig(self):                                                            #绘制matplot图形
        ax = self.fig.subplots()
        t = np.linspace(0,2*np.pi,50)
        ax.plot(t,np.sin(t))
        ax.autoscale_view()


app=QApplication(sys.argv)
win = MainWin()
win.show()
sys.exit(app.exec())
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/f7459698b26f42dd9ed34fcb93206830.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



## 4 Matplotlib与GUI互动操作

应用程序与嵌入GUI的matplotlib图形交互实际上就是对Figure、FigureCanvas和NavigationToolbar进行操作。除了绘图函数外，比较有用的函数汇总如下：


+ FigureCanvas
    + canvas.draw()：重绘
    + canvas.resize(w,h)：设置画布尺寸(像素)
    + canvas.set_cursor(cusor)：设置鼠标指针形状
    + canvas.mpl_connect("key_press_event",func)：绑定事件，返回cid。
    + canvas.mpl_disconnect(cid)：解除事件绑定
+ NavigationToolbar
    + toolbar.home()：恢复home视图    
    + toolbar.back()：退回上一个历史视图
    + toolbar.forward()：向前到下一个视图
    + toolbar.set_history_buttons()：启用、禁用back/forward按钮
    + toolbar.pan()：开关移动视图模式
    + toolbar.zoom()：开关缩放视图模式
    + toolbar.configure_subplots()：打开设置对话框
    + toolbar.save_figure()：打开保存对话框
    + toolbar.set_message("text")：在工具栏的状态显示区显示文本。
+ plt
    + plt.gcf()：获取当前figure
    + plt.gca()：获取当前axes
+ Figure
    + fig.clf()：清空当前figure
    + fig.gca()：获取当前axes
    + fig.sca(ax)：把ax设置为当前axes
    + fig.delaxes(ax)：删除某个axes
    + fig.set_canvas(figurecanvase)：为figure设置FigureCanvas
    + fig.show()：使用GUI作为backend，显示figure



## 5 Matplotlib嵌入Qt实例



```python
import sys
import numpy as np
from qtpy.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,QPushButton
from matplotlib.backends.backend_qtagg import FigureCanvas, NavigationToolbar2QT as NavigationToolbar
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("matplotlib embeded in Python Qt")
        self.initUI()
        self.plotfig()
        
    def initUI(self):
        self.fig = plt.figure()
        self.canvas=FigureCanvas(self.fig)
        self.figtoolbar=NavigationToolbar(self.canvas, self)
        
        self.btn_start=QPushButton("start")
        self.btn_pause=QPushButton("pause")
        hlayout=QHBoxLayout()
        hlayout.addStretch(1)
        hlayout.addWidget(self.btn_start)
        hlayout.addWidget(self.btn_pause)
        hlayout.addStretch(1)
        
        vlayout=QVBoxLayout()
        vlayout.addWidget(self.figtoolbar)
        vlayout.addWidget(self.canvas)
        vlayout.addLayout(hlayout)
        widget=QWidget()
        widget.setLayout(vlayout)
        self.setCentralWidget(widget)
        
        
            
    def plotfig(self):
        ax = self.fig.subplots()
        self.t = np.linspace(0,2*np.pi,50)
        self.lines=ax.plot(np.sin(self.t))
        ax.autoscale_view()

        def aniupdate(i):
            t=self.t+2*np.pi*i/50
            self.lines[0].set_ydata(np.sin(t))
            return self.lines
        self.ani=FuncAnimation(self.fig, aniupdate, interval=100)
        self.btn_start.clicked.connect(self.ani.resume)
        self.btn_pause.clicked.connect(self.ani.pause)
        
app=QApplication(sys.argv)
win = MainWin()
win.show()
sys.exit(app.exec())
```

![在这里插入图片描述](https://img-blog.csdnimg.cn/da0199f293724913a7f174b4306f9fd0.gif#pic_center)


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
