Matplotlib系列（二）：设置、样式及颜色风格
==============================


@[TOC]


# 简介

<font color=#888>Matplotlib提供了非常强大的自定义配置接口和文本标注功能。

<font color=#888>Matplotlib提供了三种方式自定义配置。

+ <font color=#888>统一的设置样式和属性的接口rcParams。可以在运行时修改设置，改变显示效果。
+ <font color=#888>预定义好的rcParams可以保存为style样式，可以随时切换不同的样式。
+ <font color=#888>保存rcParams的matplotlibrc配置文件，可以设置项目或全局配置。

<font color=#888>同时Matplotlib提供了很多预定义的不同风格的颜色ColorMap。可以方便的切换不同风格的预定义颜色。


# 思维导图

![在这里插入图片描述](https://img-blog.csdnimg.cn/47dc9571203940d38c3a29e469227d31.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


<br>

# Matplotlib样式设置及注释文本

## 1. rcParams配置

### 1.1 配置字典rcParams和配置文件

matplotlib有一个专门的配置图形样式的数据字典rcParams(rcParams是一个可以验证数据类型的字典)。
rcParams中配置了几乎所有可视化图形属性。（图形属性包括包括窗体大小、每英寸的点数、线条宽度、颜色、样式、坐标轴、坐标和网络属性、文本、字体等），称为rc（runtime configuration）参数或者rc配置。

### 1.2 获取配置

**获取当前配置**

+ `mpl.rcParams`： #获取当前配置字典。注意P是大写
+ `plt.rcParams`： #获取当前配置字典。注意P是大写

> mpl指matplotlib模块。plt指matplotlib.pyplot模块。

**获取默认配置

+ `mpl.rcParamsDefault`： #获取默认配置字典
+ `plt.rcParamsDefault`： #获取默认配置字典
+ `mpl.rc_params()`：     #获取默认配置字典

> 注意修改当前配置不影响默认配置。修改默认配置也不影响当前配置。
> 但是可以用默认配置恢复（覆盖）当前配置。

### 1.3 修改配置

+ 直接修改当前配置字典
    + `plt.rcParams['figure.figsize']=(6,4)`
    + `plt.rcParams['lines.linewidth']=2`
+ rc函数修改
    + `plt.rc('lines',linewidth=4, color='g')`
+ 恢复默认参数
    + `plt.rcdefaults()`：把rcParams恢复到Matplotlib内部默认值
    + `mpl.rc_file_defaults()`：把rcParams恢复到Matplotlib最初从配置文件加载的值
+ 从rc文件加载更新
    + `mpl.rc_file('rcfile')`

> 上述方法修改配置，只影响当前程序。

使用with关键字，可以仅对部分语句修改配置：

```
with plt.rc_context({'lines.linewidth': 2, 'lines.linestyle': ':'}):
    plt.plot(x,y)
```

> + mpl.rcParams与plt.rcParams相同
> + mpl.rc()与plt.rc()相同
> + mpl.rcdefaults()与plt.rcdefaults()相同
> + mpl.rc_context()与plt.rc_context()相同
> + plt无rc_file_defaults()和rc_file()函数

### 1.4 配置文件

matplotlib配置信息是从配置文件中读取的。如果不希望在每次代码开始时进行参数配置，可以设置全局或者项目中的配置文件。配置文件有三个位置：

+ 系统级配置文件
    + 通常在python的`Lib\site-packages\matplotlib\mpl-data`文件夹下，文件名称是`matplotlibrc`。
    + 每次重装matplotlib之后该配置文件就会被覆盖。
+ 用户级配置文件
    + 通常在$HOME\.matplotlib目录下。可以用`matplotlib.get_configdir()`函数来查找当前用户的配置文件目录。
    + 可以通过MATPLOTLIBRC变量修改它的位置。
+ 项目级配置文件
    + 当前工作目录，即项目的目录下，可以为目录所包含的当前项目给出配置文件，文件名为`matplotlibrc`。

> 每次启动时会自动加载配置文件，优先级顺序是： 当前工作目录 > 用户级配置文件 > 系统级配置文件。

查看当前加载使用的配置文件的路径的方式：用`mpl.matplotlib_fname()`函数，通常在大家未设置项目级和用户级配置文件时，显示系统及配置文件位置。

配置包含关键词值对，文件格式如下：

```ini
lines.color       : C0         ## has no affect on plot(); see axes.prop_cycle
lines.marker    : None    ## the default marker
```

### 1.5 常用配置选项

+ font：字体设置
    + `plt.rcParams['font.sans-serif']='SimHei'`：设置字体，可以用列表设置多个字体
    + `plt.rcParams['font.size']=12`
+ figure：外框画布
    + `plt.rcParams['figure.figsize']`：画布大小
    + `plt.rcParams['figure.dpi']`：画布分辨率
    + `plt.rcParams['figure.h_pad']`:设置子图高度方向边距
    + `plt.rcParams['figure.w_pad']`:设置子图宽度方向边距
    + `plt.rcParams['figure.facecolor']`:设置颜色
+ axes：内框画布
    + `plt.rcParams['axes.unicode_minus']=False`：设置字符
    + `plt.rcParams['axes.titlesize']`：图形的标题大小
    + `plt.rcParams['axes.labelsize']`	：图形的x,y轴标签大小
    + `plt.rcParams['axes.prop_cycle']`：可以通过C0,C1,C2,...CN方式循环访问的颜色列表。
+ lines：线条设置
    + `plt.rcParams['lines.color'] = 'blue'`：设置线条颜色
    + `plt.rcParams['lines.linestyle'] = '-'`：设置线条样式
    + `plt.rcParams['lines.linewidth'] = 3`：设置线条宽度
    + `plt.rcParams['lines.marker'] = None`：设置点样式
    + `plt.rcParams['lines.markersize'] = 6`：设置点大小
    + `plt.rcParams['lines.markerfacewidth'] = '#aaa'`：设置点填充颜色
    + `plt.rcParams['lines.markeredgewidth'] = 0.5`：设置点边缘线宽
+ grid：网格设置
    + `plt.rcParams['grid.color'] = 'gray'`：设置网格颜色
    + `plt.rcParams['grid.alpha'] = 0.8`：设置网格颜色
    + `plt.rcParams['grid.linestyle'] = ':'`：设置网格线型
    + `plt.rcParams['grid.linewidth'] = 0.5`：设置网格线宽
+ xtick, ytick：x,y轴刻度及文本设置
    + `plt.rcParams['xtick.labelsize']`：x轴刻度字体大小
    + `plt.rcParams['ytick.labelsize']`：y轴刻度字体大小
    + `plt.rcParams['xtick.major.size']`：x轴最大刻度
    + `plt.rcParams['ytick.major.size']`：y轴最大刻度
+ legend：图例设置
    + `plt.rcParams['legend.fontsize']='large'`：设置图例字体大小
+ text
+ patch：填充2D空间的图形图像，如多边形和圆。控制线宽、颜色和抗锯齿设置等
+ image：图片设置
+ animation：动画设置

> 实际上和各个对象的数据及绘图函数的参数是对应的。


#### 中文字体配置

```
plt.rcParams['font.sans-serif'] = ['SimHei']  #指定默认字体
plt.rcParams['axes.unicode_minus']=False      #解决保存图像是负号'-'显示为方块的问题
```

windows下默认支持字体：

+ 黑体："SimHei"
+ 宋体："SimSun"
+ 仿宋："FangSong"
+ 楷体："KaiTi"
+ 微软雅黑："Microsoft YaHei"



## 2. 样式表StyleSheet

样式表StyleSheet类似与rcParams，但是只包含与具体绘图外观相关的参数。

> 也就是说相比rcParams，有少量参数样式表不支持。
> 不支持的参数有backend,backend_fallback,datapath,date.epoch,docstring.hardcopy,figure.max_open_warning,figure.raise_window,interactive,savefig.directory,timezone,tk.window_focus,toolbar,webagg.address,webagg.open_in_browser,webagg.port,webagg.port_retries。所以，日常我们常用的rcParams参数，style中都支持。

### 2.1 使用样式的方法

Matplotlib提供了很多预定义的样式表。可以非常方便的切换应用。切换方法如下：

```
plt.style.use('ggplot')
```

在某个语句块内使用样式方法如下：

```
with plt.style.context('ggplot'):
    plt.plot(x,y)
```

**组合使用多个样式**

我们可以用一个样式表定义颜色，另一个样式表定义元素的尺寸。通过传入样式表列表，将表组合在一起。

```
plt.style.use(['ggplot','grayscale'])
```

需要注意的是，多个表中有相同的参数时右边的会覆盖左边的。

### 2.2 预定义样式

实际上，每个样式表都保存在一个样式表文件中，每一个预设的style都是一个style文件，它是以.mplstyle   为后缀的文件。预设的样式表文件都在`Lib\site-packages\matplotlib\mpl-data\stylelib`文件夹下。

`plt.style.available`可以列出所有可用的样式。

常用的预设样式有：

+ classic：默认风格
+ grayscale：灰度
+ ggplot
+ seaborn
+ fivethirtyeight
+ Solarize_Light2


### 2.3 自定义样式

样式表文件和rcParams配置文件格式相同。
我在可以在样式表文件夹下自定义一个myownstyle.mplstyle文件，里面的内容如下所示：

```
lines.color: green
lines.linewidth:8
text.color: white
axes.facecolor: yellow
axes.edgecolor:black
```

然后用`plt.style.use('myownstyle')`使用样式。


## 3. ColorMap颜色风格

Matplotlib预定义了很多不同风格的ColorMap，可以为不同风格的图形提供一系列颜色，类似调色板。
ColorMap实际上就是一系列渐变的颜色。可以用渐变的颜色显示数据规律，比如密度大小，不同高度等；也可以用渐变的颜色显示一系列图形，用于区分。


> Matplotlib支持多种颜色表达方式，如：
> + 基本颜色简写：'r','g','c','k'等等
> + 颜色名称：'gray','pink','lightblue'等等
> + 灰度数字：'0.5','0.3'等等
> + 循环引用颜色：'C0', 'C1', 'C100'等
> + 十六进制颜色字符串:'#RGB', '#RRGGBB', '#RRGGBBAA'三种形式
> + RGB元组：(0.5,0.3,0.2), (0.2,0.3,0.5,0.5)等
> + ColorMap
>> 颜色表达方法详见Matplotlib系列(一)

### 3.1 ColorMap用法

通常我们不直接从colormap中提取某个颜色。而是用较小的值表示colormap开始的颜色，用较大的值表示colormap最后的颜色，用数据插值表示不同的颜色。

colormap的典型用法：

```
x=np.arange(5)
y=x**2
plt.scatter(x,y,c=[0.7,0.3,0.1,0.5,0.8],cmap='Blues')
#参数c即color，表示每个点的颜色映射值。注意c的长度要和点数相同
#参数cmap设置应用的colormap。用cmap=plt.cm.Blues也是ok的。
```

**数据映射ColorMap颜色详解**

‎默认情况下使用浮点数线性映射色彩映射表中的颜色。例如：

```
plt.scatter(x,y,c=color,cmap='Blues',vmin=1,vmax=10)
#用1到10的数据映射ColorMap。
```

Matplotlib根据数据映射颜色，需要两个步骤：

+ 把color浮点数列表，从`[1,10]`转换（规范化）到`[0,1]`
+ 然后根据数值插值获取颜色。

这个过程称为规范化（Normalization)。在未指定vmin和vmax的时候，取最小值作为vmin，最大值作为vmax。


> 通常采用线性插值方法获取颜色，但是matplotlib还支持对数等插值方法。详见[文档](https://matplotlib.org/stable/tutorials/colors/colormapnorms.html)。

### 3.2 从ColorMap获取颜色列表

有时候我们需要颜色列表赋值给函数参数，用get_cmap根据名称获取colormap实例，可以很方法的获取颜色列表。

> 并不是所有函数的颜色参数都能支持赋值为浮点数映射colormap的。
> + 很多函数不支持cmap参数，颜色参数只能用颜色值，不能用浮点数或浮点数列表作为参数值
> + 部分绘图(主要是生成的图形为collection的函数，比如voxels函数)虽然支持cmap参数，但是颜色一样不能用浮点数映射。

+ c=plt.get_cmap('RdBu')：获取colormap实例
+ c(0.5)：获取对应的颜色，得到的是一个(r,g,b,a)元祖
+ `c([0.1,0.2,0.3])`：获取一组颜色
    + 参数可以是整数、可以是浮点数。并且都可以是列表
    + 对于浮点数，数值应为`[0,1]`。大于1取1，小于0取0
    + 对于整数，数值应为`[0, Colormap.N)`。通常N为256。
+ c=plt.get_cmap('OrRd‘,300)：获取colormap实例，并且通过重新采样设置Colormap.N=300。

> ColorMap类实现了__call__函数，colormap的实例可以当做函数调用。

colormap还可以用名称来调用。比如`plt.cm.RdBu`

```python
> colors=plt.get_cmap('OrRd')([0.1,0.5,0.8])  #从名为'OrRd'的ColorMap获取三个颜色
>>> colors
array([[0.99692426, 0.92249135, 0.81476355, 1.        ],
       [0.9874356 , 0.55048058, 0.34797386, 1.        ],
       [0.78666667, 0.11294118, 0.07294118, 1.        ]])
```


### 3.3 内置ColorMap简介


matplotlib内置了很多colormap。按照类别列举如下：

```yaml
+ Perceptually Uniform Sequential系列（亮度变化，饱和度增量）
    - 'viridis'
    - 'plasma', 'inferno', 'magma'
    - 'cividis'
+ Sequential系列（亮度变化，单一颜色）
    - 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds'
    - 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd'
    - 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn'
+ Sequential (2)系列（亮度变化，单一颜色）
    - 'binary', 'gist_yarg', 'gist_gray', 'gray'
    - 'bone'
    - 'pink',
    - 'spring', 'summer', 'autumn', 'winter'
    - 'cool', 'Wistia', 'hot', 'afmhot', 'gist_heat', 'copper'
+ Diverging系列（亮度变化，两种颜色饱和度变化）
    - 'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu', 'RdYlBu', 'RdYlGn'
    - 'Spectral', 'coolwarm', 'bwr', 'seismic'
+ Cyclic系列（两种不同颜色亮度变化，饱和度循环，首尾颜色相同）
    - 'twilight', 'twilight_shifted'
    - 'hsv'
+ Qualitative系列（离散颜色，没有顺序关系）
    - 'Pastel1', 'Pastel2'
    - 'Paired'
    - 'Accent', 'Dark2'
    - 'Set1', 'Set2', 'Set3'
    - 'tab10', 'tab20', 'tab20b', 'tab20c'
+ Miscellaneous系列（特定用途，如海洋地形等）
    - 'flag', 'prism'
    - 'ocean'
    - 'gist_earth'
    - 'terrain'
    - 'gist_stern',
    - 'gnuplot', 'gnuplot2'
    - 'CMRmap'
    - 'cubehelix'
    - 'brg',
    - 'gist_rainbow'
    - 'rainbow'
    - 'jet'
    - 'turbo'
    - 'nipy_spectral',
    - 'gist_ncar'
```

在函数中使用Colomap的方式为`cmap='Summer'`。在Colormap后边加上_r，得到一个颜色顺序逆向排列的Colormap。比如`cmap='Summer_r'`

预设Colormap的颜色效果见：

+ <https://matplotlib.org/stable/gallery/color/colormap_reference.html>
+ <https://matplotlib.org/stable/tutorials/colors/colormaps.html>



## 参考资料

+ 官方样式和配置帮助文档：<https://matplotlib.org/stable/tutorials/introductory/customizing.html>
+ 官方颜色文档：<https://matplotlib.org/stable/tutorials/colors/colors.html>
+ 官方ColorMap文档：<https://matplotlib.org/stable/gallery/color/colormap_reference.html>
+ 官方ColorMap用法详细文档：<https://matplotlib.org/stable/tutorials/colors/colormapnorms.html>

<br>
<hr>

> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
>  <font color=#888>**未经允许请勿转载。**