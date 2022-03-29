# Python科学计算和数据分析库简介



<font color=#888>数据表示、数据清理、数据统计、数据可视化等算法库是科学计算、数据分析以及人工智能的基础。

<font color=#888>python拥有非常丰富数据科学相关的库，被广泛用于工程领域、数据分析领域以及人工智能领域。主要库有：

# **Python科学计算和数据分析库**

<hr>

+ [**Numpy**：科学计算和数据分析基础库](https://blog.csdn.net/hustlei/article/details/122005976)
+ **Scipy**：科学计算工具库
+ [**Pandas**：数据分析工具库](https://blog.csdn.net/hustlei/article/details/122178919)
+ **Matplotlib**：2D可视化绘图库
+ **Seaborn**：更加易用的2D可视化绘图库
+ **Mayavi2**：可交互3D可视化绘图库

<font color=#AAA>Matplotlib和Seaborn都是静态绘图库，不支持交互式控件。交互式绘图库推荐PyEcharts和PyQtGraph。
+ <font color=#AAA>PyEcharts是一个功能强大的基于web的可嵌入网页的交互式绘图库。plotly也不错，但是部分功能收费。
+  <font color=#AAA>PyQtGraph是一个基于qt的高性能交互式绘图库，主要用于数学、科学、工程领域。

> + Numpy，Pandas，Matplotlib是Python数据分析使用频率最高的库。
> + Numpy，Scipy可以代替Matlab的常规矩阵运算、科学计算功能（不含Simulink和专用行业部分）
> + Tensorflow、Pytorch等深度学习库和Numpy、Pandas、Matplotlib、Seaborn配合，也能够更加高效的完成工作

<br>

# **Numpy，Scipy，Pandas，Matplotlib，Seaborn，Mayavi2简介**
<hr>

## 1. **Numpy**
<font color=#888>Numpy（Numeric Python）是**高性能科学计算和数据分析**的基础库。

<font color=#888>Numpy的核心由**多维数组对象**和用于**处理数组的函数**组成。Numpy提供了N维数组基础操作，数组的算术和逻辑运算，随机数和随机分布，线性代数，统计，傅里叶变换等内置函数。代码简洁且速度快。

> Numpy是几乎所有数据分析高级库（比如scipy，pandas）的构建基础。

<br>

## 2. **Scipy**

<font color=#888>Scipy是基于Numpy的**科学计算**工具库，方便、易于使用、专为科学和工程设计。

<font color=#888>Scipy提供了许多用户友好和高效的高阶方法，如插值，积分，统计，优化，图像处理等等。

<font color=#888>Scipy包含Matlab的大多计算功能，和数据处理的关系不大，数值计算或者工程研究应用更多一些。

> StatsModels是一个统计库，着重于统计模型。包含了许多的统计模型，线性模型、广义线性模型、
> 方差分析、和线性混合效用模型等，在统计方面有其独特的优势。可以作为Scipy.stats的补充。

<br>

## 3. **Pandas**

<font color=#888>Pandas（Panel Data）是基于NumPy的**数据分析**库。 包含许多数据模型。Pandas纳入了大量库和一些标准的数据模型，提供了高效地操作大型数据集
所需的方法，并且提供了大量能使我们快速便捷地处理数据的内置函数。

<font color=#888>Pandas的核心为三种数据结构和操作：

+ <font color=#888>Series：带索引的一维数组，与Numpy中的一维array类似。
+ <font color=#888>DataFrame：带索引的二维的表格型数据结构。

> <font color=#888>老版本有Panel(三维的数组，DataFrame的容器)数据结构，新版本已经废除。

<br>

## 4. **Matplotlib**

<font color=#888>Matplotlib 是一个**2D可视化绘图**库。可以使用这个库轻松地完成线形图、直方图、条形图、误差图或散点图等操作，设置标签、图例、调整绘图大小等。

> Numpy，Pandas，Matplotlib被称为Python数据分析三大支柱。

<br>

## 5. **Seaborn**

<font color=#888>Seaborn是基于Matplotlib的**高级2D图形可视化**工具包。
<font color=#888>Seaborn是在matplotlib的基础上进行了更高级的API封装，从而使得作图更加容易，在大多数情况下使用seaborn能做出很具有吸引力的图。

<br>

## 6. **Mayavi2**

<font color=#888>Mayavi2是基于VTK开发的可视化python库（更加高效）。Mayavi2旨在提供方便和互动的三维数据可视化。

<font color=#888>Mayavi2无缝集成numpy和3D绘图，可以嵌入到用户编写的Python程序中，并且提供了而向脚本的mlab模块，以方便用户快速绘制三维图，和matplotlib的pylab—样。


<br>
<hr>

> 持续更新及修订升级中。