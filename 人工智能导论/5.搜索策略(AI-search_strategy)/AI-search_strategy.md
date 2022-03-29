人工智能导论(5)——搜索策略(Search Strategy)
==============================

@[TOC]

<hr>

# 一、 概述

<font color=#999>人工智能经典三大基本技术为：知识表示、推理、搜索策略。
其中搜索直接关系到智能系统的性能与运行效率，搜索技术渗透在各种人工智能系统中。专家系统、自然语言理解、自动程序设计、模式识别、机器学习、信息检索和博弈等领域都广泛使用搜索技术。

> <font color=#999>为方便记忆和回顾，根据个人学习，总结人工智能基础知识和思维导图形成系列。

# 二、 重点内容

+ <font color=#999>基本概念
+ <font color=#999>状态空间图表示方法
+ <font color=#999>盲目搜索
+ <font color=#999>启发式搜索

# 三、 思维导图


![人工智能基础知识(5)——搜索策略思维导图](https://img-blog.csdnimg.cn/82ebff2e7aa5463ca925d47588920972.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)




# 四、 重点知识笔记

## 1. 概述

### 1.1 基本概念

<font color=#999>求解一个问题时，涉及到两个方面：

+ <font color=#999>问题的表示
+ <font color=#999>选择一个相对合适的求解方法


<font color=#999>问题求解的基本方法：

+ <font color=#999>搜索法
+ <font color=#999>归约法
+ <font color=#999>归结法
+ <font color=#999>推理法
+ <font color=#999>产生式

<font color=#999>搜索就是找到智能系统的操作序列(如下棋走一步棋)的过程，是一种求解问题的一般方法。

    
### 1.2 状态空间图表示

**概念**

<font color=#999>人工智能中把描述问题的有向图称为状态空间图，简称状态图。

+ <font color=#999>状态图中的结点代表问题的一种格局，一般称为问题的一个状态；
+ <font color=#999>边表示两结点之间操作关系

~~~
          状态2
         ↗
      操作1
     ╱
状态1——操作2——>状态3
     ╲
       操作3
         ↘
          状态4
~~~

**状态空间表示法**

<font color=#999>状态空间表示法是指用“状态”和“操作”组成的“状态空间”来表示问题求解的一种方法。

**（1）状态**state

<font color=#999>描述问题求解过程中不同时刻下状况的一组变量或数组。

~~~
S=[s1, s2, ...]
~~~

<font color=#999>例如：三个硬币的正反面状态

+ <font color=#999>状态1：`[正，正，正]`
+ <font color=#999>状态2：`[正，正，反]`
+ <font color=#999>状态3：`[正，反，反]`
+ <font color=#999>等共8种状态

**（2）操作**operator

<font color=#999>操作表示引起状态变化的一组关系或函数。

<font color=#999>例如：上述示例中的，给某个硬币翻面。

**（3）状态空间**state space

<font color=#999>用状态变量和操作符号，表示系统或问题。

<font color=#999>示例：八数空间问题

<font color=#999>初始状态：

$$
\color{#888}
\begin{array}{|c|c|c|}
\hline
3 & 1 & 2 \\ \hline
5 &    & 7 \\ \hline
8 & 4 & 6 \\ \hline
\end{array}
$$

<font color=#999>目标状态：

$$
\color{#888}
\begin{array}{|c|c|c|}
\hline
1 & 2 & 3 \\ \hline
8 &    & 4 \\ \hline
7 & 6 & 5 \\ \hline
\end{array}
$$

<font color=#999>状态集：数字在表格中的所有排法。

<font color=#999>操作算子：空格上移、空格左移、空格下移、空格右移。


## 2. 搜索过程及回溯策略

<font color=#999>搜索过程

1. <font color=#999>从初始状态出发
2. <font color=#999>不断地、试探性地寻找路径
3. <font color=#999>达到目的或者“死胡同”

<font color=#999>回溯策略

+ <font color=#999>遇到“死胡同”，就回溯到路径最近的父节点
+ <font color=#999>查看该节点是否还有其他子节点
    - <font color=#999>若有，沿着子节点继续搜索
    - <font color=#999>若无，继续回溯
+ <font color=#999>找到目标就成功退出搜索


![搜索回溯策略](https://img-blog.csdnimg.cn/ffdfdf57083749acb1113d667a277d0a.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_17,color_FFFFFF,t_70,g_se,x_16#pic_center)



<font color=#999>回溯搜索算法的术语说明

+ <font color=#999>PS(path states)表：当前搜索路径的状态。如果找到了目标，PS就是解的路径。
+ <font color=#999>NPS(new path states)表：新路径状态表。待搜索的状态。
+ <font color=#999>NSS(no solvable states)表：不可解状态集，即“死胡同”状态表。记录无解的路径，遇到路径上的状态就立即排除。

## 3. 盲目搜索

<font color=#999>盲目搜索是指在问题的求解过程中，不运用启发性知识，需要进行全方位的搜索，而没有选择最优的搜索途径。这种搜索具有盲目性，效率较低，容易出现“组合爆炸”问题。

<font color=#999>典型的盲目搜索有深度优先搜索和广度优先搜索。

### 3.1 宽度优先搜索

<font color=#999>宽度优先搜索（Breadth-First Search，BFS）又称为广度优先搜索。

<font color=#999>宽度优先搜索是指:

+ <font color=#999>从初始结点S0开始
+ <font color=#999>向下逐层搜索：在n层结点未搜索完之前，不进入n+1层搜索

<font color=#999>搜索路径示意图如下：

![宽度优先搜索](https://img-blog.csdnimg.cn/47775d86bb01434ab97dfe87a907a680.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)



<font color=#999>宽度优先搜索的复杂度

+ <font color=#999>时间复杂度度：O(b<sup>n</sup>)  指数
+ <font color=#999>空间复杂度度：O(b<sup>n</sup>)  最坏为指数

**宽度优先搜索特点**

+ <font color=#999>时间空间复杂度都比较高
+ <font color=#999>搜索效率低
+ <font color=#999>可总可以找到目标节点，而且是最短路径节点


### 3.1 深度优先搜索

<font color=#999>深度优先搜索（Depth-First Search，DFS）是一种一直向下的搜索策略：

+ <font color=#999>从初始结点S0开始
+ <font color=#999>按生成规则生成下一级各子结点
+ <font color=#999>逐级“纵向”深入搜索，直至达到目标节点

<font color=#999>搜索路径示意图如下：

![深度优先搜索](https://img-blog.csdnimg.cn/be44bd9f33f549ddb08893f09b16bbef.png?x-oss-process=image/watermark,type_d3F5LXplbmhlaQ,shadow_50,text_Q1NETiBAaHVzdGxlaQ==,size_20,color_FFFFFF,t_70,g_se,x_16#pic_center)


<font color=#999>深度优先搜索的复杂度

+ <font color=#999>时间复杂度度：O(b<sup>n</sup>)  指数
+ <font color=#999>空间复杂度度：O(bn)             线性


**深度优先搜索特点**

+ <font color=#999>需要较少的空间（只需要保存搜索树的一部分）
+ <font color=#999>可能搜索到了错误的路径（有些问题具有无限的搜索树，可能无法返回正确的路径）
    - <font color=#999>最终可能会陷入无限循环，不能给出答案
    - <font color=#999>或者找到一个路径很长，且不是最优的答案
    

**有界深度搜索和迭代加深搜索**

<font color=#999>对于深度比较大的情况，深度优先可能搜索需要很长的运行时间，而且可能得不到解答。一种比较好的问题求解方法是对搜索树的深度进行控制，即有界深度优先搜索方法。

<font color=#999>深度优先搜索过程总体上按深度优先算法进行，但对搜索深度需要给出一个深度限制。当深度达到了限制深度时，如果还没有找到解答，就停止对该分支的搜索，换到另一个分支进行搜索。
    
    
## 4. 启发式搜索    
    
<font color=#999>利用与问题有关的启发信息进行搜索。

<font color=#999>在搜索过程中，关键的一步是如何确定下一个要考察的节点，确定方法不同就形成了不同的搜索策略。如果在确定节点时能利用问题的启发信息，估计出节点的重要性，就可以在搜索时选择重要性高的节点。

**估价函数**

<font color=#999>用于估计节点重要性的函数称为估价函数。其一般形式为：

$f(x)=g(x)+h(x)$

+ <font color=#999>$g(x)$表示从初始节点到节点x，已经实际付出的代价
+ <font color=#999>$h(x)$表示从节点x到目标节点的最优路径的估计代价

> <font color=#999>八数码问题的多种估价函数
>
> + <font color=#999>最简单的估价函数：与目标相比，位置不符合的数字数量。
> + <font color=#999>较好的估价函数：各数字移动到目的位置所需移动距离的总和。
> + <font color=#999>第三种估价函数：对每一对逆转数字乘以一个倍数。
> + <font color=#999>第四种估价函数：将位置不符合的数字数目总和与3倍逆转数目相加。


**一般启发式图搜索算法（简记为A算法）**

<font color=#999>待搜索状态表按照f(x)进行排序。


**A\*算法**

+ <font color=#999>最小代价函数：f*(x)=g*(x)+h*(x)
    - <font color=#999>f*(x)——从初始状态到目标状态的最小代价
    - <font color=#999>g*(x)——从初始状态到x的最小代价
    - <font color=#999>h*(x)——从x到目标状态的最小代价
+ <font color=#999>估价函数f(x)=g(x)+h(x)如果满足以下条件，称为f*(x)的估价函数
    - <font color=#999>g(x)是g*(x)的估计，且g(x)>0
    - <font color=#999>h(x)是h*(x)的估计，且有：h(x)≤h*(x)
    
<font color=#999>使用f*(x)的估价函数对待搜索状态表按照f(x)进行排序的算法，称为A*算法。

<font color=#999>在A*算法中：

+ <font color=#999>g(x)笔记容易得到，随着更多搜索信息的获得，g(x)的值呈下降趋势
+ <font color=#999>h(x)的确定依赖于具体问题领域的启发性信息，其中h(x)≤h*(x)的限制十分重要，它可以保证A*算法都能找到最优解。


<br>
<hr>


> 个人总结，部分内容进行了简单的处理和归纳，如有谬误，希望大家指出，持续修订更新中。
>
> 修订历史版本见：<https://github.com/hustlei/AI_Learning_MindMap>
