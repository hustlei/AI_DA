---
jupyter:
  jupytext:
    formats: ipynb,md,Rmd
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

pytorch人工神经网络基础：线性回归（完整实现说明）
=====

线性回归是人工神经网络的基础，早期的感知机都是基于单层或多层线性回归的。

线性回归数据有监督的学习，即根据有标签（已知结果的数据）拟合线性方程权重，然后根据拟合的方程预测未知数据。

通常步骤为：

1） 准备数据：获取有标签的数据（有结果的数据）
2） 建立模型：确定线型方程形式，比如y=wx+b（w为权重，b为偏置项）
3） 训练模型：根据有标签的数据进行回归学习，得到w和b
4） 测试：根据训练好的（回归结果）线性方程模型计算，评估模型是否准确

人工神经网络的训练和测试过程也基本相同。


# 1 准备工作

导入必要的库。

```python
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
```

pytorch的所有操作都是依据张量的，尤其是自动计算梯度的操作不能和numpy数组运算。导入troch，可以使用pytorch的张量操作。

nn模块包含

+ Module(自定义模型的父类)
+ 层:比如Linear类（线性层类）以及卷积层等很多层类。
+ 损失函数：比如MSELoss类（均方误差损失函数）以及交叉熵损失函数等很多损失函数类。
+ init模块：可以用于初始化Module的可学习参数，比如线性模型中的w和b。

optim模块包含用于训练模型的优化函数类，比如SGD(随机梯度下降)。

<!-- #region -->
# 2 准备数据

## 2.1 线性方程说明

线性回归方程的形式为:

$$
y=wx+b
其中：y,w,x,b都可以是向量。
即：y=w_1x_1+w_2x_2+...+w_nx_n+b
$$

当x为二元向量，y为一元向量时：$y=w_1x_1+w_2x_2+b$

当x,y均为二元向量时，相当于基于同一组x向量，回归两个二元方程（当然y可以是多元的，相当于多个二元方程）。即：

$$
y_1=w_11x_1+w_12x_2+b_1
y_2=w_21x_1+w_22x_2+b_2
$$


本文取`w=[[1,1],[0.5,0.8]], b=[[0.5],[1]]`。即

$$
y_1=x_1+x_2+0.5
y_2=0.5x_1+0.8x_2+1
$$
<!-- #endregion -->

## 2.2 生成训练数据(带标签的数据)

在神经网络中通常输入数据为拟研究对象的特征(比如对房价进行回归时，影响房价的因素：价格、位置、大小等等)。因此，训练数据通常叫做feature。结果作为数据标签叫做label，尤其是对分类问题，叫标签更容易理解。

为了方便，本文把feature数据命名为x_train,结果命名为y_train。

在pytorch中，当x为m元向量时，N个训练数据组成的张量x_train的shape通常约定为`[N,m]`。
同样y_train为n元向量时，N个y组成的y_train张量shape通常约定为`[N,n]`。

通过计算方程，加噪声的方法生成数据。

> pytorch都是基于张量数据的，尤其是需要自动求梯度的操作是不能和numpy数组相互运算的，会出错。
> 数据的维度、形状尽可能和模型一致，比如，本文中x,y,w,b都以矩阵形式(虽然用向量有些操作并不会有问题)，避免出错。

```python
N = 1000  #训练数据样本个数。
in_features = 2  #单个输入样本的变量个数。
out_features = 2  #单个输出的变量个数。

x_train = torch.randn(N,2)  #形状为（N,in_features）
w = torch.tensor([[1,1],[0.5,0.8]])  #注意形状和方程顺序相同
b = torch.tensor([0.5,1])  #注意形状为(out_features)，这样y可以用x@w.T+b计算
y_train = x_train@w.T+b  #形状为（N,out_features）
y_train += torch.randn(N,2)  #增加数据噪声


# matplotlib绘图查看生成数据
fig = plt.figure()  #创建matplotlib画布
ax3d = fig.add_subplot(projection="3d")  #创建3d坐标系
ax3d.scatter(x_train[:,0],x_train[:,1],y_train[:,0],c='orange')  #第一个方程生成的数据点，绘制散点图
ax3d.plot_trisurf(x_train[:,0],x_train[:,1],(x_train@w.T+b)[:,0])
```

# 3 建立模型

## 3.1 pytorch建立模型的一般方法

通常直接继承pytorch的Module类，添加神经网络层创建模型。一般形式为：

```
class Net(nn.Module):
    def __init__(self):
        self.layer1 = ...
        
    def forward(self,x):
        return self.layer1(x)
```

根据Net类创建的对象可以直接像函数一样调用。`net=Net(), y=net(x)`,实现forward操作，也就是根据输入计算输出的计算。模型对象的parameters()函数可以获取自身所有对象的可学习参数，用于后续训练作为优化函数的输入。

> Module类支持嵌套，事实上pytorch预定义的层，比如Linear也是Module。


## 3.2 nn.Linear线性层类

pytorch预定义了很多层，可以直接调用，线性层是最常用的一个。nn.Linear类可以定义一个线性函数对象。它把通过$y=xw^T+b$，把输入x转换为输出y。

+ 构造函数：`nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`
    + in_features：整数。输入x包含的变量个数。如果x有N个样本，则输入形状为(N,in_features)
    + out_features：整数。每个样本计算后输出的y长度。如果x有N个样本，则输出形状为(N,out_features)
    + bias：布尔值。是否设置偏置b。
+ Linear对象的参数
    + weight：可学习的参数。形状为(out_features,in_features)。创建对象的时候会自动初始化一个随机值。
    + bias：可学习的参数。形状为(out_features）。创建对象的时候会自动初始化一个随机值。


```python
# nn.Linear用法示例

# 一元线性方程：in_features=1,out_features=1创建一元线性方程对象。

linear = nn.Linear(1,1) #创建输入为1元变量，输出也为1元变量的线性对象。

x = torch.tensor([2.])  #注意x必须是≥1维的浮点数张量。
y = linear(x)
print(y)

#通常数据样本会有多个，以(N,in_features)形式输入数据。注意x必须是浮点数。
x = torch.tensor([[2],[1],[3]],dtype=torch.float)  #（3,1）形状输入数据
y = linear(x)  #注意y的形状为(N,1)。
print(y.data)

#注意：x必须是至少一维的浮点数张量。不能是标量，不能是numpy数组，不能是整型张量。
```

```python
# 多元一次线性方程(in_features>1,out_features=1)

linear = nn.Linear(3,1,bias=False) #创建输入为3元变量，输出为1元变量的线性对象。
x=torch.arange(12,dtype=torch.float).reshape(4,3)  #输入样本包含4个元素，每个元素长度为3
y=linear(x)  #输出张量形状为(4,1),即4个元素，每个元素是一个值
print(y)

x=torch.ones(3,3,3) #输入样本形状为(3,3,3)
y=linear(x) #输出张量形状为(3,3,1)
print(y)
```

```python
## 多元一次方程组(in_features>1,out_features>1)

linear = nn.Linear(3,2)  #创建输入为3元变量，输出为2元变量的线性对象。即两个三元一次方程组成的方程组。
x = torch.ones(2,3)
y = linear(x)  #输出y的形状为(2,2)
print(y)

#根据weight和bias参数计算

print(x@linear.weight.T+linear.bias)
print(torch.mm(x,linear.weight.T)+linear.bias)

```

## 3.3 用nn.Linear创建线性模型

直接继承nn.Module可以创建神经网络模型，还可以用nn.Sequential创建模型，nn.Linear本身也是个模型，单层网络可以直接用nn.Linear。

### 3.3.1 继承nn.Module建立模型

```python
class Net(nn.Module):
    def __init__(self, in_features, out_features): #两个参数是用于线性层创建
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)  #注意必须是模型的成员，才能获取parameter
        
    def forward(self, x):
        return self.layer1(x)
    
# 简单试验模型
x = torch.ones(1,2)  #1个样本
net = Net(2,2)
y = net(x)
print(y)
```

### 3.3.2 应用nn.Sequential建立模型

nn.Sequential是一个把Module包装成类似顺序字典的容器。可以用类似字典或列表的方式创建模型。创建的Sequential对象包含Module的所有功能。

nn.Sequential三种创建方式：

```python
# 方法一
# 层对象按顺序当做Sequential参数
net = nn.Sequential(nn.Linear(2,2))

# 同时传入多个层(线性层、卷积层等都ok)也是可以的
net1 = nn.Sequential(nn.Linear(1,1),
                    nn.Linear(1,2))
```

```python
# 方法二
# 用OrderedDict作为参数创建
from collections import OrderedDict
net = nn.Sequential(OrderedDict([
    ['layer1',nn.Linear(2,2)],
    ['layer2',nn.Linear(1,2)]
]))
```

```python
# 方法三
# 动态添加层
net = nn.Sequential()
net.add_module('layer1',nn.Linear(1,1))
net.add_module('layer2',nn.Linear(2,1))
```

### 3.3.3 创建线性模型

本文模型是一个非常简单的单层线性模型，用`net=nn.Linear(2,2)`创建模型对象也是ok的。但是通常模型都不是这样的，所以我们不这么做。
