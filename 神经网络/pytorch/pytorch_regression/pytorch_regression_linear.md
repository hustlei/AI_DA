pytorch人工神经网络基础：单层神经网络(线性回归)
=====

线性回归是人工神经网络的基础，感知机都就是单层或少数层的线性回归。线性回归属于有监督的学习，即根据有标签（已知结果的数据）拟合线性方程权重，然后根据拟合的方程预测未知数据。

通常步骤为：

1. 准备数据：获取有标签的数据（有结果的数据）。
2. 建立模型：根据线性方程设计模型。
3. 配置模型：确定损失函数、优化方法、初始化参数。
4. 训练模型：根据有标签的数据进行回归学习。
5. 测试：根据训练好的（回归结果）线性方程模型计算，评估模型是否准确。

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
y_1=w_11x_1+w_12x_2+b_1 \\
y_2=w_21x_1+w_22x_2+b_2
$$


本文取`w=[[1,1],[0.5,0.8]], b=[[0.5],[1]]`。即

$$
y_1=x_1+x_2+0.5 \\
y_2=0.5x_1+0.8x_2+1
$$

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




    <mpl_toolkits.mplot3d.art3d.Poly3DCollection at 0x1bd29f1c280>




    
![png](output_6_1.png)
    


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

    tensor([-1.3209], grad_fn=<AddBackward0>)
    tensor([[-1.3209],
            [-0.6011],
            [-2.0407]])
    


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

    tensor([[0.5917],
            [2.0620],
            [3.5323],
            [5.0026]], grad_fn=<MmBackward0>)
    tensor([[[0.4901],
             [0.4901],
             [0.4901]],
    
            [[0.4901],
             [0.4901],
             [0.4901]],
    
            [[0.4901],
             [0.4901],
             [0.4901]]], grad_fn=<UnsafeViewBackward0>)
    


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

    tensor([[ 1.0342, -0.9466],
            [ 1.0342, -0.9466]], grad_fn=<AddmmBackward0>)
    tensor([[ 1.0342, -0.9466],
            [ 1.0342, -0.9466]], grad_fn=<AddBackward0>)
    tensor([[ 1.0342, -0.9466],
            [ 1.0342, -0.9466]], grad_fn=<AddBackward0>)
    

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

    tensor([[0.1891, 0.7207]], grad_fn=<AddmmBackward0>)
    

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


```python
### 3.3.3 用nn.Module还是nn.Sequential

nn.Sequential继承自Module，像pytorch预定义的层一样，nn.Sequential可以直接作为模型使用，也可以作为层嵌套在Module中（通常对于复杂的模型非常有用）。

```
class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Sequentail(Linear(1,1))
        
    def forward(self, x):
        return self.layer1(x)
```
```

### 3.3.3 创建线性模型

本文模型是一个非常简单的单层线性模型，用`net=nn.Linear(2,2)`创建模型对象也是ok的。但是通常模型都不是这样的，所以我们一般也不这么做。在这里我们还是使用Module类定义：


```python
class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1=nn.Linear(in_features, out_features)
        
    def forward(self,x):
        return self.layer1(x)
    
net = Net(in_features, out_features)  #创建神经网络对象
```

# 4 配置模型

训练模型前还需要:

1. 初始化模型参数
2. 定义损失函数(即优化目标)
3. 定义优化方法(更新可学习参数的方法)

+ nn.init模块中包含了常数、随机数等初始化参数的方法。
+ nn模块中包含了很多常用的损失函数，比如均方误差、交叉熵等。
+ optim模块中包含了很多常用的优化方法，比如随机梯度下降方法。


```python
nn.init.normal_(net.layer1.weight, mean=0, std=0.01)  #正态分布随机数初始化线性层权重
nn.init.constant_(net.layer1.bias, 0)  #常数初始化偏置项。net.layer1.bias.data.fill_(0)也ok
mseloss = nn.MSELoss()  #均方误差对象作为损失函数
sgd = optim.SGD(net.parameters(), lr=0.03)  #用随机梯度下降对象作为优化函数。
```

# 5 训练模型

训练模型通常采用循环迭代如下步骤完成：

1. 前向传播，根据输入数据预测输出数据
2. 反向传播，计算梯度
3. 优化，更新参数
4. 清零梯度数据


```python
for epoch in range(1001):
    y_predict = net(x_train) #前向传播，自动调用Module的forward方法
    loss = mseloss(y_predict, y_train)  #计算损失函数
    loss.backward() #反向传播
    sgd.step() #更新参数(weight和bias)
    sgd.zero_grad() #清零梯度数据
    
    np.set_printoptions(precision=2)
    if(epoch%100==0):
        print(f"epoch:{epoch}, loss:{loss:.2f}, weight:{net.layer1.weight.data.numpy()}, bias:{net.layer1.bias.data.numpy()}")
```

    epoch:0, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:100, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:200, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:300, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:400, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:500, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:600, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:700, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:800, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:900, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    epoch:1000, loss:1.03, weight:[[0.97 1.03]
     [0.49 0.84]], bias:[0.51 0.99]
    

训练好的模型可以用于预测，即根据未知结果的x计算预测y。由于我们已知线性回归的系数。这里就不在预测了。

# 6 完整代码


```python
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt

# 生成训练数据

N = 1000  #训练数据样本个数。
in_features = 2  #单个输入样本的变量个数。
out_features = 2  #单个输出的变量个数。

x_train = torch.randn(N, 2)  #形状为（N,in_features）
w = torch.tensor([[1, 1], [0.5, 0.8]])  #注意形状和方程顺序相同
b = torch.tensor([0.5, 1])  #注意形状为(out_features)，这样y可以用x@w.T+b计算
y_train = x_train @ w.T + b  #形状为（N,out_features）
y_train += torch.randn(N, 2)  #增加数据噪声

#创建模型


class Net(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer1(x)


net = Net(in_features, out_features)  #创建神经网络对象

#配置模型

nn.init.normal_(net.layer1.weight, mean=0, std=0.01)  #正态分布随机数初始化线性层权重
nn.init.constant_(net.layer1.bias,
                  0)  #常数初始化偏置项。net.layer1.bias.data.fill_(0)也ok
mseloss = nn.MSELoss()  #均方误差对象作为损失函数
sgd = optim.SGD(net.parameters(), lr=0.03)  #用随机梯度下降对象作为优化函数。

#训练模型

for epoch in range(1001):
    y_predict = net(x_train)  #前向传播，自动调用Module的forward方法
    loss = mseloss(y_predict, y_train)  #计算损失函数
    loss.backward()  #反向传播
    sgd.step()  #更新参数(weight和bias)
    sgd.zero_grad()  #清零梯度数据

    np.set_printoptions(precision=2)
    if (epoch % 100 == 0):
        print(
            f"epoch:{epoch}, loss:{loss:.2f}, weight:{net.layer1.weight.data.numpy()}, bias:{net.layer1.bias.data.numpy()}"
        )
```

    epoch:0, loss:2.92, weight:[[0.04 0.03]
     [0.02 0.02]], bias:[0.01 0.03]
    epoch:100, loss:1.04, weight:[[0.96 0.91]
     [0.48 0.74]], bias:[0.48 0.97]
    epoch:200, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.79]], bias:[0.51 1.03]
    epoch:300, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.8 ]], bias:[0.51 1.03]
    epoch:400, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.8 ]], bias:[0.51 1.03]
    epoch:500, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.8 ]], bias:[0.51 1.03]
    epoch:600, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.8 ]], bias:[0.51 1.03]
    epoch:700, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.8 ]], bias:[0.51 1.03]
    epoch:800, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.8 ]], bias:[0.51 1.03]
    epoch:900, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.8 ]], bias:[0.51 1.03]
    epoch:1000, loss:1.03, weight:[[1.01 0.98]
     [0.52 0.8 ]], bias:[0.51 1.03]
    
