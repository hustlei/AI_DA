

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
pytorch内置的线性层nn.Linear

@author: lei<hustlei@sina.cn>
"""


'''
# pytorch nn.Linear简介

nn.Linear类可以定义一个线性函数对象。它把通过$y=xw^T+b$，把输入x转换为输出y。构造函数：

+ 构造函数：`nn.Linear(in_features, out_features, bias=True, device=None, dtype=None)`
    + in_features：输入x包含的变量个数。如果x有N个样本，则输入形状为(N,in_features)
    + out_features：每个样本计算后输出的y长度。如果x有N个样本，则输出形状为(N,out_features)
    + bias：是否设置偏置b。
+ 对象的参数
    + weight：可学习的参数。形状为(out_features,in_features)。创建对象的时候会自动初始化一个随机值。
    + bias：可学习的参数。形状为(out_features）。创建对象的时候会自动初始化一个随机值。
                      
> + Linear对象可以当作函数，把x作为参数计算y。即y=nn.Linear(...)(x)
> + 注意：x必须是至少一维的浮点数张量。不能是标量，不能是numpy数组，不能是整型张量。
'''


import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt


# nn.Linear用法
## 一元一次线性方程(in_features=1,out_features=1)

linear = nn.Linear(1,1) #创建输入为1元变量，输出也为1元变量的线性对象。

x = torch.tensor([2.])  #注意x必须是≥一维的浮点数张量。
y = linear(x)
print(y)

x = torch.tensor([[2],[1],[3]],dtype=torch.float)  #注意x的最后一维必须是1。
y = linear(x)  #注意y的最后一维同样是1。
print(y.data)

## 多元一次线性方程(in_features>1,out_features=1)

linear = nn.Linear(3,1,bias=False) #创建输入为3元变量，输出为1元变量的线性对象。
x=torch.arange(12,dtype=torch.float).reshape(4,3)  #输入样本包含4个元素，每个元素长度为3
y=linear(x)  #输出张量形状为(4,1),即4个元素，每个元素是一个值
print(y)

x=torch.ones(3,3,3) #输入样本形状为(3,3,3)
y=linear(x) #输出张量形状为(3,3,1)
print(y)

## 多元一次方程组(in_features>1,out_features>1)

linear = nn.Linear(3,2)  #创建输入为3元变量，输出为2元变量的线性对象。即两个三元一次方程组成的方程组。
x = torch.ones(2,3)
y = linear(x)  #输出y的形状为(2,2)
print(y)





# 应用pytorch nn.Linear进行线性回归

## 数据准备

'''用3元线性方程组(2个方程)
$$
y_1=x_1+2x_2+3x_3+4
y_2=2x_1+3x_2+4x_3+5
$$
生成数据'''

N = 100 #数据样本数，即生成N个数据点
x = torch.rand(N,3)
w_true = torch.tensor([[1,2,3],[2,3,4]],dtype=torch.float)
b_true = torch.tensor([4,5])
y = x@w_true.T+b_true  #y形状为(N,2)
y += torch.randn(N,2)*0.0001  #加上随机噪声。

'''查看数据'''
fig = plt.figure()
ax = fig.add_subplot(projection="3d") #2行3列子图

ax.scatter(x[:,0],x[:,1],x[:,2],c=y[:,0],s=y[:,1]) #x与y的关系。颜色和点大小表示y。
plt.show()

## 建立模型

linear1=nn.Linear(3,2)  #输入为3元变量，输出为2元变量的线性对象。

    
## 训练




