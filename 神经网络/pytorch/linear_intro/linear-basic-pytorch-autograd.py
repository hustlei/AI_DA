#!/usr/bin/env python
# -*- coding: utf-8 -*-

#人工神经网络基础：最小二乘法线性回归(应用pytorch自动求梯度功能)
'''最小二乘法回归y=wx+b。应用pytorch自动求梯度功能
by lei<hustlei@sina.cn>
'''

import torch
import numpy as np
import matplotlib.pyplot as plt

#数据集

##生成数据集
'''在线性方程y=2x+3上，随机取N个点，加上随机噪声，作为回归对象'''
N=200  #回归数据点个数。
x = torch.randn(N)  #使用torch的自动求梯度功能时，需要求梯度的量与numpy数组运算会出错。
y = 2*x + 3 + torch.randn(N)

##查看数据集
plt.scatter(x,y)
plt.plot(x, 2*x+3)
plt.show()


#线性回归

##初始化参数
w=torch.tensor(0.0, requires_grad=True)  #初始化权重(y=wx+b斜率)，注意只有浮点类型数据才能自动求梯度
b=torch.tensor(0.0, requires_grad=True)  #初始化偏置(y=wx+b截距)
lr=0.03  #递归下降学习率(迭代步长)，注意步长不能太大

##循环回归(最小二乘法)
for epoch in range(100):
    y_pre = w*x+b  #根据预估参数预测y
    loss = torch.sum((y_pre-y)**2)  #残差的平方(二乘)和表征误差。注意不能用np.sum
    loss.backward()  #反向传播求梯度
        
    w.data -= w.grad*lr/N  #更新权重。注意用-=，即inplace算法时，不能直接作用在leaf节点w上。
    b.data -= b.grad*lr/N  #更新偏置量。
        
    w.grad.data.zero_()  #清零梯度值。注意：对grad的数值data进行清零，避免把清零操作当作求梯度的一部分。
    b.grad.data.zero_()

    if (epoch+1)%10==0:  
        print(f"loop count:{epoch+1},weight:{w:.2f},bias:{b:.2f},loss:{loss:.2f}")
    # loss稳定后退出循环
    # if(abs(loss-losslast)<0.001):  #需要在重新计算loss前保存lastloss
        # break

with torch.no_grad():#因为w*x+b计算中会把tensor转换为numpy数组，但是自动求梯度的tensor不能转换
    plt.scatter(x,y)
    plt.plot(x,w*x+b)#直接用w.item()或者w.detach().numpy()也可以消除自动求梯度
    plt.show()

