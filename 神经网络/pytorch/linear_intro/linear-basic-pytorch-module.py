#!/usr/bin/env python
# -*- coding: utf-8 -*-

#人工神经网络基础：线性回归(应用pytorch神经网络模型)
'''最小二乘法回归y=wx+b。应用pytorch神经网络模型
by lei<hustlei@sina.cn>
'''

import torch
from torch import nn
from matplotlib import pyplot as plt


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


#建立模型

## 构造简单单层神经网络
class LinearNet(nn.Module):  #定义神经网络模型
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.rand(1))    #定义需要拟合的参数。模型默认对Parameter参数自动求梯度。
        self.b = nn.Parameter(torch.zeros(1))   #w,b参数一般可以初始化值可以是随机值或者指定值。
        
    def forward(self, x):    #x为输入数据(待拟合或训练数据)，forward用x求预测值y。
        return self.w*x+self.b

#训练模型

##初始化参数
lr = 0.03    #学习率(迭代步长)，注意步长不能太大
layer=LinearNet()    #创建模型实例

##循环训练
for epoch in range(100):
    y_predict = layer(x)     #layer(x)自动执行forward
    loss = torch.mean((y_predict-y)**2)    #用残差的平方的均值(最小二乘法一般用平方和)最为损失函数。
    
    loss.backward()    #反向传播求梯度
    layer.w.data -= lr*layer.w.grad    #更新权重w。注意用-=，即inplace算法时，不能直接作用在leaf节点w上。
    layer.b.data -= lr*layer.b.grad    #更新偏置b。
    layer.w.grad.zero_()    #清零梯度值。
    layer.b.grad.zero_()

    if (epoch+1)%10==0:  
        print(f"loop count:{epoch+1},weight:{layer.w.item():.2f},bias:{layer.b.item():.2f},loss:{loss:.2f}")
    # loss稳定后退出循环
    # if(abs(loss-losslast)<0.001):  #需要在重新计算loss前保存lastloss
        # break

#结果显示
plt.scatter(x,y)
#tensor与numpy数组运算时，需要转换为numpy数组，但是自动求梯度的tensor不能转换
plt.plot(x,layer.w.item()*x+layer.b.item())#用w.item()或者w.detach().numpy()可以消除自动求梯度
plt.show()
