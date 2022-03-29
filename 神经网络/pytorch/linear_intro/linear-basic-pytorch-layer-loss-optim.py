#!/usr/bin/env python
# -*- coding: utf-8 -*-

#人工神经网络基础：线性回归(应用pytorch预定义层、损失函数及优化方法)
'''多元线性回归y=w1x1+w2x2+...+wnxn+b。应用pytorch神经网络模型

+ 数据准备
+ 模型定义
    + 神经网络模型定义(网络层定义）
    + 损失函数（均方误差）
    + 优化方法（梯度下降）
+ 模型训练
+ 结果测试
+ 结果可视化

by lei<hustlei@sina.cn>
'''

import torch
from torch import nn, optim
from matplotlib import pyplot as plt


#数据准备

##生成数据集
'''在线性方程y=x1+2x2+3上，随机取N个点，加上随机噪声，作为回归对象'''
N=200  #回归数据点个数。
x = torch.randn(N, 2)  #输入数据x（N行2列），最后一维和多元方程x变量元数相同。x需是浮点数张量类型。
y = x[:,0] +2*x[:,1] + 3 + torch.randn(N) #加上随机噪声
y = y[:,None]  #转换为N行1列二维张量。对二元线性方程y=x1+2x2+3，nn.Linear的输出形状为(N,1)

##查看数据集
ax=plt.figure().add_subplot(projection="3d")
ax.scatter(x[:,0],x[:,1],y)  #绘制数据散点图
ax.plot_trisurf(x[:,0],x[:,1],x[:,0]+2*x[:,1]+3,alpha=0.5)  #绘制线性方程平面
plt.show()


#建立模型

## 构造单层神经网络
class LinearNet(nn.Module):  #定义神经网络模型。
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(2,1)   #创建一个二元线性的线性层(即输入x为2元，输出y为1元)。
        '''nn.Linear类可以创建一个线性层对象，创建的对象可以像函数一样调用，如：linear(x)。
        线性层对象有weight和bias两个可学习参数。'''
        
    def forward(self, x):    #x为输入数据(待拟合或训练数据)，forward用x求预测值y。
        return self.linear(x)

lr = 0.03    #学习率(迭代步长)，注意步长不能太大
net = LinearNet()  #实例化神经网络模型。可以当作函数直接执行forward函数。可以通过parameters()函数可以获取所有成员的可学习参数。
#net = nn.Linear(2,1) #对于单层线性模型，直接用Linear代替Module也是可以的。用法一样。
mseloss = nn.MSELoss()  #用MSELoss类创建均方误差对象，mseloss可以作为损失函数调用。用法:mseloss(y_predict,y).backward()
opt = optim.SGD(net.parameters(),lr) #用SGD类创建递归下降算法优化对象。可以用step()函数更新学习参数，zero_grad()函数清零参数梯度。

#训练模型
for epoch in range(100):
    y_predict = net(x)     #net(x)自动执行forward
    loss = mseloss(y_predict, y)  #调用军方误差对象计算损失函数。torch.mean((y_predict-y)**2)，也是L2范数
    loss.backward()    #反向传播求梯度
    opt.step()  #更新梯度
    opt.zero_grad()    #清零梯度值

    if (epoch+1)%10==0:
        w = list(net.parameters())[0]
        b = list(net.parameters())[1]
        print(f"loop count:{epoch+1}")
        print(f"weight:{w[0][0].item():.2f},{w[0][1].item():.2f}")
        print(f"bias:{b.item():.2f}")
        print(f"loss:{loss:.2f}")
        print("---------------")


#结果显示
ax.scatter(x[:,0],x[:,1],y) #绘制训练数据点(拟合用的数据)
x1,x2=x[:,0],x[:,1]
w = list(net.parameters())[0]
b = list(net.parameters())[1]
y=(x@w.data.T+b.data).squeeze()
ax.plot_trisurf(x1,x2,y)#weight.detach().numpy()或weight.data可以消除自动求梯度。
plt.show()




## 只用pytorch的autograd
"""
#!/usr/bin/env python
# -*- coding: utf-8 -*-



import torch
import numpy as np
import matplotlib.pyplot as plt

#数据集

##生成数据集
'''在线性方程y=x1+2x2+3上，随机取N个点，加上随机噪声，作为回归对象'''
N=200  #回归数据点个数。
x = torch.randn(N, 2)  #输入数据x（N行2列），最后一维和多元方程x变量元数相同。x需是浮点数张量类型。
y = x[:,0] +2*x[:,1] + 3 + torch.randn(N) #加上随机噪声

#线性回归

##初始化参数
w1=torch.tensor(0.0, requires_grad=True)
w2=torch.tensor(0.0, requires_grad=True)  #初始化权重(y=wx+b斜率)，注意只有浮点类型数据才能自动求梯度
b=torch.tensor(0.0, requires_grad=True)  #初始化偏置(y=wx+b截距)
lr=0.03  #递归下降学习率(迭代步长)，注意步长不能太大

##循环回归(最小二乘法)
for epoch in range(100):
    y_pre = w1*x[:,0]+w2*x[:,1]+b  #根据预估参数预测y
    loss = torch.mean((y_pre-y)**2)  #残差的平方(二乘)和表征误差。注意不能用np.sum
    loss.backward()  #反向传播求梯度
        
    w1.data -= w1.grad*lr
    w2.data -= w2.grad*lr
    b.data -= b.grad*lr  #更新偏置量。
        
    w1.grad.data.zero_()
    w2.grad.data.zero_()
    b.grad.data.zero_()

    if (epoch+1)%10==0:  
        print(f"loop count:{epoch+1},weight:{w1.data:.2f},{w2.data:.2f},bias:{b:.2f},loss:{loss:.2f}")
"""
