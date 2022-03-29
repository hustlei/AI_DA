#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''最小二乘法回归y=wx+b。
by lei<hustlei@sina.cn>
'''

import numpy as np
import matplotlib.pyplot as plt

#回归数据集

##生成数据集
'''在线性方程y=2x+3上，随机取N个点，加上随机噪声，作为回归对象'''
N=200  #回归数据点个数。
np.random.seed(2022)  #设置随机种子，保证重复运行，生成的随机数相同。
x = np.random.normal(size=N)
y = 2*x + 3 + np.random.normal(size=N)

##查看数据集
plt.scatter(x,y)
plt.plot(x, 2*x+3)
plt.show()


#线性回归

##初始化参数
w=0  #初始化权重(y=wx+b斜率)
b=0  #初始化偏置(y=wx+b截距)
lr=0.03  #递归下降学习率(迭代步长)

##循环回归(最小二乘法)
for epoch in range(100):
    y_pre = w*x+b  #根据预估参数预测y
    loss = np.sum((y_pre-y)**2)  #残差的平方(二乘)和表征误差。误差最小时，拟合精度最高。
    grad_w = np.mean(2*(y_pre-y)*x)  #梯度公式为np.sum(2*(y_pre-y)*x)。除以N，可以让梯度和权重w在同一个数量级。
    grad_b = np.mean(2*(y_pre-y))  #计算梯度
    w -= grad_w*lr  #更新权重
    b -= grad_b*lr  #更新偏置量
    if (epoch+1)%10==0:  
        print(f"loop count:{epoch+1},weight:{w:.2f},bias:{b:.2f},loss:{loss:.2f}")
    # loss稳定后退出循环
    # if(abs(loss-losslast)<0.001):  #需要在重新计算loss前保存lastloss
        # break
        
plt.scatter(x,y)
plt.plot(x,w*x+b)
plt.show()




"""最小二乘法，最优数学解
根据数学公式，残差的平方和最小值在导数为0处。
w的最优解$w=\frac{X^Ty-\overline x \overline y}{X^TX-\overline x^2}$，其中X为列向量。
用numpy求解：

x=x[:,None]
w=(x.T@y-np.mean(x)*np.mean(y))/(x.T@x-np.mean(x)**2)
print(w.item())  #最优解为2.016666

b的最优解：$b=\overline y - w \overline x$。

b=np.mean(y)-w*np.mean(x)   #b最优解为2.97
"""