#!/usr/bin/env python
# coding: utf-8

# pytorch tensor学习

import numpy as np
import torch
from torch.autograd import variable
from matplotlib import pyplot as plt

#定义图像和三维格式坐标轴
#fig=plt.figure()
#ax = Axes3D(fig)
#ax.plot3D(xs,ys,zs,'gray')    #绘制空间曲线
 #显示图像
#plt.show()

#====================
#创建
#====================

print("#直接创建tensor")
a=torch.empty(2,3) #与torch.Tensor(2,3)相同，创建一个2x3的未初始化张量
b=torch.zeros(2,3) #创建一个全部为0的2x3的张量
c=torch.ones(2,3) #创建一个全部为1的2x3的张量
d=torch.eye(2,3) #创建一个对角为1的张量
e=torch.full((2,3),4) #创建一个全部为指定数字的张量
print("a:",a)
print("b:",b)
print("c:",c)
print("d:",d)
print("e:",e)
#以下方式也可以
#shape=(2,3,)
#torch.empty(shape)
#torch.ones(shape) #创建一个大小为shape的张量

print("#从list或者numpy创建tensor")
a=torch.tensor([1,3])
b=torch.from_numpy(np.array([2,2]))
print(a)
print(b)

print("#根据已有tensor创建，所有tesor在新地址创建")
a=torch.tensor([1,3])
b=a.new_ones(2,3)#返回的tensor具有相同的torch.dtype和torch.device
c=a.new_ones(2,3,dtype=torch.float64)

d=torch.empty_like(c)#使用tesor的shape dtype device，创建新tesor
d=torch.zeros_like(c)
d=torch.ones_like(c)#no eye_like
d=torch.full_like(c,4)
e=torch.rand_like(c)
#e=torch.randint_like(c,0,5)
#e=torch.randn_like(c)
print(a)
print(b)
print(c)
print(d)

print("#获取tensor的形状,返回值是个tupe")
print(d.size())
print(d.shape)
print(d.dtype)
print(d.device)

print("根据范围和间距创建tesor")
a=torch.arange(0,5,1)#[0,5)步长为1
b=torch.range(0,5,1)#[0,5]步长为1
c=torch.linspace(0,5,6)#[0,5]分成6步
d=torch.logspace(0,3,4,base=10.0)#base的0次方到3次方，分成4步
print(a)
print(b)
print(c)
print(d)

print("随机数创建tesor")
a=torch.rand(2,2)#[0,1)均匀分布
#b=torch.randint(5,(2,2))#[0,5)整数均匀分布
b=torch.randint(1,5,(2,2))#[1,5)整数均匀分布
c=torch.randperm(10)#[0,10)整数随机分布
d=torch.randn(2,2)#标准正态分布（均值/期望为0，方差为1）
#e=torch.normal(0,1,(2,2))#正态分布
#f=torch.bernoulli(torch.empty(2,2))
#g=torch.poisson(torch.rand(4, 4))
print(a)
print(b)
print(c)
print(d)
#print(e)


#====================
#操作
#====================


