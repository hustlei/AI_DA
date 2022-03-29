#!/usr/bin/env python
# coding: utf-8

# 人工神经网络：简单图像分类（Fashion_mnist数据集）

import torch
from torch import nn, optim

from torchvision import datasets, transforms
from torchvison.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

# 准备数据

## 加载数据集

data_train = datasets.FashionMNIST(root="~/datasets/fashionminist", download=True, train=True, transform=transforms.ToTensor())
data_test = datasets.FashionMNIST(root="~/datasets/fashionmnist",download=True,train=False,transform=transforms.ToTensor())
labels_text = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')


## 查看数据集

print(len(data_train))#60000个元组，每个元组包含一个图片，一个整数(labels_text的序号)
fig = plt.figure()
axes = iter(fig.subplots(1,10))  #1行10列子图
for i in np.random.randint(0,6000,10): #随机显示10个数据
    x,y = data_train[i]  #获取图片和标签序号
    ax = next(axes)  #获取坐标系
    ax.imshow(x.view(28,28).numpy())  #图片张量数组转换为numpy方形数组，然后显示
    ax.set_title(labels_text[y])  #设置标题为文本标签
    ax.axis(False)  #不显示坐标系
plt.show()

## 数据批量读取

batch_size = 100  #批量规模
train_iter = DataLoader(data_train, batch_zise=batch_size, shuffle=True)
test_iter = DataLoader(data_test, batch_size=batch_size, shuffle=True)

# 建立模型

class Net(nn.Module):#线性模型
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1=nn.Linear(in_features, out_features)
        
    def forward(self, x):
        return self.layer1(x)

in_features = 28*28  #一个图片的数据点数
out_features = 10   #10个分类
net = Net(in_features, out_features)

# 配置模型

