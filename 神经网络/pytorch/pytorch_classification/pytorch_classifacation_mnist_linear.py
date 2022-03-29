#!/usr/bin/env python
# coding: utf-8

'''人工神经网络：图像分类（线性单层模型，Fashion_mnist数据集）

pytorch-classification-linear-FashionMNIST.py

@author：lileilei<hustlei@sina.cn>
'''

import torch
from torch import nn, optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

import numpy as np
import matplotlib.pyplot as plt

device = torch.device('cpu')
if torch.cuda.is_available():  #判断是否GPU可用
    print("cuda.device_count(GPU数量):", torch.cuda.device_count())
    gpu_index = torch.cuda.current_device
    print("cuda.current_device(当前GPU索引号)",  gpu_index)
    print("device_name(当前GPU名字):", torch.cuda.get_device_name(gpu_index))
    device = torch.device('cuda')

# 准备数据

## 加载数据集

data_train = datasets.FashionMNIST(root="~/Datasets", download=True, train=True,
                                   transform=transforms.ToTensor())
data_test = datasets.FashionMNIST(root="~/Datasets", download=True, train=False,
                                  transform=transforms.ToTensor())
labels_text = ('t-shirt', 'trouser', 'pullover', 'dress', 'coat', 'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot')

## 查看数据集

print(len(data_train))  # 60000个元组，每个元组包含一个图片，一个整数(labels_text的序号)
fig = plt.figure(figsize=(10, 2))
axes = iter(fig.subplots(1, 10))  # 1行10列子图
for i in np.random.randint(0, 6000, 10):  # 随机显示10个数据
    x, y = data_train[i]  # 获取图片和标签序号
    ax = next(axes)  # 获取坐标系
    ax.imshow(x.view(28, 28).numpy())  # 图片张量数组转换为numpy方形数组，然后显示
    ax.set_title(labels_text[y])  # 设置标题为文本标签
    ax.axis(False)  # 不显示坐标系
plt.show()

## 数据批量读取

batch_size = 100  # 批量规模
# num_workers = 2  # 多进程允许进程个数
train_iter = DataLoader(data_train, batch_size=batch_size, shuffle=True)
test_iter = DataLoader(data_test, batch_size=batch_size, shuffle=True)


# 建立模型

class Net(nn.Module):  # 线性模型
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1 = nn.Linear(in_features, out_features)

    def forward(self, x):
        return self.layer1(x)


in_features = 28 * 28  # 一个图片的数据点数
out_features = 10  # 10个分类
net = Net(in_features, out_features)

net.to(device)

# 配置模型
lr = 0.03
nn.init.normal_(net.layer1.weight, mean=0, std=0.1)
nn.init.constant_(net.layer1.bias, 0)
crossloss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=lr)

# 训练模型

import time
time_start=time.time()

for epoch in range(501):  # 迭代训练
    accuracy_list=[]
    for x, y in train_iter:  # 小批量训练
        y_predict = net(x.view(batch_size, -1))  # 前向传播预测
        loss = crossloss(y_predict, y)  # 计算损失函数
        loss.backward()  # 后向传播
        optimizer.step()  # 更新梯度
        optimizer.zero_grad()  # 清零梯度数据
                
        accuracy = (y_predict.argmax(dim=-1) == y).float().mean()  # 计算准确比例
        accuracy_list.append(accuracy)

    if epoch % 10 == 0:  # 每迭代10次显示相关计算数据
        print(f'epoch:{epoch},loss:{loss:.2f},本次训练输出精确度:{np.mean(accuracy_list):.2f}')

        
print(f"训练结束：用时{time.time()-time_start:.2f}s")

# 测试
test_accuracy_list = []

for x_test, y_test in test_iter:
    y_predict_test = net(x_test.view(batch_size, -1))
    accuracy = (y_predict_test.argmax(dim=-1) == y_test).float().mean()
    test_accuracy_list.append(accuracy)
    
print(f"accuracy of test:{np.mean(test_accuracy_list):.4f}")