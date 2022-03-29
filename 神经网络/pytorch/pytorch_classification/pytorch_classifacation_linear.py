#!/usr/bin/env python
# coding: utf-8

# 人工神经网络基础：最简单的分类（softmax回归分类）
# =====
# 
# 对于回归问题，用损失函数表示神经网路预测结果与已知结果的误差，通过训练，当损失函数最小时表示训练结束。




import torch
from torch import nn, optim
import matplotlib.pyplot as plt

# 准备数据

N=100
x_train1=torch.tensor([1,1])+torch.normal(mean=0, std=0.2, size=[N//2,2])
y_train1=torch.zeros(N//2,1,dtype=torch.long)
x_train2=torch.tensor([2,2])+torch.normal(mean=0, std=0.2, size=[N//2,2])
y_train2=torch.ones(N//2,1,dtype=torch.long)

x_train=torch.cat([x_train1, x_train2])
y_train=torch.cat([y_train1, y_train2])

# 显示数据
plt.scatter(x_train[:,0],x_train[:,1],c=y_train[:,0])
plt.colorbar()
plt.show()


# 建立模型

def softmax(x):
    '''计算形状为(N,in_featrues)的x的softmax'''
    x_exp=torch.exp(x)
    return x_exp/x_exp.sum(dim=-1,keepdim=True)

class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1=nn.Linear(in_features,out_features)
                
    def forward(self, x):
        return softmax(self.layer1(x))
    
net = Net(2,2)

# 配置模型

def crossloss(y_softmax,y_label):
    '''根据softmax计算交叉熵。
    输入y为计算的N个样本的softmax概率,形状为(N,in_feautures)。
    y_true为真实标签，即每个样本真实分类的序号，形状为(N,)。'''
    log_softmax=-torch.log(y_softmax)
    label=y_label.view(-1,1) #转换为列形式
    entropy_arr=log_softmax.gather(dim=1,index=label) #label的值，即序号即概率为1，指选概率为1的值就是softmax的负对数的期望，即交叉熵。
    return entropy_arr.mean()

nn.init.normal_(net.layer1.weight, mean=0, std=0.1)
nn.init.constant_(net.layer1.bias, val=0)
optimizer = optim.SGD(net.parameters(),lr=0.03)


# # 训练模型


for epoch in range(1001):
    y_predict = net(x_train)
    loss=crossloss(y_predict, y_train.view(-1)) #y_predict形状(N,2), y参数形状(N,)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if(epoch%100==0):
        print(f"epoch:{epoch}, loss:{loss.data.item()}")


# 查看结果

p = softmax(y_predict).data.numpy()
p1 = p[:,0]<p[:,1]
plt.scatter(x_train[:,0], x_train[:,1], c=p1) #直接用分类1的概率表示颜色绘制
plt.colorbar()
plt.show()
