#线性回归
import torch
import numpy as np
import matplotlib.pyplot as plt

# 人工生成数据集
w_true = np.array([2, -3.4])
b_true = 4.2
x = np.random.normal(size=(1000,2))
y = x @ w_true + b_true + np.random.normal(0,0.01,size=1000)

features = torch.from_numpy(x)  #1000行，2列
labels = torch.from_numpy(y)   #1000

#ax=plt.figure().add_subplot(111,projection='3d')
#ax.plot(x[:,0],x[:,1],y)
plt.scatter(x[:,1],y)
plt.show()

# 批量读取数据
batch_size = 10

def data_iter(features, labels, batch_size):
  indices = np.arange(1000)
  np.random.shuffle(indices)

  start_i = 0
  if batch_size <1000:
    for end_i in range(batch_size,1000,batch_size):
      yield features[start_i:end_i,:], labels[start_i:end_i]

#for x,y in data_iter(features,labels,10):
#  print(x,y)
#  break

# 线性模型
def linear(X,w,b):
  return torch.mm(X,w)+b
# 定义损失函数
def mse_loss(y_pre,y):
  return torch.mean((y_pre-y)**2)
# 定义优化算法
def sgd(params,lr):#仅更新梯度
  for param in params:
    param.data -= lr*param.grad
    
    
# 训练模型
# 设置超参数
# 迭代周期个数和学习率都是超参数。大多超参数需要通过反复试错来不断调节。理论上迭代周期越大越有效，但是时间也会越长。

lr=0.03
num_epoches=1000

# 设置初始化参数

w=torch.rand([2,1],dtype=features.dtype,requires_grad=True)
b=w.new_zeros(1,requires_grad=True)

net=linear

for epoch in range(num_epoches):
  for X,y in data_iter(features,labels,batch_size):
    y_pre=net(X,w,b)  #神经网络预报结果
    loss=mse_loss(y_pre,y) #小批量损失函数
    loss.backward()  #求梯度
    sgd([w,b],lr)  #更新权重参数

    w.grad.data.zero_()  #梯度清零
    b.grad.data.zero_()  #梯度清零
  if epoch%10==0:
    print(f'epoch {epoch},loss:{loss},w:{w[0,0].item()}{w[1,0].item()},b:{b.item()}')