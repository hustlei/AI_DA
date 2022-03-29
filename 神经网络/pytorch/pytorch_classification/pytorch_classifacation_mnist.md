pytoch人工神经网络基础：最简单的分类（softmax回归+交叉熵分类）
=====

# softmax回归分类原理

对于回归问题，可以用模型预测值与真实值比较，用均方误差这样的损失函数表示误差，迭代使误差最小训练模型。

那么分类问题是否可以用线性回归模型预测呢。最简单的方法就是用softmax方法，softmax的原理：

> + 以in_features个特征，out_features个类别为例。比如用花瓣大小、生长位置、花瓣形状三个因素，判断荷花和梅花，则in_features为3，out_feautures为2。

模型为

$$
y_1=x_{11}w_{11}+x_{12}w_{12}+x_{13}w_{13}+b_1 \\
y_2=x_{21}w_{21}+x_{22}w_{22}+x_{23}w_{23}+b_2
$$

+ out_features个分类，用输出值的大小表示分类，$y_1,y_2$数值最大的那个表示输出分类。

比如：labels=['荷花','梅花']。输出$y_1=1,y_2=3$，则表示预测结果为梅花。输出$y_1=2,y_2=1$表示预测结果为荷花。即输出为$argmax(y_i)$

+ softmax函数把输出标准化，softmax(y_1,y_2)输出两个值：

$$
out_1=\frac{exp(y_1)}{\sum{exp(y_i)}}\\
out_2=\frac{exp(y_2)}{\sum{exp(y_i)}}\\
$$

可以看出$out_1+out_2=1$，也就是说$out_i$是一个合法的概率分布，并且输入值大小和概率大小一直。因此，通常就把softmax计算结果，叫做out_features个分类的预测概率。即：$out_i$为labels[i]的概率。

> 从以上原理可以看出，多少个分类模型就需要几个输出(即线型方程个数)。
> 理论上讲，二分类问题，一个线性方程就ok了。但是使用softmax方法就必须2个方程，因为要为每个分类单独计算概率。

通常softmax函数会同时计算N个样本的数据，即输入形状为(N,in_feature)。函数实现如下：

```
def softmax(x):
    '''计算形状为(N,in_features)的x的softmax'''
    x_exp=torch.exp(x)
    return x_exp/x_exp.sum(dim=-1,keepdim=True)
```

# 交叉熵损失函数

> 交叉熵的含义可以参考[一文搞懂熵(Entropy),交叉熵(Cross-Entropy)](https://zhuanlan.zhihu.com/p/149186719)从浅入深讲的非常通俗易懂。这里就不再详述了，这里仅简单的说明一下。

对于有out_features个类别分类的问题，可以用softmax函数把输出转换为概率。按照把正确分类对应的输出概率最大化，不断迭代就可以完成模型训练。通常我们不直接用最大化概率的方法，而是对概率进一步进行运算处理。

+ 取对数的负值。$-log(p)$。因为概率p为0-1之间值，因此计算结果为∞~0。结果最小时，概率最大。
+ 对于分类问题，通常真实概率是这样的$p_true=[1,0,0,...]或者[0,1,0,..]或者[0,0,...,1]$结果只有一个值概率是1，其他值概率为0。$-log(p)p_true$就是交叉熵损失函数。

假如softmax输出概率为$p=[out_1,out_2]$，真实概率为$p_true=[0,1]$则交叉熵函数算法为

```
-log(p)*p_true
```

假如softmax输出概率为$p=[out_1,out_2]$，真实标签用序号表示$label_true=1$（结果为第一个标签，即梅花），则交叉熵函数算法为

```
p_log=-log(p)
loss=p_log[label_true]
```

实际上softmax通常会一次计算很多个样本，形状通常为(N,out_feature)，labels通常就是一个长度为N的数组，即每个样本只有一个结果，表示序号。因此交叉熵函数算法为：

```
def crossloss(y_softmax,y_label):#参数为torch.tensor类型
    '''根据softmax计算交叉熵。
    输入y为计算的N个样本的softmax概率,形状为(N,in_feautures)。
    y_label为真实标签，即每个样本真实分类的序号，形状为(N,)。'''
    log_softmax=-torch.log(y_softmax)
    index=y_label.view(-1,1) #转换为列形式
    entropy_arr=log_softmax.gather(dim=1,index=index) #label的值，即序号即概率为1，指选概率为1的值就是softmax的负对数的期望，即交叉熵。
    return entropy_arr.mean()
```


# softmax回归分类实现

```
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


# 训练模型


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
```



# 使用pytorch预定义函数实现softmax分类

pytorch内置了softmax和CrossEntropyLoss功能。


## pytorch的softmax

pytorch中的softmax函数用法：

+ torch.softmax(input, dim=None, dtype=None)
	+ input：输入数据，pytorch浮点数张量
	+ dim：对哪个维度进行softmax计算，通常我们都是对dim=-1计算。
	+ dtype：输出数据类型。

> torch.softmax实际上是`torch.nn.functional.softmax`的别名。

```python
>>> x=torch.arange(12,dtype=torch.float).view(4,3)
>>> x
tensor([[ 0.,  1.,  2.],
        [ 3.,  4.,  5.],
        [ 6.,  7.,  8.],
        [ 9., 10., 11.]])
>>> torch.softmax(x, dim=1)
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])
```

pytorch中softmax还可以作为层layer。因此nn模块中有一个Softmax类，根据Softmax类创建对象后可以当做函数用。

+nn.Softmax(dim=None)：Softmax类构造函数只有一个参数dim

```python

>>> layer=nn.Softmax(dim=-1)
>>> layer(x)
tensor([[0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652],
        [0.0900, 0.2447, 0.6652]])
```


## pytorch的CrossEntropyLoss

pytoch中的交叉熵损失函数类，实际上包含了softmax运算，因此，使用CrossEntropyLoss时，就不需要再使用softmax函数了。

+ crossloss=CrossEntropyLoss()：创建对象
+ crossloss(y_predict, y_true)：计算交叉熵
	+ y_predict：为模型前向传播计算得到的结果，形状为(N,out_features)
	+ y_true：为N个样本的分类序号，形状为(N,)

```python
import torch
from torch import nn, optim
import numpy as np
import matplotlib.pyplot as plt
```

## 准备数据

```python
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
```

## 建立模型

```python
class Net(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.layer1=nn.Linear(in_features,out_features)
        #nn.Softmax(dim=1)  #CrossEntropyLoss包含了softmax，log，和NLLloss
                
    def forward(self, x):
        return self.layer1(x)
    
net = Net(2,2)
```

## 配置模型

```python
nn.init.normal_(net.layer1.weight, mean=0, std=0.1)
nn.init.constant_(net.layer1.bias, val=0)
crossloss = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr=0.03)
```

## 训练模型

```python
for epoch in range(1001):
    y_predict = net(x_train)
    loss=crossloss(y_predict, y_train.view(-1)) #y_predict形状(N,2), y参数形状(N,)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    if(epoch%100==0):
        p = torch.softmax(y_predict,dim=-1).data.numpy()
        print(f"epoch:{epoch}, probability:{p[0]}, loss:{loss}")
```

## 查看结果

```python
# 可视化显示

plt.scatter(x_train[:,0], x_train[:,1], c=p[:,0]) #直接用分类1的概率表示颜色绘制
plt.colorbar()
plt.show()
```

```python
## 分别用0和1表示两个分类
p[p[:,0]>p[:,1]]=0
p[p[:,0]<p[:,1]]=1
plt.scatter(x_train[:,0], x_train[:,1], c=p[:,0]) #直接用分类1的概率表示颜色绘制
plt.colorbar()
plt.show()
```
