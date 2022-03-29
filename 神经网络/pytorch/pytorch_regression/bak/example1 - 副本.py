


import torch
from torch import nn, optim, functional as F
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms

#设置初始值

lr = 0.15    #学习率
gamma = 0.8    #优化算法参数
bs = 128    #每次小批量训练个数
epochs = 10    #循环次数

# 数据集

#导入数据
mnist = datasets.FashionMNIST(root="./Datasets", train=True, download=True, transform=transforms.ToTensor())
#制作数据集
data=DataLoader(mnist, batch_size=bs, shuffle=True,drop_last=False)
#检查数据
print(data.shape)
for x,y in data:
    print(x.shape)
    print(y.shape)
    break
# 输入的维度
input_ = mnist.data[0].numel()    #784
# 输出的维度
output_ = len(mnist.targets.unique())    #10


#定义神经网络模型
class Model(nn.Module)：
    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear1 = nn.Linear(in_features, 128, bias=True)    #128个神经元的全连接层
        self.output = nn.Linear(128, out_features, bias=True)    #
        
    def forward(self, x):
        x = x.view(-1, 28*28)    #一行784列向量
        sigma1 = torch.relu(self.linear1(x))    #全连接层+relu激活函数
        sigma2 = F.log_softmax(self.output(sigma1), dim = -1)    #结果映射到标签，用softmax激活


# 5定义训练流程

# 封装训练模型的函数
def fit(net, batchdata, lr, gamma, epochs):
# 参数：模型架构、数据、学习率、优化算法参数、遍历数据次数

    # 5.1 定义损失函数
    criterion = nn.NLLLoss()
    # 5.1 定义优化算法
    opt = optim.SGD(net.parameters(), lr = lr, momentum = gamma)
    
    # 监视进度：循环之前，一个样本都没有看过
    samples = 0
    # 监视准确度：循环之前，预测正确的个数为0
    corrects = 0
    
    # 全数据训练几次
    for epoch in range(epochs):
        # 对每个batch进行训练
        for batch_idx, (x, y) in enumerate(batchdata):
            # 保险起见，将标签转为1维，与样本对齐
            y = y.view(x.shape[0])
            
            # 5.2 正向传播
            sigma = net.forward(x)
            # 5.3 计算损失
            loss = criterion(sigma, y)
            # 5.4 反向传播
            loss.backward()
            # 5.5 更新梯度
            opt.step()
            # 5.6 梯度清零
            opt.zero_grad()
            
            # 监视进度：每训练一个batch，模型见过的数据就会增加x.shape[0]
            samples += x.shape[0]
            
            # 求解准确度：全部判断正确的样本量/已经看过的总样本量
            # 得到预测标签
            yhat = torch.max(sigma, -1)[1]
            # 将正确的加起来
            corrects += torch.sum(yhat == y)
            
            # 每200个batch和最后结束时，打印模型的进度
            if (batch_idx + 1) % 200 == 0 or batch_idx == (len(batchdata) - 1):
                # 监督模型进度
                print("Epoch{}:[{}/{} {: .0f}%], Loss:{:.6f}, Accuracy:{:.6f}".format(
                    epoch + 1
                    , samples
                    , epochs*len(batchdata.dataset)
                    , 100*samples/(epochs*len(batchdata.dataset))
                    , loss.data.item()
                    , float(100.0*corrects/samples)))

6训练模型

# 设置随机种子
torch.manual_seed(51)

# 实例化模型
net = Model(input_, output_)

# 训练模型
fit(net, batchdata, lr, gamma, epochs)

#https://zhuanlan.zhihu.com/p/472999423

#DataLoader用法及输出
#Linear参数
#nn.Module init参数
#Tensor.view