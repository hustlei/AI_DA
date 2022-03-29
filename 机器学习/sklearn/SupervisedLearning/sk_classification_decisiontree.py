# 决策树分类
"""决策树基本上是二叉树流程图，其中每个节点根据一些特征变量分割一组观察。"""

from sklearn import tree, metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

## 加载数据
"""数据集为男生和女生身高和头发长度数据，包含19组"""
data = [[165,19],[175,32],[136,35],[174,65],[141,28],[176,15],[131,32],[166,6],[128,32],[179,10],[136,34],[186,2],[126,25],[176,28],[112,38],[169,9],[171,36],[116,25],[196,25]]
data_feature_names = ['height','length of hair']
target = ['Man','Woman','Woman','Man','Woman','Man','Woman','Man','Woman', 'Man','Woman','Man','Woman','Woman','Woman','Man','Woman','Woman','Man']

xtrain, xtest, ytrain, ytest = train_test_split(data, target, test_size=0.4)

## 建立模型
model = tree.DecisionTreeClassifier()

## 训练数据
model.fit(xtrain, ytrain)

## 测试
ytest_pred = model.predict(xtest)

## 可视化
success_rate = metrics.accuracy_score(ytest, ytest_pred)
print("{:.2%}".format(success_rate))

#plt.figure()
tree.plot_tree(model, filled=True)
plt.show()