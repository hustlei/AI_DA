# 随机森林分类器（基于集合的分类器)
"""集合方法是将机器学习模型组合成更强大的机器学习模型的方法。随机森林是决策树的集合，是其中之一。它优于单一决策树，因为在保留预测能力的同时，它可以通过平均结果来减少过度拟合。"""

from sklearn import ensemble, datasets, metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

## 加载数据

dataset = datasets.load_breast_cancer()
data = dataset.data
data_names = dataset.feature_names
target = dataset.target
target_names = dataset.target_names

xtrain,  xtest, ytrain, ytest = train_test_split(data, target, test_size=0.4)

## 建立模型
model = ensemble.RandomForestClassifier()

## 训练模型
model.fit(xtrain, ytrain)

## 测试
ytest_pred = model.predict(xtest)

## 可视化
test_accuracy = model.score(xtest, ytest)
#test_accuracy = metrics.accuracy_score(ytest, ytest_pred)
print("test accuracy:{:.2%}".format(test_accuracy))

n = len(data_names)
plt.barh(range(n),model.feature_importances_)#, align='center') #feature_importances_总和为1
plt.yticks(range(n),data_names)
plt.gcf().subplots_adjust(left=.3,top=None,bottom=None, right=None)
plt.show()
