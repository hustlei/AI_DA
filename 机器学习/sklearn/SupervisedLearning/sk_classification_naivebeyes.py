# 朴素贝叶斯分类(有监督的学习)

from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB, CategoricalNB, MultinomialNB, ComplementNB, BernoulliNB

## 导入数据
datas = datasets.load_breast_cancer() #威斯康星乳腺癌诊断数据，569组数据，每组包含30个数值

data = datas.data #or datas["data"]
dataname = datas.feature_names
target = datas.target
targetname = datas.target_names

train, test, target_train, target_test = train_test_split(data, target, train_size=0.6)  #朴素贝叶斯假设简单，使用数据量少，即使使用10%的数据也可能得到不错的预测效果
"""random_state=42,可以保证每次运行结果都一致"""

## 构建模型
nb = GaussianNB()
# nb = BernoulliNB() #伯努利分布下的朴素贝叶斯。适用与离散数据，本实例误差较大
# nb = CategoricalNB() #不能用
# nb = MultinomialNB() #多项式分布下的朴素贝叶斯。
# nb = ComplementNB() #补集朴素贝叶斯

## 训练模型
model = nb.fit(train, target_train)

## 测试评估
test_pred = model.predict(test)

"""
import numpy as np
rst = np.bitwise_xor(target_test, test_pred)
print("wrong number:", sum(rst))
print("correct rate:{:.3%}".format(1-sum(rst)/(len(rst))))
"""

from sklearn.metrics import accuracy_score
print("{:.2%}".format(accuracy_score(target_test, test_pred)))