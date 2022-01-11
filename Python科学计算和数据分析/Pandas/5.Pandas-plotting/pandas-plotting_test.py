import numpy as np
import pandas as pd
import matplotlib.pyplot as plt




# plotting

## start

'''
# 导入库
import pandas as pd
import matplotlib.pyplot as plt

# 初始化数据
d=pd.DataFrame({'x':[-3,-2,-1,0,1,2,3]})
d['y']=d['x']**2

# 绘图
d.plot.line(x='x', y='y')      #调用plot模块的line方法绘制折线图

# 显示
plt.show()
'''








## simple plotting

### simplest using

'''
np.random.seed(20211231)

x=np.arange(5)
y=np.random.randint(10,size=5)
d=pd.DataFrame({'x':x,'y':y})

d.plot.scatter(x='x',y='y')
d.plot.line(x='x',y='y')
d.plot.area(x='x',y='y')
d.plot.bar(x='x',y='y')
d.plot.barh(x='x',y='y')
d.plot.pie(y='y')
'''

'''
s=pd.Series([1,2,3,4,5])

s.plot.line()
s.plot.area()
s.plot.bar()
s.plot.barh()
s.plot.pie()
'''









s=pd.Series([1,2,1,3])
df=pd.DataFrame({'col1':[1,2,3,4],'col2':[2,0,2.5,4],'col3':[3,1,2,1.5]},index=['a','b','c','d'])


### line

'''
df.plot.line() 
df.plot.line(x='col1')
df.plot.line(x='col1', y='col2')
df.plot.line(x='col1', y=['col2','col3'])
df.plot.line( y=['col2','col3'])
'''
#df.plot.line(x=['col1','col1'], y=['col2','col3'])
df.plot.line(y=['col2','col3'],style=['r:','-'],legend=False)



### area

'''
df.plot.area()  #
df.plot.area(x='col1')
df.plot.area(x='col1', y='col2')
df.plot.area(x='col1', y=['col2','col3'],stacked=False)
#df.plot.area(x=['col1','col1'], y=['col2','col3'])
#stacked表示曲线叠加
'''

### bar


'''
df.plot.bar(stacked=True)  #
df.plot.bar(x='col1')
df.plot.bar(x='col1', y='col2')
df.plot.bar(x='col1', y=['col2','col3'])
#df.plot.bar(x=['col1','col1'], y=['col2','col3'])
'''

### scatter

'''
#df.plot.scatter()  #
#df.plot.scatter(x='col1')
df.plot.scatter(x='col1', y='col2')
#df.plot.scatter(x='col1', y=['col2','col3'])
df.plot.scatter(x=['col1','col1'], y=['col2','col3'])
'''

### pie

#不需要x坐标参数
'''
df.plot.pie(y='col1')
df.plot.pie(subplots=True,layout=(2,2))  #
#df.plot.pie(y=['col2','col3'],subplots=True)出错
'''

## 统计绘图

### hist
#s.plot.hist()
'''
df.plot.hist(bins=4,stacked=True)
'''
#s.plot.hist()
#df.plot.hist(by='col2',stacked=True)

### density 核密度估计：非参数方法估算随机变量的概率密度函数(用函数使用高斯核，并自动确定带宽)

'''
s.plot.density()
df.plot.kde()
'''

### kde

### box

### hexbin
#df.plot.hexbin(x=['col1'],y=['col2','col2'])

#df.hist(column='col1')




plt.show()