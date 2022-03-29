#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
#Basic numpy to seaborn
################################
'''
import numpy as np
import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt    #显示图形还是需要依靠matplotlib

#准备数据
x=np.linspace(0, 4*np.pi,100)
y=np.sin(x)

#绘图
sns.relplot(x=x,y=y,kind="line")   #用关系绘图函数绘制折线图

#显示图形
plt.show()
'''

################################
#Basic pandas to seaborn
################################
'''
import pandas as pd
import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt    #显示图形还是需要依靠matplotlib

#准备数据
df=pd.DataFrame({'x':[1,2,3,4,5] ,'y':[1,3,2,4,5]})

#绘图
sns.relplot(x='x',y='y',data=df,kind="line")   #用关系绘图函数绘制折线图

#显示图形
plt.show()
'''

################################
#短格式
################################
'''
import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt    #显示图形还是需要依靠matplotlib

#准备数据
df=sns.load_dataset("flights")      #从网络加载指定数据集。
df_wide=df.pivot(index="year",columns="month",values="passengers")  #转换为短格式


sns.relplot(data=df_wide,kind="line")  
#sns.relplot(data=df_wide.T,kind="line")  

#显示图形
plt.show()
'''



################################
#长格式
################################
'''
import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt      #显示图形还是需要依靠matplotlib

df=sns.load_dataset("flights")       #准备数据，从网络加载指定数据集。
sns.relplot(data=df,x="year",y="passengers",hue='month',kind="line")  #绘图
#sns.relplot(data=df,x="month",y="passengers",hue='year',kind="line")  #绘图
plt.show()                           #显示
'''

################################
#凌乱数据
################################

import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt      #显示图形还是需要依靠matplotlib

df=sns.load_dataset("flights")       #准备数据，从网络加载指定数据集。
sns.relplot(data=df,x="year",y="passengers",hue='month',kind="line")  #绘图
#sns.relplot(data=df,x="month",y="passengers",hue='year',kind="line")  #绘图
plt.show()                           #显示
