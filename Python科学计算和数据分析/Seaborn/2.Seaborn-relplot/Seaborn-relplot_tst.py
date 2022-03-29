#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
#relplot——simplest
################################

'''
import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt    #显示图形还是需要依靠matplotlib
tips=sns.load_dataset('tips')

#sns.relplot(data=tips,x='total_bill',y='tip') #观察账单和小费的关系，绘制散点图
#sns.relplot(data=tips,x='total_bill',y='tip',kind='line') #观察账单和小费的关系，绘制折线图

plt.show()
'''

################################
#relplot——散点图——hue分组
################################

'''
import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt    #显示图形还是需要依靠matplotlib
tips=sns.load_dataset('tips')

#hue分组
sns.relplot(data=tips,x='total_bill',y='tip',hue='day') #分别观察每天的账单和小费的关系。
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker') #分别观察抽烟和不抽烟人群的账单和小费的关系。
sns.relplot(data=tips,x='total_bill',y='tip',hue='size') #根据就餐人数，分别观察人群的账单和小费的关系。

#hue+palette颜色
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',palette='coolwarm')  #用colormap指定颜色
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',palette='ch:start=1.2,rot=.5') #用cubehelix参数指定颜色
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',palette=('r','g','b','gray')) #用列表参数指定颜色

plt.show()
'''

################################
#relplot——散点图——style分组
################################
'''
import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt    #显示图形还是需要依靠matplotlib
tips=sns.load_dataset('tips')

#style分组
sns.relplot(data=tips,x='total_bill',y='tip',style="smoker")  #抽烟的人一个样式，不抽烟的人一个样式。
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',style="smoker")  #组合样式
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',style="time") #组合样式

#style+marker
sns.relplot(data=tips,x='total_bill',y='tip',style="time",markers=["o","*"]) #用markers参数指定style分组点样式

plt.show()
'''

################################
#relplot——散点图——size分组
################################
'''
import seaborn as sns
import matplotlib.pyplot as plt
tips=sns.load_dataset('tips')

#size分组
sns.relplot(data=tips,x='total_bill',y='tip',size="smoker")#size分组(根据是否抽烟，分组)
sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',size="size")#hue+size
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',style="size",size="smoker")#hue+style+size

#设置大小数值
sns.relplot(data=tips,x='total_bill',y='tip',hue='day',size="size",sizes=(10,200)) #把size映射到10-200区间，用于显示大小

plt.show()
'''



################################
#relplot多子图
################################
'''
import seaborn as sns
import matplotlib.pyplot as plt
tips=sns.load_dataset('tips')

#sns.relplot(data=tips,x='total_bill',y='tip',col='smoker')     #不同子图绘制smoker不同的图形，绘制在不同的行 
#sns.relplot(data=tips,x='total_bill',y='tip',row='smoker',col='time') #用不同的行绘制子图区分smoker，不同的列子图区分time
#sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',col='time') #根据time绘制多列子图，每个子图用hue分组
#sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',col='day',col_wrap=2) #每列最多2个子图，超过2个自动换行
plt.show()
'''





################################
#relplot折线图--简单（排序）
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
df=pd.DataFrame(np.random.randn(500, 2).cumsum(axis=0), columns=["x", "y"])


sns.relplot(data=df,x="x",y="y",kind="line")
sns.relplot(data=df,x="x",y="y",kind="line",sort=False)
plt.show()
'''


################################
#relplot折线图——置信区间
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

#sns.relplot(data=fmri,x="timepoint",y="signal",kind="line")
#sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",ci=None) #不显示置信区间
#sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",ci="sd") #显示标准偏差
#sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",ci=50) #指定置信区间数值

plt.show()
'''

################################
#relplot折线图——置信区间
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",estimator=None) #不聚合
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",estimator=sum) #聚合函数为sum

plt.show()
'''

################################
#relplot折线图——分组
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event") #hue分组
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",style="event") #style分组
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",size="event") #size分组
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event",style="region") #hue+style
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event",size="region") #hue+size
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",style="event",size="region") #hue+size
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event",style="event",size="region") #hue+style+size

plt.show()
'''

################################
#relplot折线图——分组样式
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="region",palette="hsv") #颜色样式
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",style="region",markers=["o","*"]) #style点样式
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",style="region",dashes=[[5,1],[1,0]]) #style线样式
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",size="region",sizes=[0.5,2]) #size大小

plt.show()
'''


################################
#scatterplot
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('tips')

sns.scatterplot(data=fmri,x="total_bill",y="tip")

plt.show()
'''
'''
import matplotlib.pyplot as plt
import seaborn as sns
tips=sns.load_dataset('tips')
fig=plt.figure()
ax1=fig.add_subplot(121)
ax2=fig.add_subplot(122)
#ax1,ax2=fig.subplots(1,2)

sns.scatterplot(data=tips,x="total_bill",y="tip",hue='smoker',ax=ax1)
sns.scatterplot(data=tips,x="total_bill",y="tip",hue='smoker',size='sex',ax=ax2)

plt.show()
'''


################################
#lineplot
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.lineplot(data=fmri,x="timepoint",y="signal")

plt.show()
'''


import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')
fig=plt.figure()
(ax1,ax2),(ax3,ax4)=fig.subplots(2,2)

sns.lineplot(data=fmri,x="timepoint",y="signal",ax=ax1)
sns.lineplot(data=fmri,x="timepoint",y="signal",hue='region',ax=ax2)
sns.lineplot(data=fmri,x="timepoint",y="signal",hue='region',style='event',ax=ax3)
sns.lineplot(data=fmri,x="timepoint",y="signal",hue='region',size='event',ax=ax4)

plt.show()
