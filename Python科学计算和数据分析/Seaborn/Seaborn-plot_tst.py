#!/usr/bin/env python
# -*- coding:utf-8 -*-


################################
#relplot
################################
'''
import seaborn as sns                #惯例将seaborn导入为sns
import matplotlib.pyplot as plt    #显示图形还是需要依靠matplotlib

tips=sns.load_dataset('tips')

#sns.relplot(data=tips,x='total_bill',y='tip') #观察账单和小费的关系，绘制散点图
#sns.relplot(data=tips,x='total_bill',y='tip',kind='line') #观察账单和小费的关系，绘制折线图

#分组绘制多条曲线
#sns.relplot(data=tips,x='total_bill',y='tip',hue='day') #分别观察每天的账单和小费的关系。
#sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker') #分别观察抽烟和不抽烟人群的账单和小费的关系。
#sns.relplot(data=tips,x='total_bill',y='tip',hue='size') #根据就餐人数，分别观察人群的账单和小费的关系。

sns.relplot(data=tips,x='total_bill',y='tip',hue='day',palette='coolwarm')  #用colormap指定颜色
#sns.relplot(data=tips,x='total_bill',y='tip',hue='day',palette='ch:start=1.2,rot=.5') #用cubehelix参数指定颜色

#区分样式
#sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',style="smoker")  #抽烟的人一个样式，不抽烟的人一个样式。
#sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',style="time") #不同的时间段用不同的样式。

#区分大小
#sns.relplot(data=tips,x='total_bill',y='tip',hue='smoker',size="size")  #抽烟的人一个样式，不抽烟的人一个样式。
#sns.relplot(data=tips,x='total_bill',y='tip',hue='day',size="smoker") #不同的时间段用不同的样式。

#设置大小数值
#sns.relplot(data=tips,x='total_bill',y='tip',hue='day',size="size",sizes=(10,200)) #把size映射到10-200区间，用于显示大小


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
#relplot折线图--简单
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
#relplot折线图——分类
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event") #hue分组
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event",style="region") #hue+style

plt.show()
'''


################################
#relplot折线图——分类
################################
'''
import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')

sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event") #hue分组
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="event",style="region") #hue+style

plt.show()
'''

################################
#relplot折线图——样式分组
################################

import matplotlib.pyplot as plt
import seaborn as sns
fmri=sns.load_dataset('fmri')
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="region",size="region",sizes=[0.5,2]) #size(size参数长度必须和hue的个数相同)
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="region",size="event",sizes=[0.5,2]) #size

plt.show()



sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="region",dashes=False) #dashes全部用实线
sns.relplot(data=fmri,x="timepoint",y="signal",kind="line",hue="subject") #dashes用不同的