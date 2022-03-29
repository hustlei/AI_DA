#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
test code for scipy linear algorithm and integral
author:lileilei
email:hustlei@sina.cn
"""

################################
#Distribution
################################
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x=np.arange(-10,10,0.1)
y1=stats.laplace.pdf(x)
plt.plot(x,y1,label="laplace")
y2=stats.laplace(scale=3).pdf(x)
plt.plot(x,y2,label="scale=3")
y3=stats.laplace(loc=3).pdf(x)
plt.plot(x,y3,label="loc=3")

plt.legend()
plt.show()
'''

#峰度偏度

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

x=np.arange(-10,10,0.1)
y1=stats.laplace.pdf(x)
plt.plot(x,y1,label="laplace")
y2=stats.laplace(scale=3).pdf(x)
plt.plot(x,y2,label="scale=3")

print(stats.laplace.stats(moments="mvsk"))
print(stats.laplace(scale=3).stats(moments="mvsk"))

pts1=stats.laplace(scale=0.1).rvs(size=100000)
sns.kdeplot(pts1)
pts2=stats.laplace(scale=1).rvs(size=100000)
sns.kdeplot(pts2,color='r')
pts3=np.random.random(size=100000)
sns.kdeplot(pts3,color='g')

print(stats.kurtosis(pts1))
print(stats.kurtosis(pts2))
print(stats.kurtosis(pts3))


plt.legend()
plt.show()




################################
#Distribution-multivar-normal
################################
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x=np.arange(-10,10,0.1)
y1=stats.multivariate_normal.pdf(x)
plt.plot(x,y1)

x0,y0=np.mgrid[-5:5:0.5,-5:5:0.5]
x=np.dstack([x0,y0])
y=stats.multivariate_normal.pdf(x,[0,0], [[2,1], [1,2]])
plt.figure().add_subplot(projection="3d").plot_wireframe(x0,y0,y)

plt.show()
'''


################################
#Distribution-multivar-t
################################
'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

x0,y0=np.mgrid[-5:5:0.5,-5:5:0.5]
x=np.dstack([x0,y0])
y=stats.multivariate_t.pdf(x,shape=np.eye(2,2))
plt.figure().add_subplot(projection="3d").plot_wireframe(x0,y0,y)

x=np.array([[0.05,0.01],[0.1,0.15],[0.5,0.2]])
print(stats.multivariate_t.pdf(x,shape=np.eye(2,2)))
'''

################################
#Distribution-dirichlet
################################

'''
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

distrib_d = stats.dirichlet(alpha=[2,3])  #二元迪利克雷分布
x0=np.arange(0,1,0.01)
y0=1-x0
x=np.array([x0,y0])  #两行，两行相加所有元素都是1

y=distrib_d.pdf(x)

plt.figure().add_subplot(projection="3d").plot(x0,y0,y)
plt.show()
'''


################################
#汇总统计
################################
'''
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

x = np.random.normal(0,1,1000)
x1 = np.random.normal(0,1000,1000)
sns.kdeplot(x)
sns.kdeplot(x1)
print(stats.kurtosis(x))
print(stats.kurtosis(x1))
plt.show()
'''

################################
#kde
################################
'''
import numpy as np
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

data=[0,1,3,2,2,2,3,4,5]
kernel=stats.gaussian_kde(data)
x=np.arange(0,5,0.2)
y=kernel(x)
plt.plot(x,y)
plt.show()
'''
