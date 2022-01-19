#!/usr/bin/env python
# -*- coding:utf-8 -*-


#==============================#
#二维标量场
#==============================#

################################
#imshow
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表

X=np.random.randn(9,9)
ax.imshow(X)
#ax.imshow(X,interpolation='bicubic')

ax.axis('off')
plt.show()
'''

################################
#pcolor
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

x,y=np.mgrid[0:9,0:9]
c=np.random.randn(9,9)
ax.pcolor(x,y,c)
#ax.pcolor(c) #注意，因为mgrid的顺序问题，与不省略x,y绘制的图颜色顺序不同。

ax.axis('off')
plt.show()
'''

################################
#contour、contourf等值线图
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax1,ax2=fig.subplots(1,2) #创建图表
np.random.seed(202201)
#plt.style.use('ggplot')

x,y=np.mgrid[-1:1:0.1,-1:1:0.1]
z=x**2+y**2
#ax.contourf(x,y,z)
ax1.contour(x,y,z,levels=15,origin='upper')
ax2.contourf(x,y,z,levels=15,origin='upper')

plt.show()
'''

#==============================#
#二维矢量场
#==============================#

################################
#quiver
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)
plt.style.use('ggplot')

x,y=np.mgrid[-1:1:0.1,-1:1:0.1]
u=x+y
v=y-x
#ax.quiver(x,y,u,v)
ax.quiver(x,y,u,v,units='xy')

plt.show()
'''

################################
#streamplot
################################

'''
import numpy as np
import matplotlib.pyplot as plt
fig=plt.figure() #创建画布
ax=fig.subplots() #创建图表
np.random.seed(202201)

y,x=np.mgrid[-1:1:0.1,-1:1:0.1]  #streamplot要求x每行必须相同
u=x+y
v=y-x
#ax.streamplot(x,y,u,v)
ax.streamplot(x,y,u,v,density=0.5, color=np.random.randn(20,20))

plt.show()
'''

