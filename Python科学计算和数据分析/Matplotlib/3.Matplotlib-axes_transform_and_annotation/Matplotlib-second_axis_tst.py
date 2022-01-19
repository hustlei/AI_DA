import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax=fig.subplots()

x=np.linspace(-np.pi,np.pi,200) 
y=np.sin(x) 
ax.plot(x,y)
ax.set_xlabel('x')

xaxis2=ax.secondary_xaxis('top', functions=(np.rad2deg,np.deg2rad)) #设置x轴第二坐标
xaxis2.set_xlabel('angle[rad]')

plt.show()