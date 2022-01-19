import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax1=fig.add_subplot(221, projection='aitoff')
ax2=fig.add_subplot(222, projection='hammer')
ax3=fig.add_subplot(223, projection='mollweide')
ax4=fig.add_subplot(224, projection='lambert')
#aitoff, hammer:椭圆地图投影，纬线也是弧形
#mollweide:椭圆地图投影，纬线也是水平直线
#lambert:圆形地图投影

x=np.linspace(-np.pi,np.pi,200)  #经度-π~π(-180~180)
y=np.sin(x)                         #维度-π/2~π/2(-90~90)

ax1.plot(x,y)
ax2.plot(x,y)
ax3.plot(x,y)
ax4.plot(x,y)

ax1.grid(True)
ax2.grid(True)
ax3.grid(True)
ax4.grid(True)

fig.tight_layout()
plt.show()