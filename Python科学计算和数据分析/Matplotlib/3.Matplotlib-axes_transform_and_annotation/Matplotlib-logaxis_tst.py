import numpy as np
import matplotlib.pyplot as plt

fig=plt.figure()
ax1=fig.subplots()

###############################
#logx
###############################
x=np.array([1,10,100,1000,10000])
y=(np.log10(x)-2)**2

ax1.semilogx(x,y)


###############################
#logy
###############################
# y=np.array([1,10,100,1000,10000])
# x=(np.log10(x)-2)**2

# ax1.semilogy(x,y)

###############################
#loglog
###############################
# y=np.array([1,10,100,1000,10000])
# x=y

# ax1.loglog(x,y)

#################################

ax1.grid(True)
fig.tight_layout()
plt.show()