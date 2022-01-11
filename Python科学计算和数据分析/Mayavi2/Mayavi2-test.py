from mayavi import mlab
import numpy as np

# 最简单的绘图

'''
d=np.random.random((50,50))
mlab.imshow(d)
mlab.show()
'''

# Create the data.
# from numpy import pi, sin, cos, mgrid
# dphi, dtheta = pi/250.0, pi/250.0
# [phi,theta] = mgrid[0:pi+dphi*1.5:dphi,0:2*pi+dtheta*1.5:dtheta]
# m0 = 4; m1 = 3; m2 = 2; m3 = 3; m4 = 6; m5 = 2; m6 = 6; m7 = 4;
# r = sin(m0*phi)**m1 + cos(m2*phi)**m3 + sin(m4*theta)**m5 + cos(m6*theta)**m7
# x = r*sin(phi)*cos(theta)
# y = r*cos(phi)
# z = r*sin(phi)*sin(theta)

#View it.
# from mayavi import mlab
# s = mlab.mesh(x, y, z)
#mlab.show()



import numpy as np
a = np.random.random((4, 4))
from mayavi.api import Engine
e = Engine()
e.start()
s = e.new_scene()
from mayavi.sources.api import ArraySource
src = ArraySource(scalar_data=a)
e.add_source(src)
from mayavi.filters.api import WarpScalar, PolyDataNormals
warp = WarpScalar()
e.add_filter(warp, obj=src)
normals = PolyDataNormals()
e.add_filter(normals, obj=warp)
from mayavi.modules.api import Surface
surf = Surface()
e.add_module(surf, obj=normals)
mlab.show()