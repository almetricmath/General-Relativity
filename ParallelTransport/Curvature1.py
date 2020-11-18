from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import draw as dw
import coordUtils as cu


_dw = dw.draw()
_cu = cu.coordUtils()

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1 = fig1.gca(projection='3d')
ax1.spines['bottom'].set_visible(False)
ax1.set_xticklabels([])
ax1.set_yticklabels([])        
ax1.set_zticklabels([])
ax1.grid(False)
plt.axis("off")

_dw.plotSphere(1, np.pi/6, np.pi/2,0, 2*np.pi, 100, 'green', ax1)
_dw.plotSphere(1, 0, np.pi/6,0, 2*np.pi, 100, 'blue', ax1)
_dw.plotLatitudeCircle(1, np.pi/6, 30, 'black', ax1)
_u0_sp = np.array([1, np.pi/6, 0])
du = 0.05
_u1_sp = _u0_sp + np.array([0, -du, 0])
_u0_xyz = _cu.SphericalToCartesian(_u0_sp)
_u1_xyz = _cu.SphericalToCartesian(_u1_sp)
_dw.drawXYZPoint(_u0_sp, ax1, 'red')
_dw.drawXYZPoint(_u1_sp, ax1, 'red')
_dw.drawVecXYZ(_u0_sp, _u1_sp, ax1)
_u1_sp = _u0_sp + np.array([0, -0.04, -0.03 ])
_dw.drawXYZPoint(_u1_sp, ax1, 'red')
_dw.drawVecXYZ(_u0_sp, _u1_sp, ax1)

