from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import draw as dw
import coordUtils as cu
import GreatCircle as gc

# This file plots a section of a sphere and shows an exaggerated 
# parallel transport for illustration

fig1 = plt.figure()
ax1 = fig1.add_subplot(111, projection='3d')
ax1 = fig1.gca(projection='3d')
ax1.set_xlabel('\nX', fontsize = 15)
ax1.set_ylabel('\nY', fontsize = 15)
ax1.set_zlabel('Z', fontsize = 15)
txt = 'Figure 3'
fig1.text(.5, 0.01, txt, ha='center', fontsize = 40)
plt.title('Parallel Transport Exaggerated on Sphere', fontsize = 40)


_dw = dw.draw()
_cu = cu.coordUtils()
_gc = gc.greatCircle()

_dw.plotSphere(1, 0, np.pi/3 + 0.1, -0.1, np.pi/2, 500, "green", ax1)

_u0_sp = np.array([1, np.pi/3, 0])
du = 0.3
_u1_sp = _u0_sp + np.array([0, -du, 0])
step = np.array([0, 0, 0.5])
_u0_next_sp = _u0_sp + step

_dw.drawXYZPoint(_u0_sp, ax1, 'red')
_dw.drawXYZPoint(_u0_next_sp, ax1, 'red')
_dw.drawXYZPoint(_u1_sp, ax1, 'red')
arc = _gc.arc(_u1_sp, _u0_next_sp, 1, 7)
# draw xyz great circle
_dw.drawXYZArc(arc[0], ax1, 3, 'red')
midpoint = arc[1]
_dw.drawXYZPoint(midpoint, ax1, 'blue')
t_arc = _gc.twice_arc(_u0_sp, midpoint,1, 14)
_u1_next_sp = t_arc[-1]
_dw.drawXYZArc(t_arc, ax1, 3, 'red')
_dw.drawXYZPoint(t_arc[-1], ax1, 'red')
_dw.drawVecXYZ(_u0_sp, _u1_sp, ax1)
_dw.drawVecXYZ(_u0_next_sp, _u1_next_sp, ax1)

