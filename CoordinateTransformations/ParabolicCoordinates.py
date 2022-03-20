# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 14:26:14 2022

@author: Al Bernstein
"""

import numpy as np
from matplotlib import pyplot as plt
import draw as dw
_dw = dw.draw()
import itertools


def x(_coords):
    rho, tau = _coords
    return rho*tau

def y(_coords):
    rho, tau = _coords
    return (1/2)*(tau**2 - rho**2)

fig = plt.figure()
ax = fig.add_subplot(111)# initial vector
#plot constant rho and variable tau

tau =  np.linspace(-5,5,10)
rho = np.linspace(-5,5,10)

coords = []
for i in itertools.product(rho, tau):
    coords.append(i)

for i in coords:
        _dw.drawXYPoint([x(i),y(i)],ax, 'black', 40)
        ax.plot(x((i[0], tau)), y((i[0],tau) ) )  
        ax.plot(x((rho, i[1])), y((rho,i[1]) ) )  
            


i = coords[80]
dx = 0.5
dy = 0.5

A = np.array([[i[1],-i[0]],[i[0], i[1]]])
 
# draw rho vector
_dw.drawVecXY([x(i), y(i)], [x(i)+dx*A[0][0], y(i)+dy*A[0][1]], ax,'green')
plt.text(x(i)+dx*(A[0][0]),y(i)+dy*A[0][1]/2,r'$\rho$',size=15)
# draw tau vector
_dw.drawVecXY([x(i), y(i)], [x(i)+dx*A[1][0], y(i)+dy*A[1][1]], ax,'red')
plt.text(x(i)+dx*(A[1][0]),y(i)+dy*A[1][1]/2,r'$\tau$',size=15)

plt.title('Parabolic Coordinate Patch', fontsize = 40)


