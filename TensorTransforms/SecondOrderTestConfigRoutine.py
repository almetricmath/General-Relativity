# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 08:41:00 2022

@author: Al
"""

import numpy as np
import TensorTransforms as t
import sys

# Run the different configurations to see that the results are the same

r = 2
theta = np.pi/3

# 2nd order tensor
# contravariant tensor

# create tensor 
T = np.array([[1,2],[3,4]])


G = np.array([[1, 0],[0, r**2]])
Ginv = np.linalg.inv(G)

# [up, up] -> [down, up]

T_down_i_up_j = np.dot(G, T)
verbose = False

print('T_down_i_up_j = ', T_down_i_up_j, '\n')

# initial configuration 
inPosLst = [t.pos.up, t.pos.up]

posLst = [t.pos.down, t.pos.up]

# Test config routine

utils = t.utils()

T_test = utils.changeConfig(T, inPosLst, posLst, G, Ginv)
print('config output = ', T_test, '\n')

# [up, up] -> [up, down]

T_up_i_down_j = np.dot(T, G)
print('T_up_i_down_j = ', T_up_i_down_j, '\n')

posLst = [t.pos.up, t.pos.down]

T_test = utils.changeConfig(T, inPosLst, posLst, G, Ginv)
print('config output = ', T_test, '\n')

# [up, up] -> [down, down]

tmp = np.dot(T, G)
T_down_i_down_j = np.dot(G, tmp)
print('T_down_i_down_j = ', T_down_i_down_j, '\n')

posLst = [t.pos.down, t.pos.down]

T_test = utils.changeConfig(T, inPosLst, posLst, G, Ginv)
print('config output = ', T_test, '\n')

# Test inverse transforms

# [down, down] -> [up, down]

# initial configuration 
inPosLst = [t.pos.down, t.pos.down]

posLst = [t.pos.up, t.pos.down]

T_test = utils.changeConfig(T_down_i_down_j, inPosLst, posLst, G, Ginv)
print('T_up_i_down_j = ', T_up_i_down_j, '\n')
print('config output = ', T_test, '\n')

# [down, down] -> [down, up]

posLst = [t.pos.down, t.pos.up]

T_test = utils.changeConfig(T_down_i_down_j, inPosLst, posLst, G, Ginv)
print('T_down_i_up_j = ', T_down_i_up_j, '\n')
print('config output = ', T_test, '\n')

# [down, down] -> [up, up]

posLst = [t.pos.up, t.pos.up]

T_test = utils.changeConfig(T_down_i_down_j, inPosLst, posLst, G, Ginv)
print('T_down_i_up_j = ', T_down_i_up_j, '\n')
print('config output = ', T_test, '\n')









 




