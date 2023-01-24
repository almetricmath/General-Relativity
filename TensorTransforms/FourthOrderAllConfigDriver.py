# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 16:18:42 2023

@author: Al
"""

import TensorTransforms as t
import numpy as np

r = 2
theta = np.pi/3

t4 = t.fourthOrderTensor(r, theta)

latex = t.convertToLatex()
coords = t.coordinateTransforms()

# define tensor components

T = [[ np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])], [np.array([[9,10],[11,12]]), np.array([[13,14],[15,16]])]]
T = np.array(T)

inPosLst = [t.pos.up, t.pos.down, t.pos.down, t.pos.up]

verbose = False
unprimed = True

result = t4.computeTensorOuterProduct(T, inPosLst, unprimed, 2)
print('Original Configuration = ', inPosLst, '\n')
print('4th order tensor by outer product\n')
l_result = t4.convertElementToLatex(result, 2)
print(l_result, '\n')


# declare the metric and inverse metric
 
G = np.array([[1, 0],[0, r**2]])
Ginv = np.linalg.inv(G)

# Set up the initial configuration to be [t.pos.up, t.pos.up, t.pos.up, t.pos.up]

init_posLst = [t.pos.up, t.pos.up, t.pos.up, t.pos.up]
T_init = t4.changeConfig(T, inPosLst, init_posLst, G, Ginv, 2, verbose)

print('Initial Configuration = ', init_posLst, '\n')
l_T_init =  t4.convertElementToLatex(T_init, 2)
print('T = ', l_T_init, '\n')

# Test outer product formulation

result_1 = t4.computeTensorOuterProduct(T_init, init_posLst, unprimed, 2)

l_diff = t4.convertElementToLatex(result_1 - result, 2)
print('Difference with the Original configuration = ', l_diff, '\n')

posLst = [[t.pos.up, t.pos.up, t.pos.up, t.pos.down], [t.pos.up, t.pos.up, t.pos.down, t.pos.up],\
          [t.pos.up, t.pos.up, t.pos.down, t.pos.down], [t.pos.up, t.pos.down, t.pos.up, t.pos.up],\
          [t.pos.up, t.pos.down, t.pos.up, t.pos.down], [t.pos.up, t.pos.down, t.pos.down, t.pos.up],\
          [t.pos.up, t.pos.down, t.pos.down, t.pos.down],[t.pos.down, t.pos.up, t.pos.up, t.pos.up],\
          [t.pos.down, t.pos.up, t.pos.up, t.pos.down], [t.pos.down, t.pos.up, t.pos.down, t.pos.up],\
          [t.pos.down, t.pos.up, t.pos.down, t.pos.down], [t.pos.down, t.pos.down, t.pos.up, t.pos.up],\
          [t.pos.down, t.pos.down, t.pos.up, t.pos.down], [t.pos.down, t.pos.down, t.pos.down, t.pos.up],\
          [t.pos.down, t.pos.down, t.pos.down, t.pos.down]]
          
for item in posLst:
    T_config = 0
    T_config = t4.changeConfig(T_init, init_posLst, item, G, Ginv, 2, verbose)
    print('configuration = ', item, '\n')
    print('T = ', T_config, '\n')
    result_2 = t4.computeTensorOuterProduct(T_config, item, unprimed, 2)
    print('4th order tensor by outer product\n')
    l_result = t4.convertElementToLatex(result_2, 2)
    print(l_result)
    print('\n')
    l_diff = t4.convertElementToLatex(result_2 - result, 2)
    print('Difference with the Original configuration = ', l_diff, '\n')

    
    

                        
          
 







