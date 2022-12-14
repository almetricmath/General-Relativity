# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:12:27 2022

@author: Al
"""

import TensorTransforms as t
import numpy as np
import sys

# compute 3rd order tensor using the outer product

latex = t.convertToLatex()

coords = t.coordinateTransforms()

# initial configuration T(up, down, up)

r = 2
theta = np.pi/3
t3 = t.thirdOrderTensor(r, theta)

T = np.array([[[1.0, 2.0],[3.0, 4.0]],[[5.0, 6.0],[7.0, 8.0]]])

# specify tensor index positions

inPosLst = [t.pos.up, t.pos.down, t.pos.up]
verbose = False

result = t3.computeTensorOuterProduct(T, inPosLst, True, 2, verbose)
print('3rd order tensor by outer product\n')
l_result = t3.convertToLatex(result, 2)
print(l_result)
print('\n')

# compute 3rd order tensor using the inner product

print('3rd order tensor by EᵀTE\n')
result1 = t3.computeTensorInnerProduct(T, inPosLst, True, 2, verbose)

l_result = t3.convertToLatex(result1, 2)
print(l_result)
print('\n')

diff_result = t3.convertToLatex(result1 - result, 2)
print('Difference between the inner product and outer product tensor computations in umprimed coordinates\n') 
print(diff_result, '\n\n')


# declare the metric and inverse metric
 
G = np.array([[1, 0],[0, r**2]])
Ginv = np.linalg.inv(G)

posLst = [t.pos.up, t.pos.up, t.pos.up]

T_init = t3.changeConfig(T, inPosLst, posLst, G, Ginv, 2, verbose)

result2 = t3.computeTensorOuterProduct(T_init, posLst, True, 2, verbose)
print('3rd order tensor by outer product\n')
l_result = t3.convertToLatex(result2, 2)
print(l_result)
print('\n')

diff_result = t3.convertToLatex(result2 - result, 2)
print('Difference between the inner product and outer product tensor computations in umprimed coordinates\n') 
print(diff_result, '\n\n')

posLst = [ [t.pos.up, t.pos.up, t.pos.down], [t.pos.up, t.pos.down, t.pos.up],  [t.pos.up, t.pos.down, t.pos.down], [t.pos.down, t.pos.up, t.pos.up], [t.pos.down, t.pos.up, t.pos.down], [t.pos.down, t.pos.down, t.pos.up], [t.pos.down, t.pos.down, t.pos.down]] 

inPosLst = [t.pos.up, t.pos .up, t.pos.up]


for item in posLst:
    T_config = 0
    T_config = t3.changeConfig(T_init, inPosLst, item, G, Ginv, 2, verbose)
    result3 = t3.computeTensorOuterProduct(T_config, item, True, 2, verbose)
    print('3rd order tensor by outer product\n')
    l_result = t3.convertToLatex(result3, 2)
    print(l_result)
    print('\n')
    
    diff_result = t3.convertToLatex(result3 - result, 2)
    print('Difference between the inner product and outer product tensor computations in umprimed coordinates\n') 
    print(diff_result, '\n\n')


