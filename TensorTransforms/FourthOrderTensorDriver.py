# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:25:18 2022

@author: Al
"""

import TensorTransforms as t
print(t.__file__)
import numpy as np
import sys


r = 2
theta = np.pi/3

t4 = t.fourthOrderTensor(r, theta)

latex = t.convertToLatex()
coords = t.coordinateTransforms()

# define tensor components

TW = [[ np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])], [np.array([[9,10],[11,12]]), np.array([[13,14],[15,16]])]]
TW = np.array(TW)

# specify tensor index positions

posLst = [t.pos.up, t.pos.down, t.pos.down, t.pos.up]

# compute 4th order tensor using the outer product

result_1 = t4.computeTensorOuterProduct(TW, posLst, True, 2)
l_result = t4.convertElementToLatex(result_1, 2)
print('Fourth Order Tensor by Outer Product')
print(l_result)
print('\n')

# compute 4th order tensor using the inner product

result_2 = t4.computeTensorInnerProduct(TW, posLst, True, 2)
l_result = t4.convertElementToLatex(result_2, 2)
print('Fourth Order Tensor by Inner Product')
print(l_result)
print('\n')

l_diff = t4.convertElementToLatex(result_2 - result_1, 2)
print('Difference between inner product and outer product = ', l_diff, '\n')

sys.exit(0)

# tensor under coordinate change

# transform tensor components

T_test = t4.transformTensor(TW, posLst, False, 2)

# compute tensor

result3 = t4.computeTensorInnerProduct(T_test, posLst, False, 2)
l_result = t4.convertElementToLatex(result3, 2)
print(l_result)
print('\n')    
