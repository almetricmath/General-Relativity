# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:25:18 2022

@author: Al
"""

import TensorTransforms as t
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


'''
# tensor under coordinate change

# transform tensor components

# compute tensor

T_test = t4.transformTensor(TW, posLst, False, 2)
result1 = t4.computeTensorInnerProduct(T_test, posLst, True, 2)
l_result = t4.convertElementToLatex(result1, 2)
print(l_result)
print('\n')
'''

'''
# Test transpose(B).T_ij.A pattern

B = np.array([[1,2],[3,4]])
T_ij = np.array([[5,6],[7,8]])
A = np.array([[9,10],[11,12]])
'''
    
