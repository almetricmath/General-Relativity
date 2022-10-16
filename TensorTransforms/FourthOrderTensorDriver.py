# -*- coding: utf-8 -*-
"""
Created on Fri Oct 14 10:25:18 2022

@author: Al
"""

import TensorTransforms as t
import numpy as np
import sys

# compute 3rd order tensor using the outer product

latex = t.convertToLatex()

coords = t.coordinateTransforms()

r = 2
theta = np.pi/3
t4 = t.fourthOrderTensor(r, theta)
# define tensir components

TW = [[ np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])], [np.array([[9,10],[11,12]]), np.array([[13,14],[15,16]])]]
TW = np.array(TW)

result = t4.computeTensorOuterProduct(TW, 'E', 'E', 'E', 'E', 2)
l_result = t4.convertElementToLatex(result, 2)
print('Fourth Order Tensor by Outer Product')
print(l_result)
print('\n')

T_ij = t4.computeWeightMatrix(TW, 'E', 'E', 2)

# print weight matrices
t4.printWeightMatricesToLatex(T_ij, 2)





