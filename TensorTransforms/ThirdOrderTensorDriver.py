# -*- coding: utf-8 -*-
"""
Created on Fri Oct  7 15:50:32 2022

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
t3 = t.thirdOrderTensor(r, theta)

T = np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])

result = t3.computeTensorOuterProduct(T, 'E', 'E', 'E', 2)
print('3rd order tensor by outer product\n')
l_result = t3.convertToLatex(result, 2)
print(l_result)
print('\n')


# compute 3rd order tensor using the inner product

print('3rd order tensor by EᵀTE\n')
result2 = t3.computeTensorInnerProduct(T, 'E', 'E', 'E', 2)

l_result = t3.convertToLatex(result2, 2)
print(l_result)
print('\n')

# transform 3rd order contravariant tensor
# polar based transform

print('Transform Tensor to polar sqrt system\n')

# compute transformed tensor using the outer product

print(' Inner Product - BᵀTB')
T1 = t3.computeTensorInnerProduct(T, 'B', 'B', 'B', 2)

# compute transformed tensor using transpose(E1).T1_E1_1.E1

print('Inner Product\n of result')
result4 = t3.computeTensorInnerProduct(T1, 'E', 'E1', 'E1', 2)
print('3rd order Tensor computed by E̅ᵀT̅ E̅ = \n')
l_result = t3.convertToLatex(result2, 2)
print(l_result, '\n')

