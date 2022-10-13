# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 11:37:33 2022

@author: Al
"""
import numpy as np
import TensorTransforms as t
import sys



# compute 2nd order tensor in polar coordinates 
# using both outer product and inner product

r = 2
theta = np.pi/3

# 2nd order tensor
# contravariant tensor

# create tensor 
T = np.array([[1,2],[3,4]])
_vars = t.variables(r, theta)
vari = _vars.getVars()

secondOrder = t.secondOrderTensor(r, theta)

result = secondOrder.computeTensorOuterProduct(T, 'E', 'E', 2)

print('Second Order Tensor Components\n')

latex = t.convertToLatex()

l_result = latex.convertMatrixToLatex(T, 2)
print('T\n')
print(l_result)
print('\n')

l_result = latex.convertMatrixToLatex(result, 2)
print('Full Tensor computed by outer products\n')
print(l_result)
print('\n')


# compute tensor using transpose(E).T.E

result1 = secondOrder.computeTensorInnerProduct(T, 'E', 'E', 2)
l_result = latex.convertMatrixToLatex(result1, 2)
print('Full Tensor computed by EᵀTE\n')
print(l_result)
print('\n')


# transform 2nd order contravariant tensor
# polar based transform

# compute second order tensor in the polarSqrt system
# using T1 = BT(T)B

print('Transform to polar sqrt coordinate system')

T1 = secondOrder.computeTensorInnerProduct(T, 'B', 'B', 2)

l_result = latex.convertMatrixToLatex(T1, 2)
print('T̅ = BᵀTB \n')
print(l_result)
print('\n')


result3 = secondOrder.computeTensorOuterProduct(T1, 'E1', 'E1', 2)
l_result = latex.convertMatrixToLatex(result3, 2)
print('Full Tensor computed by outer products\n')
print(l_result)
print('\n')

result4 = secondOrder.computeTensorInnerProduct(T1, 'E1', 'E1', 2)
l_result = latex.convertMatrixToLatex(result3, 2)
print('Full Tensor computed by inner products\n')
print(l_result)
print('\n')
