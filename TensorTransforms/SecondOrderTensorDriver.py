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

secondOrder = t.secondOrderTensor(r, theta)
posLst = [t.pos.up, t.pos.up]
verbose = True

result = secondOrder.computeTensorOuterProduct(T, posLst, True, 2, verbose)

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

result1 = secondOrder.computeTensorInnerProduct(T, posLst, True, 2, verbose)

l_result = latex.convertMatrixToLatex(result1, 2)
print('Full Tensor computed by the inner product \n')
print(l_result)
print('\n')

# compute difference between result1 and result
l_result = latex.convertMatrixToLatex(result1 - result, 2)
print('Difference')
print(l_result)
print('\n')


# transform 2nd order contravariant tensor
# polar based transform

# compute second order tensor in the polarSqrt system

print('Transform to polar sqrt coordinate system')

T1 = secondOrder.transformTensor(T, posLst, 2)

l_result = latex.convertMatrixToLatex(T1, 2)
print('T̅ = \n', l_result)

# compute tensor in primed coordinates using the outer product

result2 = secondOrder.computeTensorOuterProduct(T1, posLst, False, 2, verbose)

l_result = latex.convertMatrixToLatex(result2, 2)
print('Full Tensor in primed coordinates computed by outer products\n')
print(l_result)
print('\n')

# compute difference between result2 and result
l_result = latex.convertMatrixToLatex(result2 - result, 2)
print('Difference')
print(l_result)
print('\n')

# compute tensor in primed coordinates using the inner product

result3 = secondOrder.computeTensorInnerProduct(T1, posLst, False, 2, verbose)
l_result = latex.convertMatrixToLatex(result3, 2)
print('Full Tensor in primed coordinates computed by the inner product\n')
print(l_result)
print('\n')

# compute difference between result3 and result
l_result = latex.convertMatrixToLatex(result3 - result, 2)
print('Difference')
print(l_result)
print('\n')

