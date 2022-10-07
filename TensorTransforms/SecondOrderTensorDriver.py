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

secondOrder = t.secondOrderTensor()
E = vari['E']
result = secondOrder.computeTensorOuterProduct(T, E, E, 2)

print('Second Order Tensor Results\n')

latex = t.convertToLatex()

l_result = latex.convertMatrixToLatex(T, 2)
print('T\n')
print(l_result)
print('\n')

l_result = latex.convertMatrixToLatex(result, 2)
print('Tensor computed by outer product\n')
print(l_result)
print('\n')

_vars.printVars(['E','ET'],2)

# compute tensor using transpose(E).T.E

result1 = secondOrder.computeTensorInnerProduct(T, E, E)
l_result = latex.convertMatrixToLatex(result1, 2)
print('Tensor computed by ET(T)E\n')
print(l_result)
print('\n')


# transform 2nd order contravariant tensor
# polar based transform

# compute second order tensor in the polarSqrt system
# using T1 = BT(T)B

print('Transform to polar sqrt coordinate system')

_vars.printVars(['A','BT', 'B', 'E1', 'E1T'], 2)

B = vari['B']
tmp = np.dot(np.transpose(B),T)
T1 = np.dot(tmp, B)

l_result = latex.convertMatrixToLatex(T1, 2)
print('T1 = B(T)T \n')
print(l_result)
print('\n')


E1 = vari['E1']
result3 = secondOrder.computeTensorOuterProduct(T1, E1, E1, 2)

l_result = latex.convertMatrixToLatex(result3, 2)
print('Tensor computed by outer product\n')
print(l_result)
print('\n')