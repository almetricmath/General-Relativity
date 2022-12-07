# -*- coding: utf-8 -*-
"""
Created on Tue Dec  6 19:55:01 2022

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
verbose = False
posLst = [t.pos.up, t.pos.up]

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


# Run the different configurations so that the results are the same

G = np.array([[1, 0],[0, r**2]])
T_lower_i_upper_j = np.dot(G, T)
verbose = False

latex = t.convertToLatex()
l_result = latex.convertMatrixToLatex(T_lower_i_upper_j , 2)
print('T_lower_i_upper_j \n')
print(l_result)
print('\n')

posLst = [t.pos.down, t.pos.up]

result1 = secondOrder.computeTensorOuterProduct(T_lower_i_upper_j, posLst, True, 2, False)
l_result = latex.convertMatrixToLatex(result1, 2)
print('Full Tensor, T_lower_i_upper_j, computed by outer products\n')
print(l_result)
print('\n')

# compute difference between resul4 and result
l_result = latex.convertMatrixToLatex(result1 - result, 2)
print('Difference')
print(l_result)
print('\n')


T_upper_i_lower_j = np.dot(T, G)

l_result = latex.convertMatrixToLatex(T_upper_i_lower_j , 2)
print('T_upper_i_lower_j \n')
print(l_result)
print('\n')

posLst = [t.pos.up, t.pos.down]

result2 = secondOrder.computeTensorOuterProduct(T_upper_i_lower_j, posLst, True, 2, False)
l_result = latex.convertMatrixToLatex(result2, 2)
print(('Full Tensor, T_upper_i_lower_j, computed by outer products\n'))
print(l_result)
print('\n')

# compute difference between result5 and result
l_result = latex.convertMatrixToLatex(result2 - result, 2)
print('Difference')
print(l_result)
print('\n')

tmp = np.dot(T, G)
T_lower_i_lower_j = np.dot(G, tmp)

l_result = latex.convertMatrixToLatex(T_lower_i_lower_j , 2)
print('T_lower_i_lower_j \n')
print(l_result)
print('\n')

posLst = [t.pos.down, t.pos.down]

result3 = secondOrder.computeTensorOuterProduct(T_lower_i_lower_j, posLst, True, 2, False)
l_result = latex.convertMatrixToLatex(result3, 2)
print(('Full Tensor, T_lower_i_lower_j, computed by outer products\n'))
print(l_result)
print('\n')

# compute difference between resul6 and result
l_result = latex.convertMatrixToLatex(result3 - result, 2)
print('Difference')
print(l_result)
print('\n')




