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

T = np.array([[[1.0, 2.0],[3.0, 4.0]],[[5.0, 6.0],[7.0, 8.0]]])

# specify tensor index positions

posLst = [t.pos.up, t.pos.down, t.pos.up]
verbose = False

result = t3.computeTensorOuterProduct(T, posLst, True, 2, verbose)
print('3rd order tensor by outer product\n')
l_result = t3.convertToLatex(result, 2)
print(l_result)
print('\n')

# compute 3rd order tensor using the inner product

print('3rd order tensor by EᵀTE\n')
result2 = t3.computeTensorInnerProduct(T, posLst, True, 2, verbose)

l_result = t3.convertToLatex(result2, 2)
print(l_result)
print('\n')

diff_result = t3.convertToLatex(result2 - result, 2)
print('Difference between the inner product and outer product tensor computations in umprimed coordinates\n') 
print(diff_result, '\n\n')


# transform 3rd order contravariant tensor
# polar based transform

print('Transform Tensor to polar sqrt system\n')

# use computing with the weight matrix instead - 11/27/2022

print(' Inner Product - BᵀTB')

T1 = t3.transformTensor(T, posLst, True, 2)

print(' Compute transformed tensor using transpose(F1).[T1_n].H1 ', '\n')

result3 = t3.computeTensorInnerProduct(T1, posLst, False, 2, verbose)
l_result = t3.convertToLatex(result3, 2)
print(l_result, '\n')

diff_result = t3.convertToLatex(result3 - result, 2)
print('Difference between the inner product in the primed coordinate system and the outer product in umprimed coordinates\n') 
print(diff_result, '\n\n')
