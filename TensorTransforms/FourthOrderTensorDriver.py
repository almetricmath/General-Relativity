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

verbose = True

print('\nOuter Product Calculation\n')
result_1 = t4.computeTensorOuterProduct(TW, posLst, True, 2, verbose)
l_result = t4.convertElementToLatex(result_1, 2)

print('result = ', l_result)
print('\n')

# compute 4th order tensor using the inner product

print('Inner Product Calculation\n')
verbose = True
result_2 = t4.computeTensorInnerProduct(TW, posLst, True, 2, verbose)
l_result = t4.convertElementToLatex(result_2, 2)
print('result = ', l_result)
print('\n')

l_diff = t4.convertElementToLatex(result_2 - result_1, 2)
print('Difference between inner product and outer product = ', l_diff, '\n')

# tensor under coordinate change

# transform tensor components

print('Transform Tensor Components\n')

T_prime_block, T_ijkl = t4.transformTensor(TW, posLst, 2, verbose)

print('Block Tensor Components in the Primed Coordinate System\n')
l_Tblock = t4.convertElementToLatex(T_prime_block, 2)
print('Transformed T_block = ', l_Tblock, '\n')
print('T_ijkl in Primed Coordinate System\n')
l_T_ijkl = t4.convertElementToLatex(T_ijkl, 2)
print('Transformed T_ijkl = ', l_T_ijkl, '\n')

# compute tensor in the primed coordinates using the inner product

print('Compute Tensor in the Primed Coordinate System Using an Inner Product\n') 

result_3 = t4.computeTensorInnerProduct(T_prime_block, posLst, False, 2, verbose)
l_result = t4.convertElementToLatex(result_3, 2)
print(l_result)
print('\n')

l_diff = t4.convertElementToLatex(result_3 - result_1, 2)
print('Difference between the inner product in primed coordinates and outer product in unprimed coordinates = ', l_diff, '\n')

# compute tensor in the primed coordinates using the outer product

result_4 = t4.computeTensorOuterProduct(T_ijkl, posLst, False, 2, verbose)
l_result = t4.convertElementToLatex(result_4, 2)
print('result = ', l_result)
print('\n')

l_diff = t4.convertElementToLatex(result_4 - result_1, 2)
print('Difference between the outer product in primed coordinates and outer product in unprimed coordinates = ', l_diff, '\n')







