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

print('Compute Using Matrix Operations\n')

T_ij = t4.computeWeightMatrix(TW, 'E', 'E', 2)

# print weight matrices
t4.printWeightMatricesToLatex(T_ij, 2)

elem_list = []
subscript_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']

for i in range(2):
    for j in range(2):
        e_1 = t4.getBasisVector('E', i)
        t4.printVector(e_1[0], True, e_1[1], 2)
        e_2 = t4.getBasisVector('E', j)
        t4.printVector(e_2[0], False, e_2[1], 2)
        Eixj = np.einsum('i,j', e_1[0], e_2[0])
        label = e_1[1] + '⊗' + e_2[1]
        t4.printMatrix(Eixj, label, 2)
        index = i*2 + j
        elem_ij = t4.computeTensorElement(Eixj, T_ij[index], 2)
        l_elem_ij = t4.convertElementToLatex(elem_ij, 2)
        print('elem' + subscript_num[i+1] + subscript_num[j+1] + ' = ', l_elem_ij, '\n')
        elem_list.append(elem_ij)
        
# sum up elements 

Tsum = 0

for elem in elem_list:
    Tsum += elem
    
l_Tsum = t4.convertElementToLatex(Tsum, 2)
print('Fourth Order Tensor using Matrices = ', l_Tsum + '\n')






