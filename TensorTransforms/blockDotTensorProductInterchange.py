# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:16:04 2023

@author: Al
"""

import TensorTransforms as t
import numpy as np
import sys


def convertElementToLatex(_elem, _class, _n):
    
    ret = '\\bmatrix{'     
    
    for i in range(_n):
        for j in range(_n):
            tmp = _class.convertMatrixToLatex(_elem[i][j], _n)
            ret += tmp
            if j != _n -1:
                ret += '&'
        if i != _n - 1:            
            ret += '\\\\' 
    
    ret += '\\\\}' 
    return ret



# start with fourth order tensor components

T = [[ np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])], [np.array([[9,10],[11,12]]), np.array([[13,14],[15,16]])]]
T = np.array(T)

E = np.array([[0.500000, 0.866025], [-1.732051, 1.000000]])
W = np.array([[0.500000, 0.866025], [-0.433013, 0.250000]])

_n = 2

# first perform tensor product then dot - W.[E x T]

# tensor product E x T

tensorProduct = t.utils().tensorProduct([E], T, _n)

l_tensorProduct = convertElementToLatex(tensorProduct, t.convertToLatex(), _n)
print('[[E] x T] = ', l_tensorProduct,'\n')

result = t.utils().blockDotProduct(W, tensorProduct, _n)
l_result = convertElementToLatex(result, t.convertToLatex(), _n)
print('W.[[E] x T] = ', l_result, '\n')

# second perform dot then tensor product - [E] x [W.T]

dotProduct = t.utils().blockDotProduct(W, T, _n)
l_dotProduct = convertElementToLatex(dotProduct, t.convertToLatex(), _n)
print('[W.T] = ',l_dotProduct,'\n')

result_1 = t.utils().tensorProduct([E], dotProduct, _n)
l_result = convertElementToLatex(result_1, t.convertToLatex(), _n)

print('[E] x [W.T] = ', l_result, '\n')

l_diff = convertElementToLatex(result_1 - result, t.convertToLatex(), _n)
print('difference = ', l_diff, '\n')

