# -*- coding: utf-8 -*-
"""
Created on Sun Jan 22 18:16:04 2023

@author: Al
"""

import TensorTransforms as t
import numpy as np


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

tensorProduct  = np.array([[[[0.0]*_n]*_n]*_n]*_n)

# first perform tensor product then dot - W.[E x T]

# tensor product E x T

for k in [E]:
    for i in range(_n):
        for j in range(_n):
            tensorProduct[i][j] = np.dot(k,T[i][j])

l_tensorProduct = convertElementToLatex(tensorProduct, t.convertToLatex(), _n)
print('[[E] x T] = ', l_tensorProduct,'\n')

result = np.array([[[[0.0]*_n]*_n]*_n]*_n)

# dot product W.[E x T]

for i in range(_n):
    for j in range(_n):
        tmp = 0
        for k in range(_n):
            tmp += W[i][k]*tensorProduct[k][j]
        result[i][j] = tmp
        
l_result = convertElementToLatex(result, t.convertToLatex(), _n)
print('W.[[E] x T] = ')
print(l_result, '\n')

# second perform dot then tensor product - E x [W.T]

dotProduct  = np.array([[[[0.0]*_n]*_n]*_n]*_n)

for i in range(_n):
    for j in range(_n):
        tmp = 0
        for k in range(_n):
            tmp += W[i][k]*T[k][j]
        dotProduct[i][j] = tmp

l_dotProduct = convertElementToLatex(dotProduct, t.convertToLatex(), _n)
print('[W.T] = ', l_dotProduct, '\n')

result_1 = np.array([[[[0.0]*_n]*_n]*_n]*_n)


# tensor product E x dotProduct

for k in [E]:
    for i in range(_n):
        for j in range(_n):
            result_1[i][j] = np.dot(k,dotProduct[i][j])


l_result_1 = convertElementToLatex(result_1, t.convertToLatex(), _n)
print('[E] x [W.T] = ', l_result_1,'\n')

l_diff = convertElementToLatex(result_1 - result, t.convertToLatex(), _n)
print('difference = ', l_diff, '\n')




