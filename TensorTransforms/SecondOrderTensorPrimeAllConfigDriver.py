# -*- coding: utf-8 -*-
"""
Created on Wed Dec  7 12:22:08 2022

@author: Al
"""

import numpy as np
import TensorTransforms as t

# compute 2nd order tensor in polar coordinates 
# using both outer product and inner product

r = 2
theta = np.pi/3

# 2nd order tensor
# contravariant tensor

secondOrder = t.secondOrderTensor(r, theta)
verbose = False
utils = t.utils()

# initial configuration

# create tensor 
T = np.array([[1,2],[3,4]]) # [up, up] indices

inPosLst = [t.pos.up, t.pos.up]

# Test configurations - convert T[up, up] to possible configurations and see that
# the results are the same

posLst = [ [t.pos.down, t.pos.up], [t.pos.up, t.pos.down], [t.pos.down, t.pos.down] ] 

# initial run

# compute tensor by outer products

result_outer = secondOrder.computeTensorOuterProduct(T, inPosLst, True, 2, verbose)
print('Initial Second Order Tensor Components - ', inPosLst,'-\n')
print('T = ', T, '\n\n')
print('Full Tensor, T', inPosLst, 'computed by outer products\n')
print(result_outer, '\n\n')

# compute tensor by inner product

result_inner = secondOrder.computeTensorInnerProduct(T, inPosLst, True, 2, verbose)

print('Full Tensor, T', inPosLst, 'computed by inner products\n')
print(result_inner, '\n\n')

print('Difference between outer product and inner product computations in unprimed system\n')
print(result_inner - result_outer, '\n\n')

# Transform to Prime Coordinate System

print('Transform to polar sqrt coordinate system')

T1 = secondOrder.transformTensor(T, inPosLst, 2)
print('T̅ = ', T1, '\n\n')

# compute tensor in primed coordinates using the outer product

result_prime_outer = secondOrder.computeTensorOuterProduct(T1, inPosLst, False, 2, verbose)

print('Difference between outer products in primed and unprimed systems\n')
print(result_prime_outer - result_outer, '\n\n')

# compute tensor in primed coordinates using the inner product

result_prime_inner = secondOrder.computeTensorInnerProduct(T1, inPosLst, False, 2, verbose)
print('Difference between inner products in primed and unprimed systems\n')
print(result_prime_inner - result_inner, '\n\n')


# Run various configurations

G = np.array([[1, 0],[0, r**2]])
Ginv = np.linalg.inv(G)


for i in posLst:
    T_config = secondOrder.changeConfig(T, inPosLst, i, G, Ginv)
    # transform Tensor
    T1_config = secondOrder.transformTensor(T_config, i, 2)
    # Compute using outer products
    
    result_config_prime_outer = secondOrder.computeTensorOuterProduct(T1_config, i, False, 2, verbose)
    print('T̅ ', inPosLst, ' -> T̅', i, ' = ',T1_config, '\n\n')
    print('Difference in Full Tensor, T̅', i, ' and T', inPosLst, 'computed by outer products, \n')
    print(result_config_prime_outer - result_outer,'\n\n')
    
    # compute using inner product
    result_prime_config_inner = secondOrder.computeTensorInnerProduct(T1_config, i, False, 2, verbose)
    print('Difference in Full Tensor, T̅', i, ' and T', inPosLst, 'computed by inner products, \n')
    print(result_prime_config_inner - result_inner,'\n\n')

    
    
    
    
    

