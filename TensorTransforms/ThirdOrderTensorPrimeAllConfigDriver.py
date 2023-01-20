# -*- coding: utf-8 -*-
"""
Created on Mon Dec 12 13:12:27 2022

@author: Al
"""

import TensorTransforms as t
import numpy as np
import sys

# compute 3rd order tensor using the outer product

latex = t.convertToLatex()

coords = t.coordinateTransforms()

# initial configuration T(up, down, up)

r = 2
theta = np.pi/3
t3 = t.thirdOrderTensor(r, theta)

T = np.array([[[1.0, 2.0],[3.0, 4.0]],[[5.0, 6.0],[7.0, 8.0]]])

# specify tensor index positions

inPosLst = [t.pos.up, t.pos.down, t.pos.up]
verbose = False
unprimed = True


result = t3.computeTensorOuterProduct(T, inPosLst, unprimed, 2, verbose)
print('Original Configuration = ', inPosLst, '\n')
print('3rd order tensor by outer product\n')
l_result = t3.convertToLatex(result, 2)
print(l_result)
print('\n')

# compute 3rd order tensor using the inner product

print('3rd order tensor by inner product\n')
result1 = t3.computeTensorInnerProduct(T, inPosLst, unprimed, 2, verbose)

l_result = t3.convertToLatex(result1, 2)
print(l_result)
print('\n')

diff_result = t3.convertToLatex(result1 - result, 2)
print('Difference between the inner product and outer product tensor computations in unprimed coordinates using the original configuration\n') 
print(diff_result, '\n\n')


# declare the metric and inverse metric
 
G = np.array([[1, 0],[0, r**2]])
Ginv = np.linalg.inv(G)

posLst = [t.pos.up, t.pos.up, t.pos.up]

T_init = t3.changeConfig(T, inPosLst, posLst, G, Ginv, 2, verbose)

# Test outer product formulation

print('Initial Configuration = ', posLst, '\n')
print('T = ', T_init, '\n')

result2 = t3.computeTensorOuterProduct(T_init, posLst, unprimed, 2, verbose)
print('3rd order tensor by outer product\n')
l_result = t3.convertToLatex(result2, 2)
print(l_result)
print('\n')


diff_result = t3.convertToLatex(result2 - result, 2)
print('Difference between the outer product computations in the initial configuration and the outer product in the original configuration in unprimed coordinates\n') 
print(diff_result, '\n\n')

# Test inner product formulation

print('3rd order tensor by inner product\n')
result3 = t3.computeTensorInnerProduct(T_init, posLst, unprimed, 2, verbose)
l_result = t3.convertToLatex(result3, 2)
print(l_result, '\n')

diff_result = t3.convertToLatex(result3 - result, 2)
print('Difference between the inner product computations in the initial configuration and the outer product initial configuration in the original configuration in unprimed coordinates\n') 
print(diff_result, '\n\n')


posLst = [ [t.pos.up, t.pos.up, t.pos.down], [t.pos.up, t.pos.down, t.pos.up],  [t.pos.up, t.pos.down, t.pos.down], [t.pos.down, t.pos.up, t.pos.up], [t.pos.down, t.pos.up, t.pos.down], [t.pos.down, t.pos.down, t.pos.up], [t.pos.down, t.pos.down, t.pos.down]] 

inPosLst = [t.pos.up, t.pos .up, t.pos.up]

for item in posLst:
    T_config = 0
    T_config = t3.changeConfig(T_init, inPosLst, item, G, Ginv, 2, verbose)
    T1_n, T1_ijk = t3.transformTensor(T_config, item, 2, verbose) # transform from unprimed to prime coordinates

    print('configuration = ', item, '\n')
    print('T̅ = ', T1_ijk, '\n')
    unprimed = False # computing tensor in primed coordinate system
    result4 = t3.computeTensorOuterProduct(T1_ijk, item, unprimed, 2, verbose)
    print('3rd order tensor by outer product\n')
    l_result = t3.convertToLatex(result4, 2)
    print(l_result)
    print('\n')
    
    diff_result = t3.convertToLatex(result4 - result, 2)

    print('Difference between the outer product tensor in primed coordinates in different configurations and the outer product of the original configuration in unprimed coordinates\n') 
    print(diff_result, '\n\n')
    
    print('3rd order tensor by inner product\n')
    result5 = t3.computeTensorInnerProduct(T1_n, item, unprimed, 2, verbose)
    l_result = t3.convertToLatex(result5, 2)
    print(l_result, '\n')

    diff_result = t3.convertToLatex(result5 - result, 2)
    print('Difference between the inner product computations in different configurations in primed coordinates and the outer product of the original configuration in unprimed coordinates\n') 
    print(diff_result, '\n\n')
 

posLst = [ [t.pos.up, t.pos.up, t.pos.down], [t.pos.up, t.pos.down, t.pos.up],  [t.pos.up, t.pos.down, t.pos.down], [t.pos.down, t.pos.up, t.pos.up], [t.pos.down, t.pos.up, t.pos.down], [t.pos.down, t.pos.down, t.pos.up], [t.pos.down, t.pos.down, t.pos.down]] 
posLst.reverse()

del posLst[0]
posLst[-1] = [t.pos.up, t.pos.up, t.pos.up]

inPosLst = [t.pos.down, t.pos.down, t.pos.down]
 
T_init = np.array([[[  1.,   8.], [  3.,  16.]], [[ 20.,  96.], [ 28., 128.]]])

print('Running configurations in Reverse\n')
print('Initial Configuration = ', inPosLst, '\n')
print('T = ', T_init, '\n')

for item in posLst:
    T_config = 0
    T_config = t3.changeConfig(T_init, inPosLst, item, G, Ginv, 2, verbose)
    T1_n, T1_ijk = t3.transformTensor(T_config, item, 2, verbose) # transform from unprimed to prime coordinates

    print('configuration = ', item, '\n')
    print('T̅ = ', T1_ijk, '\n')
    unprimed = False # computing tensor in primed coordinate system
    result6 = t3.computeTensorOuterProduct(T1_ijk, item, unprimed, 2, verbose)
    print('3rd order tensor by outer product\n')
    l_result = t3.convertToLatex(result6, 2)
    print(l_result)
    print('\n')
    
    diff_result = t3.convertToLatex(result6 - result, 2)

    print('Difference between the outer product tensor in primed coordinates in different configurations and the outer product of the original configuration in unprimed coordinates\n') 
    print(diff_result, '\n\n')
    
    print('3rd order tensor by inner product\n')
    result7 = t3.computeTensorInnerProduct(T1_n, item, unprimed, 2, verbose)
    l_result = t3.convertToLatex(result7, 2)
    print(l_result, '\n')

    diff_result = t3.convertToLatex(result7 - result, 2)
    print('Difference between the inner product computations in different configurations in primed coordinates and the outer product of the original configuration in unprimed coordinates\n') 
    print(diff_result, '\n\n')
 

    


