# -*- coding: utf-8 -*-
"""
Created on Wed Jan 25 15:09:08 2023

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

T = [[ np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])], [np.array([[9,10],[11,12]]), np.array([[13,14],[15,16]])]]
T = np.array(T)

inPosLst = [t.pos.up, t.pos.down, t.pos.down, t.pos.up]

verbose = False
unprimed = True

result = t4.computeTensorOuterProduct(T, inPosLst, unprimed, 2, verbose)
l_result = t4.convertElementToLatex(result, 2)
print('Original Configuration = ', inPosLst, '\n')
print('4th order tensor by outer product\n')
print(l_result, '\n')

result_1 = t4.computeTensorInnerProduct(T, inPosLst, unprimed, 2, verbose)
l_result = t4.convertElementToLatex(result_1, 2)
print('\nOriginal Configuration' '4th order tensor by inner product\n', l_result,'\n')
l_diff = t4.convertElementToLatex(result_1 - result, 2)
print('Difference with the outer product calculation = ', l_diff, '\n')


# declare the metric and inverse metric
 
G = np.array([[1, 0],[0, r**2]])
Ginv = np.linalg.inv(G)

# Set up the initial configuration to be [t.pos.up, t.pos.up, t.pos.up, t.pos.up]

init_posLst = [t.pos.up, t.pos.up, t.pos.up, t.pos.up]
T_init = t4.changeConfig(T, inPosLst, init_posLst, G, Ginv, 2, verbose)

print('Initial Configuration = ', init_posLst, '\n')
l_T_init =  t4.convertElementToLatex(T_init, 2)
print('T = ', l_T_init, '\n')

# Test outer product in initial configuration

result_1 = t4.computeTensorOuterProduct(T_init, init_posLst, unprimed, 2, verbose)
l_diff = t4.convertElementToLatex(result_1 - result, 2)
print('Difference of the Initial Configuration Outer Product with the Original Configuration = ', l_diff, '\n')

# Test inner product in initial configuration

result_2 = t4.computeTensorInnerProduct(T, inPosLst, unprimed, 2, verbose)
l_diff = t4.convertElementToLatex(result_2 - result, 2)
print('Difference of the Initial Configuration Inner Product with the Original Configuration = ', l_diff, '\n')



posLst = [[t.pos.up, t.pos.up, t.pos.up, t.pos.down], [t.pos.up, t.pos.up, t.pos.down, t.pos.up],\
          [t.pos.up, t.pos.up, t.pos.down, t.pos.down], [t.pos.up, t.pos.down, t.pos.up, t.pos.up],\
          [t.pos.up, t.pos.down, t.pos.up, t.pos.down], [t.pos.up, t.pos.down, t.pos.down, t.pos.up],\
          [t.pos.up, t.pos.down, t.pos.down, t.pos.down],[t.pos.down, t.pos.up, t.pos.up, t.pos.up],\
          [t.pos.down, t.pos.up, t.pos.up, t.pos.down], [t.pos.down, t.pos.up, t.pos.down, t.pos.up],\
          [t.pos.down, t.pos.up, t.pos.down, t.pos.down], [t.pos.down, t.pos.down, t.pos.up, t.pos.up],\
          [t.pos.down, t.pos.down, t.pos.up, t.pos.down], [t.pos.down, t.pos.down, t.pos.down, t.pos.up],\
          [t.pos.down, t.pos.down, t.pos.down, t.pos.down]]

unprimed = False
verbose = False

for item in posLst:
    T_config = 0
    T_config = t4.changeConfig(T_init, init_posLst, item, G, Ginv, 2, verbose)
    print('configuration = ', item, '\n')
    print('T = ', T_config, '\n')
    # transform tensor components
    T_config_Prime, T_Prime_ijkl = t4.transformTensor(T_config, item, 2, verbose)
    result_3 = t4.computeTensorOuterProduct(T_Prime_ijkl, item, unprimed, 2, verbose)
    print('4th order tensor by outer product\n')
    l_result = t4.convertElementToLatex(result_3, 2)
    print(l_result)
    print('\n')
    l_diff = t4.convertElementToLatex(result_3 - result, 2)
    print('Difference of Outer Product with the Original configuration = ', l_diff, '\n')
    result_4 = t4.computeTensorInnerProduct(T_config_Prime, item, unprimed, 2, verbose)
    l_diff = t4.convertElementToLatex(result_4 - result, 2)
    print('Difference of Inner Product with the Original configuration = ', l_diff, '\n')
    
# raise the different combinations of indices

posLst.reverse()
del posLst[0]
posLst[-1] = [t.pos.up, t.pos.up, t.pos.up, t.pos.up]
init_posLst = [t.pos.down, t.pos.down, t.pos.down, t.pos.down]

print('Running configurations in Reverse\n')
print('Initial Configuration = ', inPosLst, '\n')
T_init = T_config
print('T = ', T_init, '\n')

for item in posLst:
  T_config = 0
  T_config = t4.changeConfig(T_init, init_posLst, item, G, Ginv, 2, verbose)
  print('configuration = ', item, '\n')
  print('T = ', T_config, '\n')
  # transform tensor components
  T_config_Prime, T_Prime_ijkl = t4.transformTensor(T_config, item, 2, verbose)
  result_3 = t4.computeTensorOuterProduct(T_Prime_ijkl, item, unprimed, 2, verbose)
  print('4th order tensor by outer product\n')
  l_result = t4.convertElementToLatex(result_3, 2)
  print(l_result)
  print('\n')
  l_diff = t4.convertElementToLatex(result_3 - result, 2)
  print('Difference of Outer Product with the Original configuration = ', l_diff, '\n')
  result_4 = t4.computeTensorInnerProduct(T_config_Prime, item, unprimed, 2, verbose)
  l_diff = t4.convertElementToLatex(result_4 - result, 2)
  print('Difference of Inner Product with the Original configuration = ', l_diff, '\n')
 





