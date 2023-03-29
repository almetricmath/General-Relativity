# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:13:10 2022

@author: Al
"""

# This program tests tensor various transforms

# second order contravariant tensor

import numpy as np

def polarVectorBasis(_r, _theta):
    ret = np.array([[np.cos(_theta),np.sin(_theta)],[-_r*np.sin(_theta), _r*np.cos(_theta)]])
    return ret

def polarOneFormBasis(_r, _theta):
    ret = np.array([[np.cos(_theta), np.sin(_theta)],[-np.sin(_theta)/_r, np.cos(_theta)/_r]])
    return ret

def polarSqrtAMatrix(_r1, _theta1):
    ret = np.array([[1/(2*np.sqrt(_r1)),0],[0,1/(2*np.sqrt(_theta1)) ]])
    return ret
 
def polarSqrtBMatrix(_r1, _theta1):
    ret = np.array([[2*np.sqrt(_r1), 0],[0, 2*np.sqrt(_theta1)]])
    return ret
    
def computeSecondOrderTensorOuterProduct(_T, _Basis_1, _Basis_2, _n):
    
    ret = 0
    for i in range(_n):
        for j in range(_n):
            v_i = _Basis_1[i,:]
            v_j =  _Basis_2[j,:]
            ret += _T[i,j]*np.einsum('i,j', v_i, v_j)
    return ret

def computeSecondOrderTensorInnerProduct(_T, _Basis_1, _Basis_2):
    
    tmp = np.dot(np.transpose(_Basis_1),_T)
    ret = np.dot(tmp, _Basis_2)
    return ret
    
def computeThirdOrderTensorCustom(_T, _Basis_1, _Basis_2, _Basis_3, _n):
    
    # put basis matrices into vectors
    
    b_1 = []
    for i in _Basis_1:
        b_1.append(i)

    b_1 = np.array(b_1)
    
    b_2 = []
    for i in _Basis_2:
        b_2.append(i)

    b_2 = np.array(b_2)
    
    
    # Combine weight matrix
    
    T_weight = []
    
    for j in range(_n):
        T_tmp = 0
        for i in range(_n):
            T_tmp += _T[i]*_Basis_3[i,j]
        T_weight.append(T_tmp)
      
    T_weight = np.array(T_weight) 
   
   # Perform outer product 
   
    T = [] 
    
    for k in range(_n):
        T_tmp = 0
        for i in range(_n):
            for j in range(_n):
                T_tmp += T_weight[k,i,j]*np.einsum('i,j', b_1[i],b_2[j])
        T.append(T_tmp)
                
             
    return np.array(T)
  
def computeThirdOrderTensorOuterProduct(_T, _Basis_1, _Basis_2, _Basis_3, _n):
    
    # put basis matrices into vectors
    
    b_1 = []
    for i in _Basis_1:
        b_1.append(i)

    b_1 = np.array(b_1)
    
    b_2 = []
    for i in _Basis_2:
        b_2.append(i)

    b_2 = np.array(b_2)
    
    b_3 = []
    for i in _Basis_3:
        b_3.append(i)

    b_3 = np.array(b_3)
    

    ret = 0
    
    for i in range(_n):
        for j in range(_n):
            for k in range(_n):
                tmp = _T[j,k,i]
                ret += tmp*np.einsum('i,j,k',b_1[j], b_2[k], b_3[i])
    return ret
  
    
def computeThirdOrderTensorInnerProduct(_T, _Basis_1, _Basis_2, _Basis_3, _n): 
    
    
    # Compute weight matrix
    
    T_weight = []
    
    for j in range(_n):
        T_tmp = 0
        for i in range(_n):
            T_tmp += _T[i]*_Basis_3[i,j]
        T_weight.append(T_tmp)
      
    T_weight = np.array(T_weight) 
    
    # compute T for each element
    ret = []
    
    for i in range(_n):
        tmp = np.dot(np.transpose(_Basis_1), T_weight[i])
        tmp = np.dot(tmp, _Basis_2)
        ret.append(tmp)
        
    return ret
 
def transformThirdOrderTensor(_T, _Matrix_1, _Matrix_2, _Matrix_3 ):
   
    n = np.shape(_Matrix_3)[0]
  
    # Compute weight matrix
     
    T_weight = []
     
    for j in range(n):
        T_tmp = 0
        for i in range(n):
            T_tmp += _T[i]*_Matrix_3[i,j]
        T_weight.append(T_tmp)
       
    T_weight = np.array(T_weight) 
          
    # compute T for each element
    ret = []
    
    for i in range(n):
        tmp = np.dot(np.transpose(_Matrix_1), T_weight[i])
        tmp = np.dot(tmp, _Matrix_2)
        ret.append(tmp)
        
    return np.array(ret)

def computeWeight(_TW, _b_1, _b_2, _n):
    
    ret = 0

    for i in range(_n):
        for j in range(_n):
            tmp = np.einsum('i,j',_b_1[i],_b_2[j])
            ret += _TW[i,j]*tmp
        
    return np.array(ret)
   
def computeFourthOrderWeightMatrix(_T, _Basis_1, _Basis_2, _n):
    
    b_1 = []
    for i in _Basis_1:
        b_1.append(i)

    b_1 = np.array(b_1)
    
    b_2 = []
    for i in _Basis_2:
        b_2.append(i)

    b_2 = np.array(b_2)
    
    ret = []
    
    for i in range(_n):
        for j in range(_n):
            TW = _T[i,j]
            result = computeWeight(TW, b_1, b_2, _n)
            ret.append(result)
    
    return np.array(ret)
  
def allocateFourthOrderBasis(_n):
    ret=np.array([[[[[[0.0]*_n]*_n]*_n]*_n]*_n]*_n)
    return ret
def allocateFourthOrderElement(_n):
    ret_ij = np.array([[[[0.0]*_n]*_n]*_n]*_n)
    return ret_ij

def computeFourthOrderTensorOuterProduct(_T, _Basis_1, _Basis_2, _Basis_3, _Basis_4, _n):

    b_1 = []
    for i in _Basis_1:
        b_1.append(i)

    b_1 = np.array(b_1)
    
    b_2 = []
    for i in _Basis_2:
        b_2.append(i)

    b_2 = np.array(b_2)
   
    b_3 = []
    for i in _Basis_3:
        b_3.append(i)

    b_3 = np.array(b_3)
    
    b_4 = []
    for i in _Basis_4:
        b_4.append(i)

    b_4 = np.array(b_4)
    
    ret=np.array([[[[[[0.0]*_n]*_n]*_n]*_n]*_n]*_n)

    for i in range(_n):
        for j in range(_n):
            result = 0
            for k in range(_n):
                for l in range(_n):
                    tmp = _T[i, j, k, l]
                    result += tmp*np.einsum('i,j,k,l',b_1[k], b_2[l], b_3[i],b_4[j])
            ret[i][j] = result
        
    return ret


def computeFourthOrderMatrixOuterProduct(_T, _Basis_1, _Basis_2, _Basis_3, _Basis_4, _n):
    
    TW = computeFourthOrderWeightMatrix(_T, _Basis_1, _Basis_2, _n)
    
    b_3 = []
    for i in _Basis_3:
        b_3.append(i)

    b_3 = np.array(b_3)
    
    b_4 = []
    for i in _Basis_4:
        b_4.append(i)

    b_4 = np.array(b_4)
   
    ret = allocateFourthOrderBasis(_n)
    
    for k in range(_n):
        for l in range(_n):
            index = k*_n + l
            tw = TW[index]
            ret_ij = allocateFourthOrderElement(_n)
            for i in range(_n):
                for j in range(_n):
                    tmp = tw[i,j]
                    basis = np.einsum('i,j', b_3[k], b_4[l] )
                    ret_ij[i][j] = tmp*basis
            ret[k][l] = ret_ij
            
    return ret 


# convert a matrix to  Latex

def convertMatrixToLatex(_result, _n):
    
    ret = '\\bmatrix{'
       
    for i in range(_n):
        for j in range(_n):
            ret += str("{:.8f}".format(_result[i][j]))
            if j != _n - 1:
                ret += '&'
                
        if i != _n - 1:
            ret += '\\\\'

    ret += '}'
    return ret
        
def createLatexBlockMatrix(_matrixLst, _n):
    
    ret = '\\bmatrix{'
    
    for i in range(_n):
        for j in range(_n):
            index = i*_n + j
            ret += convertMatrixToLatex(_matrixLst[index], _n)
            if j != _n - 1:
                ret += '&'
        if i != _n - 1:
            ret += '\\\\' 
        
    ret += '}'
    
    return ret
           
def convertResultsToLatex(_result, _n):
    
    ret = []
    for i in range(2):
        for j in range(2):
            matrixLst = []
            for k in range(2):
                for l in range(2):
                    matrixLst.append(result12[i][j][k][l])
            ret.append(createLatexBlockMatrix(matrixLst, 2))
    
    return ret
           
            
# compute 2nd order tensor in polar coordinates 
# using both outer product and inner product

r = np.sqrt(6)
theta = np.pi/13

E = polarVectorBasis(r, theta)
e1 = E[0,:]
e2 = E[1,:]
W = polarOneFormBasis(r, theta)

# 2nd order tensor
# contravariant tensor

# create tensor 
T = np.array([[1,2],[3,4]])

result = computeSecondOrderTensorOuterProduct(T, E, E, 2)

print('Tensor computed by outer product\n')
print(result)
print('\n')

# compute tensor using transpose(E).T.E

result1 = computeSecondOrderTensorInnerProduct(T, E, E)
print('Tensor computed by transpose(E).T.E\n')
print(result1)
print('\n')

# transform 2nd order contravariant tensor
# polar based transform

A = polarSqrtAMatrix(r, theta)
B = polarSqrtBMatrix(r,theta)

E1 = np.dot(A,E)
W1 = np.dot(np.transpose(B),W)

# compute second order tensor in the polarSqrt system
# using T1 = transpose(B).T.B

tmp = np.dot(np.transpose(B),T)
T1 = np.dot(tmp, B)

print('T1 = ', T1, '\n')

result3 = computeSecondOrderTensorOuterProduct(T1, E1, E1, 2)
        
print('Tensor computed by outer product\n')
print(result3)
print('\n')

result4 = computeSecondOrderTensorInnerProduct(T1, E1, E1)
print('Tensor computed by transpose(E1).T1.E1\n')
print(result4)
print('\n')

# compute 3rd order tensor using the outer product

r = 2
theta = np.pi/3
E = polarVectorBasis(r, theta)

T = np.array([[[1, 2],[3, 4]],[[5, 6],[7, 8]]])
result5 = computeThirdOrderTensorCustom(T, E, E, E, 2)

print('3rd order tensor by outer product\n')
print(result5)
print('\n')

# compute 3rd order tensor using the inner product

result6 = computeThirdOrderTensorInnerProduct(T, E, E, E, 2)
print('3rd order tensor by outer product\n')
print(result6)
print('\n')

result7 = computeThirdOrderTensorOuterProduct(T, E, E, E, 2)
print('3rd order tensor by outer product\n')
print(result7)
print('\n')



# transform 3rd order contravariant tensor
# polar based transform

r1 = r**2
theta1 = theta**2
W = polarOneFormBasis(r, theta)
 

A = polarSqrtAMatrix(r1, theta1)
B = polarSqrtBMatrix(r1,theta1)

E1 = np.dot(A,E)
W1 = np.dot(np.transpose(B),W)


T1 = transformThirdOrderTensor(T, B, B, B )


result8 = computeThirdOrderTensorCustom(T1, E1, E1, E1, 2)
print('3rd order transformed tensor by outer product\n')
print(result8)
print('\n')

# compute transformed tensor using transpose(E1).T1_E1_1.E1

result9 = computeThirdOrderTensorInnerProduct(T1, E1, E1, E1, 2)
print('3rd order Tensor computed by transpose(E1).T_1.E1\n')
print(result9)
print('\n')


result10 = computeThirdOrderTensorOuterProduct(T1, E1, E1, E1, 2)
print('3rd order Tensor computed using the outer product\n')
print(result10)
print('\n')

# compute 4th order tensor using the outer product

TW = [[ np.array([[1,2],[3,4]]), np.array([[5,6],[7,8]])], [np.array([[9,10],[11,12]]), np.array([[13,14],[15,16]])]]
TW = np.array(TW)

# compute weight matrix
 
result11 = computeFourthOrderTensorOuterProduct(TW, E, E, E, E, 2)

result12 = computeFourthOrderMatrixOuterProduct(TW,E, E, E, E, 2)
l_results = convertResultsToLatex(result12, 2)
for lm in l_results:
    print('\n')
    print(lm)
    print('\n')




