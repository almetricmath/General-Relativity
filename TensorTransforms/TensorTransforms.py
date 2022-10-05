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
    
    #ret=np.array([[[[[[0.0]*_n]*_n]*_n]*_n]*_n]*_n)
    result = 0

    for i in range(_n):
        for j in range(_n):
            #result = 0
            for k in range(_n):
                for l in range(_n):
                    tmp = _T[i, j, k, l]
                    result += tmp*np.einsum('i,j,k,l',b_1[k], b_2[l], b_3[i],b_4[j])
            #ret[i][j] = result
        
    return result


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
   
    bases = []
    coords = []
    
    ret = allocateFourthOrderBasis(_n)
    bases = allocateFourthOrderBasis(_n)
    
    for k in range(_n):
        for l in range(_n):
            index = k*_n + l
            tw = TW[index]
            ret_ij = allocateFourthOrderElement(_n)
            basis_ij = allocateFourthOrderElement(_n)
            coord_ij = np.array([[0.0, 0.0],[0.0, 0.0]])
            for i in range(_n):
                for j in range(_n):
                    tmp = tw[i,j]
                    coord_ij[i][j] = tmp
                    basis_ij[i][j] = np.einsum('i,j', b_3[k], b_4[l] )
                    ret_ij[i][j] = tmp*basis_ij[i][j]
            coords.append(coord_ij)
            bases[k][l] = basis_ij
            ret[k][l] = ret_ij
            
    return ret, np.array(coords), np.array(bases)


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
        

def convertResultToLatex(_result, _weights, _n):
    
    ret = '\\bmatrix{' 
    
    for i in range(_n):
        for j in range(_n):
            tmp = _result[i][j]
            if type(_weights) == np.ndarray:
                ret +=  str('(' + "{:.8f}".format(_weights[i][j])) + ')'
            ret += convertMatrixToLatex(tmp, _n)
            if j != _n - 1:
                ret += '&'
        if i != _n - 1:
            ret += '\\\\' 
        
    ret += '}'
    
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
l_results = convertResultToLatex(result11, 0, 2)
print('Summed Results\n')
print(l_results)
print('\n')

 
result12, coords, bases = computeFourthOrderMatrixOuterProduct(TW,E, E, E, E, 2)

for i in range(2):
    for j in range(2):
        print('weighted sub-matrix',str(i),str(j),'\n')
        index = i*2 + j
        l_result = convertResultToLatex(bases[i][j], coords[index], 2)
        print(l_result, '\n')
        print('result sub-matrix', str(i), str(j),'\n')
        l_result = convertResultToLatex(result12[i][j],0, 2)
        print(l_result, '\n')
        
        
# compute T_ij using  ET(TW)E       
# T_11     
   
l_result = convertMatrixToLatex(E, 2)
print('E Matrix')
print(l_result)
print('\n')

l_result = convertMatrixToLatex(np.transpose(E), 2)
print('ET Matrix')
print(l_result)
print('\n')

# compute T_11 matrix

# Test ET(Tijkl)E = ET(TW)E

T11 = TW[0][0]
tmp = np.dot(T11, E)
T_11 = np.dot(np.transpose(E),tmp)
l_result = convertMatrixToLatex(T_11, 2)
print(' T_11 components - ET(T11)E\n')
print(l_result)
print('\n')

# compute T_12 matrix

T12 = TW[0][1]
tmp = np.dot(T12, E)
T_12 = np.dot(np.transpose(E),tmp)
l_result = convertMatrixToLatex(T_12, 2)
print(' T_12 components - ET(T12)E\n')
print(l_result)
print('\n')

# compute T_21 matrix

T21 = TW[1][0]
tmp = np.dot(T21, E)
T_21 = np.dot(np.transpose(E),tmp)
l_result = convertMatrixToLatex(T_21, 2)
print(' T_21 components - ET(T21)E\n')
print(l_result)
print('\n')

# compute T_22 matrix

T22 = TW[1][1]
tmp = np.dot(T22, E)
T_22 = np.dot(np.transpose(E),tmp)
l_result = convertMatrixToLatex(T_22, 2)
print(' T_22 components - ET(T22)E\n')
print(l_result)
print('\n')


# add submatrices to get the final result
summed_matrix = allocateFourthOrderElement(2)
for i in range(2):
    for j in range(2):
        summed_matrix += result12[i][j]

l_result = convertResultToLatex(summed_matrix, 0, 2)
print('Summed Results\n')
print(l_result)
print('\n')
        


# 11 component

T__11 = np.array([[coords[0][0][0],coords[1][0][0]],[coords[2][0][0],coords[3][0][0]]])
l_result = convertMatrixToLatex(T__11, 2)
print('T__11 matrix \n')
print(l_result)
print('\n')



tmp = np.dot(T__11, E)
Tsum_11 = np.dot(np.transpose(E),tmp)
l_result = convertMatrixToLatex(Tsum_11, 2)
print('Summed Results 11 components \n')
print(l_result)
print('\n')

# transform T_kl back to T_ijkl using T_ijkl = W.T_kl.WT

l_result = convertMatrixToLatex(W, 2)
print('W matrix \n')
print(l_result)
print('\n')

l_result = convertMatrixToLatex(np.transpose(W), 2)
print('WT matrix \n')
print(l_result)
print('\n')


tmp = np.dot(T_11, np.transpose(W))
testT_ij11 = np.dot(W, tmp)
l_result = convertMatrixToLatex(testT_ij11,2)
print('testT_ij11 matrix \n')
print(l_result)
print('\n')

# transform 4th order contravariant tensor
# polar based transform

r = 2
theta = np.pi/3
E = polarVectorBasis(r, theta)
W = polarOneFormBasis(r, theta)
r1 = r**2
theta1 = theta**2
A = polarSqrtAMatrix(r1, theta1)
B = polarSqrtBMatrix(r1,theta1)
E1 = np.dot(A,E)

l_result = convertMatrixToLatex(E1,2)
print('E1 matrix \n')
print(l_result)
print('\n')

l_result = convertMatrixToLatex(np.transpose(E1),2)
print('E1T matrix \n')
print(l_result)
print('\n')


W1 = np.dot(np.transpose(B),W)


#compute [T]i1,j1
print('4th order coordinate change\n')



l_result = convertMatrixToLatex(np.transpose(B),2)
print('BT matrix \n')
print(l_result)
print('\n')

l_result = convertMatrixToLatex(B,2)
print('B matrix \n')
print(l_result)
print('\n')

T__11 = np.array([[coords[0][0][0],coords[1][0][0]],[coords[2][0][0],coords[3][0][0]]])
l_result = convertMatrixToLatex(T__11, 2)
print('T__11 matrix \n')
print(l_result)
print('\n')

# compute [T]_ij in the primed coordinate system
# 11 component

tmp = np.dot(T__11, B)
Tprime__11 = np.dot(np.transpose(B),tmp)
l_result = convertMatrixToLatex(Tprime__11, 2)
print('Tprime__11 matrix\n')
print(l_result)
print('\n')

# 12 component

T__12 = np.array([[coords[0][0][1],coords[1][0][1]],[coords[2][0][1],coords[3][0][1]]])
l_result = convertMatrixToLatex(T__12, 2)
print('T__12 matrix \n')
print(l_result)
print('\n')

tmp = np.dot(T__12, B)
Tprime__12 = np.dot(np.transpose(B),tmp)
l_result = convertMatrixToLatex(Tprime__12, 2)
print('Tprime__12 matrix\n')
print(l_result)
print('\n')

# 21 component

T__21 = np.array([[coords[0][1][0],coords[1][1][0]],[coords[2][1][0],coords[3][1][0]]])
l_result = convertMatrixToLatex(T__21, 2)
print('T__21 matrix \n')
print(l_result)
print('\n')

tmp = np.dot(T__21, B)
Tprime__21 = np.dot(np.transpose(B),tmp)
l_result = convertMatrixToLatex(Tprime__21, 2)
print('Tprime__21 matrix\n')
print(l_result)
print('\n')

# 22 component

T__22 = np.array([[coords[0][1][1],coords[1][1][1]],[coords[2][1][1],coords[3][1][1]]])
l_result = convertMatrixToLatex(T__22, 2)
print('T__22 matrix \n')
print(l_result)
print('\n')

tmp = np.dot(T__22, B)
Tprime__22 = np.dot(np.transpose(B),tmp)
l_result = convertMatrixToLatex(Tprime__22, 2)
print('Tprime__22 matrix\n')
print(l_result)
print('\n')

# compute the complete tensor for each submatrix

# 11 component

tmp = np.dot(Tprime__11, E1)
Tprime__11Sum = np.dot(np.transpose(E1),tmp)
l_result = convertMatrixToLatex(Tprime__11Sum, 2)
print('Tprime__11Sum matrix\n')
print(l_result)
print('\n')

# 12 component

tmp = np.dot(Tprime__12, E1)
Tprime__12Sum = np.dot(np.transpose(E1),tmp)
l_result = convertMatrixToLatex(Tprime__12Sum, 2)
print('Tprime__12Sum matrix\n')
print(l_result)
print('\n')

# 21 component

tmp = np.dot(Tprime__21, E1)
Tprime__21Sum = np.dot(np.transpose(E1),tmp)
l_result = convertMatrixToLatex(Tprime__21Sum, 2)
print('Tprime__21Sum matrix\n')
print(l_result)
print('\n')

# 22 component

tmp = np.dot(Tprime__22, E1)
Tprime__22Sum = np.dot(np.transpose(E1),tmp)
l_result = convertMatrixToLatex(Tprime__22Sum, 2)
print('Tprime__22Sum matrix\n')
print(l_result)
print('\n')




















