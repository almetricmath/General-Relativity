# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:13:10 2022

@author: Al
"""

# This program tests tensor various transforms

# second order contravariant tensor

import numpy as np

class coordinateTransforms:

    def polarVectorBasis(self, _r, _theta):
        ret = np.array([[np.cos(_theta),np.sin(_theta)],[-_r*np.sin(_theta), _r*np.cos(_theta)]])
        return ret
    
    def polarOneFormBasis(self, _r, _theta):
        ret = np.array([[np.cos(_theta), np.sin(_theta)],[-np.sin(_theta)/_r, np.cos(_theta)/_r]])
        return ret
    
    def polarSqrtAMatrix(self, _r1, _theta1):
        ret = np.array([[1/(2*np.sqrt(_r1)),0],[0,1/(2*np.sqrt(_theta1)) ]])
        return ret
     
    def polarSqrtBMatrix(self, _r1, _theta1):
        ret = np.array([[2*np.sqrt(_r1), 0],[0, 2*np.sqrt(_theta1)]])
        return ret

class secondOrderTensor:

    def __init__(self):
        self._latex = convertToLatex()

    def computeTensorOuterProduct(self, _T, _Basis_1, _Basis_2, _n):
        
        ret = 0
        for i in range(_n):
            for j in range(_n):
                v_i = _Basis_1[i,:]
                v_j =  _Basis_2[j,:]
                self.printVector(v_i, True,'v_' + str(i+1) + '^T', _n)
                self.printVector(v_j, False,'v_' + str(j+1), _n) 
                tmp = _T[i,j]
                print('T_' + str(i+1) + str(j+1) + '= ', tmp)
                outer = np.einsum('i,j', v_i, v_j)
                self.printMatrix(outer,'v_' + str(i+1) + 'v_' + str(j+1), _n)
                ret += tmp*outer
        return ret
    
    def computeTensorInnerProduct(self, _T, _Basis_1, _Basis_2):
        
        tmp = np.dot(np.transpose(_Basis_1),_T)
        ret = np.dot(tmp, _Basis_2)
        return ret

    def printMatrix(self, _M, _label, _n):
        l_result = self._latex.convertMatrixToLatex(_M, _n)
        print(_label + ' = ', l_result, '\n')
        
    def printVector(self, _vec, _transpose, _label, _n):
        l_result = self._latex.converVectorToLatex(_vec, _transpose, _n)
        print(_label + ' = ', l_result, '\n')
        
 
class thirdOrderTensor:
    
    def computeTensorCustom(self, _T, _Basis_1, _Basis_2, _Basis_3, _n):
        
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
      
    def computeTensorOuterProduct(self, _T, _Basis_1, _Basis_2, _Basis_3, _n):
        
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
      
        
    def computeTensorInnerProduct(self, _T, _Basis_1, _Basis_2, _Basis_3, _n): 
        
        
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
     
    def transformTensor(self, _T, _Matrix_1, _Matrix_2, _Matrix_3 ):
       
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
    
    def computeWeight(self, _TW, _b_1, _b_2, _n):
        
        ret = 0
    
        for i in range(_n):
            for j in range(_n):
                tmp = np.einsum('i,j',_b_1[i],_b_2[j])
                ret += _TW[i,j]*tmp
            
        return np.array(ret)

class fourthOrderTensor:
    
    def computeFourthOrderWeightMatrix(self, _T, _Basis_1, _Basis_2, _n):
        
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
                result = self.computeWeight(TW, b_1, b_2, _n)
                ret.append(result)
        
        return np.array(ret)
      
    def allocateFourthOrderBasis(self, _n):
        ret=np.array([[[[[[0.0]*_n]*_n]*_n]*_n]*_n]*_n)
        return ret
    def allocateFourthOrderElement(self, _n):
        ret_ij = np.array([[[[0.0]*_n]*_n]*_n]*_n)
        return ret_ij
    
    def computeFourthOrderTensorOuterProduct(self, _T, _Basis_1, _Basis_2, _Basis_3, _Basis_4, _n):
    
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
    
    
    def computeFourthOrderMatrixOuterProduct(self, _T, _Basis_1, _Basis_2, _Basis_3, _Basis_4, _n):
        
        TW = self.computeFourthOrderWeightMatrix(_T, _Basis_1, _Basis_2, _n)
        
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
        
        ret = self.allocateFourthOrderBasis(_n)
        bases = self.allocateFourthOrderBasis(_n)
        
        for k in range(_n):
            for l in range(_n):
                index = k*_n + l
                tw = TW[index]
                ret_ij = self.allocateFourthOrderElement(_n)
                basis_ij = self.allocateFourthOrderElement(_n)
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


class convertToLatex:

    def convertMatrixToLatex(self, _matrix, _n):
        
        ret = '\\bmatrix{'
           
        for i in range(_n):
            for j in range(_n):
                ret += str("{:.6f}".format(_matrix[i][j]))
                if j != _n - 1:
                    ret += '&'
                    
            if i != _n - 1:
                ret += '\\\\'
    
        ret += '}'
        return ret
    
    def converVectorToLatex(self, _vec, transposeFlag, _n):
    
        ret = '\\bmatrix{'
        
        for i in range(_n):
            ret += str("{:.6f}".format(_vec[i]))
            if i != _n - 1:
                if not transposeFlag:
                    ret += '&'
                else:
                    ret += '\\\\'
        
        ret += '\\\\}'
        return ret

    def convertResultToLatex(self, _result, _weights, _n):
        
        ret = '\\bmatrix{' 
        
        for i in range(_n):
            for j in range(_n):
                tmp = _result[i][j]
                if type(_weights) == np.ndarray:
                    ret +=  str('(' + "{:.6f}".format(_weights[i][j])) + ')'
                ret += self.convertMatrixToLatex(tmp, _n)
                if j != _n - 1:
                    ret += '&'
            if i != _n - 1:
                ret += '\\\\' 
            
        ret += '}'
        
        return ret
    
class variables:

    def __init__(self, _r, _theta):
        coords = coordinateTransforms()
        self._vars = {}
        self._vars['E'] = coords.polarVectorBasis(_r, _theta)
        self._vars['ET'] = np.transpose(self._vars['E'])
        self._vars['W'] = coords.polarOneFormBasis(_r, _theta)
        r1 = _r**2
        theta1 = _theta**2
        self._vars['A'] = coords.polarSqrtAMatrix(r1, theta1)
        self._vars['AT'] = np.transpose(self._vars['A'])
        self._vars['B'] = coords.polarSqrtBMatrix(r1, theta1) 
        self._vars['BT'] = np.transpose(self._vars['B'])
        self._vars['E1'] = np.dot(self._vars['A'],self._vars['E'])
        self._vars['E1T'] = np.transpose(self._vars['E'])
        self._vars['W1'] = np.dot(np.transpose(self._vars['B']), self._vars['W'])
        self._latex = convertToLatex()
        

    def getVars(self):
        return self._vars

    def printVars(self, _lst, _n):
        for item in _lst:
            l_result = self._latex.convertMatrixToLatex(self._vars[item], _n)
            print(item + '\n')
            print(l_result)
            print('\n')







