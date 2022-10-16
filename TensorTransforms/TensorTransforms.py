# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:13:10 2022

@author: Al
"""

# This program tests tensor various transforms

# second order contravariant tensor

import numpy as np
import copy

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

    def __init__(self, _r, _theta):
        self._latex = convertToLatex()
        self._vars = variables(_r, _theta)

    def computeTensorOuterProduct(self, _T, _indxBasis_1, _indxBasis_2, _n):
        
        subscript_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']
        
        _Basis_1 = self._vars._vars[_indxBasis_1].value
        symbol_1 = self._vars._vars[_indxBasis_1].symbol
        _Basis_2 = self._vars._vars[_indxBasis_2].value
        symbol_2 = self._vars._vars[_indxBasis_2].symbol
        
        ret = 0
        for i in range(_n):
            for j in range(_n):
                v_i = _Basis_1[i,:]
                v_j =  _Basis_2[j,:]
                self.printVector(v_i, True, symbol_1.lower() + subscript_num[i+1] + 'ᵀ', _n)
                self.printVector(v_j, False, symbol_2.lower() + subscript_num[j+1], _n) 
                coeff = _T[i,j] 
                print('T' + subscript_num[i+1] + subscript_num[j+1] + ' = ', coeff)
                outer = np.einsum('i,j', v_i, v_j)
                self.printMatrix(outer,'outer' + subscript_num[i+1] + subscript_num[j+1], _n)
                ret += coeff*outer
        return ret
    
    def computeTensorInnerProduct(self, _T, _indxBasis_1, _indxBasis_2, _n):
        
        _Basis_1T = self._vars._vars[_indxBasis_1 + 'T'].value  # transpose of _Basis_1
        _labelBasis_1T = self._vars._vars[_indxBasis_1 + 'T'].symbol
        _Basis_2 = self._vars._vars[_indxBasis_2].value
        _labelBasis_2 = self._vars._vars[_indxBasis_2].symbol
        l_result = self._latex.convertMatrixToLatex(_T, _n)
        print('T' + '\n', l_result, '\n')
        l_result = self._latex.convertMatrixToLatex(_Basis_1T, _n)
        print(_labelBasis_1T, l_result, '\n')
        l_result = self._latex.convertMatrixToLatex(_Basis_2, _n)
        print(_labelBasis_2, l_result, '\n')
        
       
        tmp = np.dot(_Basis_1T, _T)
        ret = np.dot(tmp, _Basis_2)
        return ret

    def printMatrix(self, _M, _label, _n):
        l_result = self._latex.convertMatrixToLatex(_M, _n)
        print(_label + ' = ', l_result, '\n')
        
    def printVector(self, _vec, _transpose, _label, _n):
        l_result = self._latex.converVectorToLatex(_vec, _transpose, _n)
        print(_label + ' = ', l_result, '\n')
        
 
class thirdOrderTensor:
    
    def __init__(self, _r, _theta):
        self._latex = convertToLatex()
        self._vars = variables(_r, _theta)
    
      
    def computeTensorOuterProduct(self, _T, _labelBasis_1, _labelBasis_2, _labelBasis_3, _n):
        
        # put basis matrices into vectors
        
        subscript_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']
        
        _Basis_1 = self._vars._vars[_labelBasis_1].value
        symbol_1 = self._vars._vars[_labelBasis_1].symbol
       
        b_1 = []
        for i in _Basis_1:
            b_1.append(i)
    
        b_1 = np.array(b_1)
        
        _Basis_2 = self._vars._vars[_labelBasis_2].value
        symbol_2 = self._vars._vars[_labelBasis_2].symbol
       
        b_2 = []
        for i in _Basis_2:
            b_2.append(i)
    
        b_2 = np.array(b_2)
        
        _Basis_3 = self._vars._vars[_labelBasis_3].value
        symbol_3 = self._vars._vars[_labelBasis_3].symbol
       
        b_3 = []
        for i in _Basis_3:
            b_3.append(i)
    
        b_3 = np.array(b_3)
        
        
        ret = 0
        
        for i in range(_n):
            for j in range(_n):
                for k in range(_n):
                    coeff = _T[i,j,k]
                    print('T' + subscript_num[i+1] + subscript_num[j+1] + subscript_num[k+1] + ' = ', coeff)
                    self.printVector(b_1[i], False, symbol_1.lower() + subscript_num[i+1], _n)
                    self.printVector(b_2[j], False, symbol_2.lower() + subscript_num[j+1], _n) 
                    self.printVector(b_3[k], False, symbol_3.lower() + subscript_num[k+1], _n)
                    outer = np.einsum('i,j,k',b_1[i], b_2[j], b_3[k])
                    l_outer = self.convertToLatex(outer, _n)
                    print('outer' + subscript_num[i+1] + subscript_num[j+1] + subscript_num[k+1] + '\n')
                    print(l_outer, '\n')
                    ret += coeff*outer
        return ret
      
        
    def computeTensorInnerProduct(self, _T, _labelBasis_1, _labelBasis_2, _labelBasis_3, _n): 
        
        
        # Compute weight matrix
        
        _Basis_1 = self._vars._vars[_labelBasis_1].value
        symbol_1 = self._vars._vars[_labelBasis_1].symbol
        subscript_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']
        
        T_weight = []
        
        for i in range(_n):
            T_tmp = 0
            print('[T]' + subscript_num[i+1] + ' Calculation\n')
            for j in range(_n):
                T_tmp += _T[j]*_Basis_1[j,i]
                l_result = self._latex.convertMatrixToLatex(_T[j], _n)
                print('T' + subscript_num[j+1] +'\n', l_result, '\n')
                print(symbol_1 + subscript_num[j+1] + subscript_num[i+1] + ' = ' + str(_Basis_1[j,i]) + '\n')
            T_weight.append(T_tmp)
            l_result = self._latex.convertMatrixToLatex(T_tmp, _n)
            print('[T]' + subscript_num[i+1] +'\n', l_result, '\n')
            
          
        T_weight = np.array(T_weight)
        
            
        # compute T1 for each element
        ret = []
        _Basis_2 = self._vars._vars[_labelBasis_2].value
        symbol_2 = self._vars._vars[_labelBasis_2].symbol
        _Basis_3 = self._vars._vars[_labelBasis_3].value
        symbol_3 = self._vars._vars[_labelBasis_3].symbol
        l_result = self._latex.convertMatrixToLatex(np.transpose(_Basis_2), _n)
        print( symbol_2  + 'ᵀ = ', l_result, '\n')
        l_result = self._latex.convertMatrixToLatex(_Basis_3, _n)
        print(symbol_3 + ' = ', l_result, '\n')
       
        for i in range(_n):
            
            tmp = np.dot(np.transpose(_Basis_2), T_weight[i])
            tmp = np.dot(tmp, _Basis_3)
            ret.append(tmp) 
            
        return ret
     
  
    def printMatrix(self, _M, _label, _n):
         l_result = self._latex.convertMatrixToLatex(_M, _n)
         print(_label + ' = ', l_result, '\n')
     
    def printVector(self, _vec, _transpose, _label, _n):
         l_result = self._latex.converVectorToLatex(_vec, _transpose, _n)
         print(_label + ' = ', l_result, '\n')
         
    def convertToLatex(self, _result, _n):
        
        ret = '\\bmatrix{'     
        
        for i in range(_n):
            tmp = self._latex.convertMatrixToLatex(_result[i], _n)
            ret += tmp
            if i != _n - 1:
                ret += '&'
         
        ret += '\\\\}' 
        return ret
             

class fourthOrderTensor:
    
    def __init__(self, _r, _theta):
        self._latex = convertToLatex()
        self._vars = variables(_r, _theta)
    
    def computeWeightElement(self, _T, _i, _j, _b_1, _symbol_1, _b_2, _symbol_2, _n):
        
        subscript_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']
        ret = 0
        
        for k in range(_n):
            for l in range(_n):
                coeff = _T[k,l]
                print('T' + subscript_num[_i+1] + subscript_num[_j+1] + subscript_num[k+1] + subscript_num[l+1] + ' = ', coeff)
                self.printVector(_b_1[k], False, _symbol_1.lower() + subscript_num[k+1], _n)
                self.printVector(_b_2[l], False, _symbol_2.lower() + subscript_num[l+1], _n) 
                outer = np.einsum('k,l', _b_1[k], _b_2[l])
                l_outer = self._latex.convertMatrixToLatex(outer, _n)
                print('outer' + subscript_num[k+1] + subscript_num[l+1] + '\n')
                print(l_outer, '\n')
                ret += coeff*outer
    
        return ret
                
    def computeWeightMatrix(self, _T, _indxBasis_1, _indxBasis_2, _n):
        
        
        _Basis_1 = self._vars._vars[_indxBasis_1].value
        symbol_1 = self._vars._vars[_indxBasis_1].symbol
        _Basis_2 = self._vars._vars[_indxBasis_2].value
        symbol_2 = self._vars._vars[_indxBasis_2].symbol
        
       
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
                result = self.computeWeightElement(TW, i, j, b_1, symbol_1, b_2, symbol_2, _n)
                ret.append(result)
        
        return np.array(ret)
      
    def allocateFourthOrderBasis(self, _n):
        ret=np.array([[[[[[0.0]*_n]*_n]*_n]*_n]*_n]*_n)
        return ret
    def allocateFourthOrderElement(self, _n):
        ret_ij = np.array([[[[0.0]*_n]*_n]*_n]*_n)
        return ret_ij
    
    def computeTensorOuterProduct(self, _T, _indxBasis_1, _indxBasis_2, _indxBasis_3, _indxBasis_4, _n):
    
        subscript_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']
            
        _Basis_1 = self._vars._vars[_indxBasis_1].value
        symbol_1 = self._vars._vars[_indxBasis_1].symbol
        _Basis_2 = self._vars._vars[_indxBasis_2].value
        symbol_2 = self._vars._vars[_indxBasis_2].symbol
        _Basis_3 = self._vars._vars[_indxBasis_3].value
        symbol_3 = self._vars._vars[_indxBasis_3].symbol
        _Basis_4 = self._vars._vars[_indxBasis_4].value
        symbol_4 = self._vars._vars[_indxBasis_4].symbol
        

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
        
        
        result = 0
    
        for i in range(_n):
            for j in range(_n):
                #result = 0
                for k in range(_n):
                    for l in range(_n):
                        coeff = _T[i, j, k, l]
                        print('T' + subscript_num[i+1] + subscript_num[j+1] + subscript_num[k+1] + subscript_num[l+1] + ' = ', coeff)
                        self.printVector(b_1[i], False, symbol_1.lower() + subscript_num[i+1], _n)
                        self.printVector(b_2[j], False, symbol_2.lower() + subscript_num[j+1], _n) 
                        self.printVector(b_3[k], False, symbol_3.lower() + subscript_num[k+1], _n)
                        self.printVector(b_4[l], False, symbol_4.lower() + subscript_num[l+1], _n)
                        outer = np.einsum('i,j,k,l',b_1[i], b_2[j], b_3[k],b_4[l])
                        l_outer = self.convertElementToLatex(outer, _n)
                        print('outer' + subscript_num[i+1] + subscript_num[j+1] + subscript_num[k+1] + subscript_num[l+1],'\n')
                        print(l_outer, '\n')
                        
                        
                        result += coeff*outer
            
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
    
    def printMatrix(self, _M, _label, _n):
         l_result = self._latex.convertMatrixToLatex(_M, 2)
         print(_label + ' = ', l_result, '\n')
     
    def printVector(self, _vec, _transpose, _label, _n):
         l_result = self._latex.converVectorToLatex(_vec, _transpose, 2)
         print(_label + ' = ', l_result, '\n')

    def convertElementToLatex(self, _elem, _n):
        
        ret = '\\bmatrix{'     
        
        for i in range(_n):
            for j in range(_n):
                tmp = self._latex.convertMatrixToLatex(_elem[i][j], _n)
                ret += tmp
                if j != _n -1:
                    ret += '&'
            if i != _n - 1:            
                ret += '\\\\' 
        
        ret += '\\\\}' 
        return ret

    def printWeightMatricesToLatex(self, _T, _n):
        
        subscript_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']
        
        for i in range(_n):
            for j in range(_n):
                index = i*_n + j
                self.printMatrix(_T[index], 'T' + subscript_num[i+1] + subscript_num[j+1], _n)

    def convertResultsToLatex(self, _result, _weights, _n):
        
        ret = []
        for i in range(_n):
            for j in range(_n):
                matrixLst = []
                for k in range(_n):
                    for l in range(_n):
                        matrixLst.append(_result[i][j][k][l])
                    
                if type(_weights) == np.ndarray:
                    index = i*_n + j
                    weights = _weights[index]
                else:
                    weights = 0
                #ret.append(self.createLatexBlockMatrix(matrixLst, weights, _n))
        
        return ret
    
    
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
    
class dictElement:
    
    value = []
    symbol = ''
    lsymbol = ''
    
    
    
class variables:

    def __init__(self, _r, _theta):
        coords = coordinateTransforms()
        self._vars = {}
        tmpElement = dictElement()
       
        
        tmpElement.symbol = 'E'
        tmpElement.value = coords.polarVectorBasis(_r, _theta)
        self._vars['E'] = copy.deepcopy(tmpElement)
        tmpElement.symbol = 'Eᵀ'
        tmpElement.value =  np.transpose(self._vars['E'].value)
        self._vars['ET'] = copy.deepcopy(tmpElement)
        tmpElement.symbol ='W'
        tmpElement.value = coords.polarOneFormBasis(_r, _theta)
        self._vars['W'] = copy.deepcopy(tmpElement)
        tmpElement.symbol ='Wᵀ'
        tmpElement.value = np.transpose(self._vars['W'].value)
        self._vars['WT'] = copy.deepcopy(tmpElement)
        

        r1 = _r**2  
        theta1 = _theta**2
        tmpElement.symbol = 'A'
        tmpElement.value = coords.polarSqrtAMatrix(r1, theta1)
        self._vars['A'] = copy.deepcopy(tmpElement)
        tmpElement.symbol = 'Aᵀ'
        tmpElement.value = np.transpose(self._vars['A'].value) 
        self._vars['AT'] = copy.deepcopy(tmpElement)
        tmpElement.symbol = 'B'
        tmpElement.value = coords.polarSqrtBMatrix(r1, theta1)
        self._vars['B'] = copy.deepcopy(tmpElement)
        tmpElement.symbol = 'Bᵀ'
        tmpElement.value = np.transpose(self._vars['B'].value)
        self._vars['BT'] = copy.deepcopy(tmpElement)
        tmpElement.symbol = 'E̅'
        tmpElement.value = np.dot(self._vars['A'].value,self._vars['E'].value)
        self._vars['E1'] = copy.deepcopy(tmpElement)
        tmpElement.symbol = 'E̅ᵀ'
        tmpElement.value = np.transpose(self._vars['E1'].value)
        self._vars['E1T'] = copy.deepcopy(tmpElement)
        tmpElement.symbol = 'W̅'
        tmpElement.value = np.dot(np.transpose(self._vars['B'].value), self._vars['W'].value)
        self._vars['W1'] = copy.deepcopy(tmpElement)
        tmpElement.symbol = 'W̅ᵀ'
        tmpElement.value = np.transpose(self._vars['W1'].value)
        self._vars['W1T'] = copy.deepcopy(tmpElement)
        self._latex = convertToLatex()
        

    def getVars(self):
        return self._vars

    def printVars(self, _lst, _n):
        for item in _lst:
            l_result = self._latex.convertMatrixToLatex(self._vars[item], _n)
            print(item + '\n')
            print(l_result)
            print('\n')







