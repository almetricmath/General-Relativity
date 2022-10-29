# -*- coding: utf-8 -*-
"""
Created on Tue Sep  6 15:13:10 2022

@author: Al
"""

# This program tests tensor various transforms

# second order contravariant tensor

import numpy as np
import copy
from enum import Enum

class coordinateTransforms:
    
    def __init__(self):
        self._transform = {}

        

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
        self._posNum = posNum()
    
    def computeWeightElement(self, _T, _i, _j,  _posLst, _basisLst, _symbolLst, _n):
        
        ret = 0
        
        for k in range(_n):
            for l in range(_n):
                coeff = _T[k,l]
                print('T' + self._posNum.indice(_posLst[0], _i) + self._posNum.indice(_posLst[1], _j) + self._posNum.indice(_posLst[2], k) + self._posNum.indice(_posLst[3], l) + ' = ', coeff)
                self.printVector(_basisLst[2][k], False, _symbolLst[2].lower() + self._posNum.indice(_posLst[2], k), _n)
                self.printVector(_basisLst[3][l], False, _symbolLst[3].lower() + self._posNum.indice(_posLst[3], l), _n) 
                outer = np.einsum('k,l',_basisLst[2][k], _basisLst[3][l])
                l_outer = self._latex.convertMatrixToLatex(outer, _n)
                print('outer' + self._posNum.indice(_posLst[2], k) + self._posNum.indice(_posLst[3], l) + '\n')
                print(l_outer, '\n')
                ret += coeff*outer
    
        return ret
                
    def computeWeightMatrix(self, _T, _posLst, _basisLst, _symbolLst, _n):
        

        ret = self.allocateFourthOrderElement(_n)
        
        for i in range(_n):
            for j in range(_n):
                TW = _T[i,j]
                result = self.computeWeightElement(TW, i, j, _posLst, _basisLst, _symbolLst, _n)
                ret[i][j] = result
        
        return ret
    
    def computeWeightMatrix1(self, _T, _posLst, _basisLst, _symbolLst, _n):
        
        # computes weight matrix using matrix operations B3^T(T)B4
        
        l_B3T = self._latex.convertMatrixToLatex(np.transpose(_basisLst[2]), _n)
        print(_symbolLst[2] + 'ᵀ' + ' = ', l_B3T, '\n')
      
        l_B4 = self._latex.convertMatrixToLatex(_basisLst[3], _n)
        print(_symbolLst[3] + ' = ', l_B4, '\n')
       
        
        ret = self.allocateFourthOrderElement(_n)
        for i in range(_n):
            for j in range(_n):
                TW = _T[i,j]
                l_t_matrix = self._latex.convertMatrixToLatex(TW, _n)
                print('T' + self._posNum.indice(_posLst[0], i, False) + self._posNum.indice(_posLst[1], j, False) + self._posNum.indice(_posLst[2], 2, True) + self._posNum.indice(_posLst[3], 3, True), l_t_matrix,'\n')
                tmp = np.dot(TW, _basisLst[3])
                result = np.dot(np.transpose(_basisLst[2]), tmp)
                l_result = self._latex.convertMatrixToLatex(result, _n)
                print('T' +self._posNum.indice(_posLst[0], i, False) + self._posNum.indice(_posLst[1], j, False), l_result,'\n') 
                ret[i][j] = result
            
        return ret
          
      
    def allocateFourthOrderBasis(self, _n):
        ret=np.array([[[[[[0.0]*_n]*_n]*_n]*_n]*_n]*_n)
        return ret
    def allocateFourthOrderElement(self, _n):
        ret_ij = np.array([[[[0.0]*_n]*_n]*_n]*_n)
        return ret_ij
    
    
    def getBasisIndex(self, _pos, _unprimed):
        
        ret = ''
        suffix = ''
        
        if not _unprimed:
            suffix = '1'
            
        if _pos == pos.up:
            ret = 'E' + suffix
        elif _pos == pos.down:
            ret = 'W' + suffix
        else:
            print('Error - position needs to be specified')
            ret = None
        
        return ret
    
    def getTransformIndex(self, _pos):
        
        if _pos == pos.up:
            ret = 'B'
        elif _pos == pos.down:
            ret = 'A'
        else:
            print('Error - position needs to be specified')
            ret = None
        return ret
       
    def processTensorInput(self, _posLst, _col, _unprimed, _n):
        
        _basisLst = [0]*4
        _symbolLst = [0]*4
        # set up matrix of vectors
        
        col_1 = [np.array([])]*_n
        col_2 = [np.array([])]*_n
       
        
        indx = 0
        for p in _posLst:
            bIndx = self.getBasisIndex(p, _unprimed)
            if bIndx == None:
                return None
            _basisLst[indx] = self._vars._vars[bIndx].value
            m_latex = self._latex.convertMatrixToLatex(_basisLst[indx], _n)
            _symbolLst[indx] = self._vars._vars[bIndx].symbol
            print(_symbolLst[indx] + ' = ', m_latex)
            indx += 1
        
        if _col:
        
           
            # get the column vectors from each Basis Matrix
             
            for j in range(_n):
               tmp = []
               for i in range(_n):
                   tmp.append(_basisLst[0][i,j])
               col_1[j] = np.array(tmp)

            for j in range(_n):
               tmp = []
               for i in range(_n):
                   tmp.append(_basisLst[1][i,j])
               col_2[j] = np.array(tmp)
            
        ret = _basisLst, _symbolLst, col_1, col_2
    
    
        return ret
          
      
    def computeTensorOuterProduct(self, _T, _posLst, _unprimed, _n):
     
        _basisLst, _symbolLst, col_1, col_2 = self.processTensorInput(_posLst, False, _unprimed, _n)
        
        ret = 0
    
        for i in range(_n):
            for j in range(_n):
                for k in range(_n):
                    for l in range(_n):
                        coeff = _T[i, j, k, l]
                        print('T' + self._posNum.indice(_posLst[0], i)  +  self._posNum.indice(_posLst[1], j) + self._posNum.indice(_posLst[2], k) + self._posNum.indice(_posLst[3], l)  + ' = ', coeff)
                        self.printVector( _basisLst[0][i], False, _symbolLst[0].lower() + self._posNum.indice(_posLst[0], i), _n)
                        self.printVector( _basisLst[1][j], False, _symbolLst[1].lower() + self._posNum.indice(_posLst[1], j), _n) 
                        self.printVector( _basisLst[2][k], False, _symbolLst[2].lower() + self._posNum.indice(_posLst[2], k), _n)
                        self.printVector( _basisLst[3][l], False, _symbolLst[3].lower() + self._posNum.indice(_posLst[3], l), _n)
                        outer = np.einsum('i,j,k,l',_basisLst[0][i], _basisLst[1][j], _basisLst[2][k],_basisLst[3][l])
                        l_outer = self.convertElementToLatex(outer, _n)
                        print('outer' + self._posNum.indice(_posLst[0], i) + self._posNum.indice(_posLst[1], j) + self._posNum.indice(_posLst[2], k) + self._posNum.indice(_posLst[3], l),'\n')
                        print(l_outer, '\n')
                             
                        ret += coeff*outer
            
        return ret
    
    def computeTensorInnerProduct(self, _T, _posLst, _unprimed, _n):
       
         # implements streamlined calculation
         # compute weight matrices
             
         ret = self.allocateFourthOrderElement(_n)
         
         _basisLst, _symbolLst, col_1, col_2 = self.processTensorInput(_posLst, True, _unprimed, _n)
         
         
        # get weights for the T_ij matrix and compute submatrix
        
         print('Compute Tensor Inner Product\n')
         
         #print('Weight Matrix Computed with Outer Products\n')
         
         #T_ij = self.computeWeightMatrix(_T, _posLst, _basisLst, _symbolLst, _n)
         #self.printWeightMatrices(T_ij, _posLst, _symbolLst, _n)
        
         print('Weight Matrix Computed with Matrix Product\n')
        
         T_ij1 = self.computeWeightMatrix1(_T, _posLst, _basisLst, _symbolLst, _n)
         
        
         for i in range(_n):
             for j in range(_n):
                 # output latex for variables
                self.printVector(col_1[i], False, 'col(' + _symbolLst[0] + ', ' + str(i+1) + ')ᵀ', _n)
                l_col = self._latex.convertVectorToLatex(col_2[j], True, 2)
                print('col(' + _symbolLst[1] + ', ' + str(j+1) + ') = ', l_col )
                tmp = self.blockMatrixVectorMult(T_ij1,col_2[j], _n)
                l_tmp_0 = self._latex.convertMatrixToLatex(tmp[0], 2)
                l_tmp_1 = self._latex.convertMatrixToLatex(tmp[1], 2)
                l_tmp = '\\bmatrix{' + l_tmp_0 + '\\\\' + l_tmp_1 + '\\\\}'
                print('intermediate results = ',l_tmp, '\n')
                ret[i][j] = self.vectorTransposeBlockVectorMult(tmp, col_1[i], 2) 
        
         return ret
        

    def processTransformInput(self, _posLst, _unprimed, _n):
        
        _transformLst = [0]*4
        _transformSymbolLst = [0]*4
      
        indx = 0
        for p in _posLst:
            bIndx = self.getTransformIndex(p)
            if bIndx == None:
                return None
            _transformLst[indx] = self._vars._vars[bIndx].value
            _transformSymbolLst[indx] = self._vars._vars[bIndx].symbol
            indx += 1
       
        return _transformLst, _transformSymbolLst
      
        
    # transform tensor
    
    def transformTensor(self, _T, _posLst, _unprimed, _n):
        
        # 1st work unprimed to primed system
       
        _basisLst, _symbolLst, col_1, col_2 = self.processTensorInput(_posLst, False, True, _n)
       
        T_ij = self.computeWeightMatrix(_T, _posLst, _basisLst, _symbolLst, _n)
        self.printWeightMatrices(T_ij, _n)
        
        _transformLst, _transformSymbolLst = self.processTransformInput(_posLst, True, _n)
        
        # allocate return structure
        
        T1_ij = self.allocateFourthOrderElement(_n)
        symbol_3 = _transformSymbolLst[2]
        symbol_4 = _transformSymbolLst[3]
        
        for i in range(_n):
            for j in range(_n):
                acc = 0
                for k in range(_n):
                    for l in range(_n):
                        acc += _transformLst[2][l][i]*T_ij[l][k]*_transformLst[3][k][j]
                T1_ij[i][j] = acc
                
        # could transform T1_ij -> T1_ijkl
        
        # T1_ij in the primed system
        # need to do transform with primed coordinates
        # get inverse bases in the primed system
        # flip up/down values in the _posLst
        
        inversePosLst = []
        for p in _posLst:
            tmp = not p
            inversePosLst.append(tmp)
        
        _inverseBasisLst, _inverseSymbolLst, col_1, col_2 = self.processTensorInput(inversePosLst, False, False, _n)
        
        ret = self.allocateFourthOrderElement(_n)
        
        for i in range(_n):
            for j in range(_n):
                tmp = np.dot(T1_ij[i][j],_inverseBasisLst[1])
                ret[i][j] = np.dot(np.transpose(_inverseBasisLst[0]), tmp)
            
        return ret
       
        
    
    # multiply a block matrix with a vector
    
    def blockMatrixVectorMult(self, _T_ij, _vec, _n):
        
        ret = []
        tmp =0
        
        for i in range(_n):
            tmp = 0
            for j in range(_n):
                tmp += _T_ij[i][j]*_vec[j]
            ret.append(tmp)
           
        return ret
       
    # multiply a transposed vector with a block matrix
    
    def vectorTransposeBlockVectorMult(self, _b_vec, _vec, _n):
        
        ret = 0
    
        for i in range(_n):
            ret += _vec[i]*_b_vec[i]
           
        return ret
       
    # multiply a block matrix by a matrix
    
    def blockMatrixMatrixMult(self, _T, _M, _n):
        
        # _T is a list of matrices
        # multiply each element of _T y _M[i,j]
        
        ret = 0
        
        for i in range(_n):
            for j in range(_n):
                index = i*_n + j
                ret += _T[index]
                
            
    
    
    def computeTensorElement(self, _E_ixj, _T_ij, _n):
        
        ret = self.allocateFourthOrderElement(_n)
        
        for i in range(_n):
            for j in range(_n):
                ret[i][j] = _E_ixj[i,j]*_T_ij
    
        return ret
        
    def getBasisVector(self, _indxBasis, _i):
        
        subscript_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']
        
        _Basis = self._vars._vars[_indxBasis].value
        symbol = self._vars._vars[_indxBasis].symbol
        ret = _Basis[_i,:], symbol.lower() + subscript_num[_i+1]
        return ret
        
    
    def printMatrix(self, _M, _label, _n):
         l_result = self._latex.convertMatrixToLatex(_M, 2)
         print(_label + ' = ', l_result, '\n')
     
    def printVector(self, _vec, _transpose, _label, _n):
         l_result = self._latex.convertVectorToLatex(_vec, _transpose, 2)
         if _transpose:
             _label += 'ᵀ = '
         else:
             _label += ' = ' 
         print(_label, l_result, '\n')

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

    def printWeightMatrices(self, _T, _posLst, _symbolLst, _n):
        
        
        for i in range(_n):
            for j in range(_n):
                self.printMatrix(_T[i][j], 'T' + self._posNum.indice(_posLst[2], i) + self._posNum.indice(_posLst[3], j), _n)
 
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
    
    def convertVectorToLatex(self, _vec, transposeFlag, _n):
    
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
  
class pos(Enum):
    init = None
    up = 0
    down = 1
    
class posNum:
    
    def __init__(self):
        self._sub_num = ['₀','₁','₂','₃','₄','₅','₆','₇','₈', '₉']
        self._sup_num = ['⁰','¹','²','³','⁴','⁵','⁶','⁷','⁸','⁹']
        self._sup_let = ['ᶦ','ʲ','ᵏ','ˡ']
        self._sub_let = ['ᵢ','ⱼ','ₖ','ₗ']
    
    def indice(self, _pos, _index, _letterFlag):
    
        if _pos == pos.up:
            if not _letterFlag:
                return self._sup_num[_index + 1]
            else:
                return self._sup_let[_index]
                
        elif _pos == pos.down:
            if not _letterFlag:
                return self._sub_num[_index + 1] 
            else:
                return self._sub_let[_index]
        else:
           return None
        
       
class dictElement:
    
    value = []
    symbol = ''

class variables:

    def __init__(self, _r, _theta):    

        self._vars = {}
        tmpElement = dictElement()
        coords = coordinateTransforms()
        
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

class utils:
    
    # utility functions for showing partial results
    
    def allocateFourthOrderElement(self, _n):
        ret_ij = np.array([[[[0.0]*_n]*_n]*_n]*_n)
        return ret_ij
   
    
    def outer(self, _v1, _v2):
       ret = np.einsum('i,j', _v1, _v2)  
       return ret

    # outer product of two matrices

    def m_outer(self, _m1, _m2, _n):
        ret = self.allocateFourthOrderElement(_n)
        for i in range(_n):
            for j in range(_n):
                ret[i][j] = _m1[i][j]*_m2
        
        return ret






