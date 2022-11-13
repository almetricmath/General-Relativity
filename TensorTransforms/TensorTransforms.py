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
        self._posNum = posNum()
        self._utils = utils()
        
        # declare configuration tables in unprimed system
        
        self._forwardTable = self._utils.computeTable(self._utils.forwardTable, True, self)
        self._transformTable = self._utils.computeTable(self._utils.transformTable, True, self)
        
        # declare configuration table in primed system
        
        self._forwardTablePrimed = self._utils.computeTable(self._utils.forwardTable, False, self)
       
    def computeTensorOuterProduct(self, _T, _posLst, _unprimed, _n):
        
        # get bases as vectors
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, True, _n)
     
        ret = 0
        for i in range(_n):
            for j in range(_n):
                coeff = _T[i, j]
                print('T' + self._posNum.indice(_posLst[0], i, False)  +  self._posNum.indice(_posLst[1], j, False), coeff)
                self.printVector( _basisLst[0][i], False, _symbolLst[0].lower() + self._posNum.indice(_posLst[0], i, False), _n)
                self.printVector( _basisLst[1][j], False, _symbolLst[1].lower() + self._posNum.indice(_posLst[1], j, False), _n)
                outer = np.einsum('i,j',_basisLst[0][i], _basisLst[1][j])
                self.printMatrix(outer,'outer' + self._posNum.indice(_posLst[0],i, False) + self._posNum.indice(_posLst[1],j, False), _n)
                ret += coeff*outer
        return ret
    
    def computeTensorInnerProduct(self, _T, _posLst, _unprimed, _class, _n):
        
        # get bases as matrices
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, False, _n)
        #basesLst = self._utils.getBases(_posLst, _unprimed, _class)
        
        
        l_result = self._latex.convertMatrixToLatex(_T, _n)
        print('T' + '\n', l_result, '\n')
        l_result = self._latex.convertMatrixToLatex(np.transpose(_basisLst[0]._transposeBasis), _n)
        print(_symbolLst[0], l_result, '\n')
        l_result = self._latex.convertMatrixToLatex(_basisLst[0]._basis, _n)
        print(_symbolLst[1], l_result, '\n')
        
       
        tmp = np.dot(np.transpose(_basisLst[0]._transposeBasis), _T)
        ret = np.dot(tmp, _basisLst[0]._basis)
        return ret

    def printMatrix(self, _M, _label, _n):
        l_result = self._latex.convertMatrixToLatex(_M, _n)
        print(_label + ' = ', l_result, '\n')
        
    def printVector(self, _vec, _transpose, _label, _n):
         l_result = self._latex.convertVectorToLatex(_vec, _transpose, 2)
         if _transpose:
             _label += 'ᵀ = '
         else:
             _label += ' = ' 
         print(_label, l_result, '\n')
    
 
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
        self._utils = utils()
        
        # declare configuration tables in unprimed system
        
        self._forwardTable = self._utils.computeTable(self._utils.forwardTable, True, self)
        self._transformTable = self._utils.computeTable(self._utils.transformTable, True, self)
        
        # declare configuration table in primed system
        
        self._forwardTablePrimed = self._utils.computeTable(self._utils.forwardTable, False, self)
        
    
    def printComputeElement(self, _elem, _n):
          l_transposeBasis = self._latex.convertMatrixToLatex(np.transpose(_elem._transposeBasis), _n)
          print(_elem._transposeSymbol  + ' = ' + l_transposeBasis, "\n")
          l_basis = self._latex.convertMatrixToLatex(_elem._basis, _n)
          print(_elem._symbol + ' = ' + l_basis, "\n")
          return 


    def computeWeightMatrix(self, _T, _posLst, _basisLst, _unprimed, _n):
        
        # computes weight matrix using matrix operations transpose(B3).T.B4
        # _tuple = (k = (up/down), l = (up/down))
        
        # get Basis 3 and Basis 4 in matrix form
        
        B3 = _basisLst[1]._transposeBasis
        B4 = _basisLst[1]._basis
        
        # print Basis 3 and Basis 4
        
        self.printComputeElement(_basisLst[1], _n)
        
        ret = self.allocateFourthOrderElement(_n)
        
        for i in range(_n):
            for j in range(_n):
                TW = _T[i,j]
                l_t_matrix = self._latex.convertMatrixToLatex(TW, _n)
                print('T' + self._posNum.indice(_posLst[0], i, False) + self._posNum.indice(_posLst[1], j, False) + self._posNum.indice(_posLst[2], 2, True) + self._posNum.indice(_posLst[3], 3, True), l_t_matrix,'\n')
                result = self._utils.matrix_1T_TW_matrix_2(B3, TW, B4, False, _n)
                l_result = self._latex.convertMatrixToLatex(result, _n)
                print('T' +self._posNum.indice(_posLst[0], i, False) + self._posNum.indice(_posLst[1], j, False), l_result,'\n') 
                ret[i][j] = result
            
        return ret

    def getColumns(self, _basis, _n):
        
        # get the column vectors from each Basis Matrix
        
        ret = [np.array([])]*_n 
        
        for j in range(_n):
           tmp = []
           for i in range(_n):
               tmp.append(_basis[i,j])
           ret[j] = np.array(tmp)

        return ret
       
    def colT_T_col(self, _col_1, _T, _col_2, _n):
        
        # compute transpose(col_1).T_ij.col_2
        
        ret = 0.0
        
        for k in range(_n):
            for l in range(_n):
                ret += _col_1[k]*_T[k,l]*_col_2[l]
        
        return ret
                
    
    def computeTensorInnerProduct(self, _T, _posLst, _unprimed, _n):
       
         # implements streamlined calculation
         # compute weight matrices
             
         ret = self.allocateFourthOrderElement(_n)
         
         print('Compute Tensor Inner Product\n')
         
         print('Weight Matrix Computed with Matrix Product\n')
         
        # compute weight matrix in the unprimed system
        
         _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, False, _n)
       
        
         if _unprimed: 
             T_ij = self.computeWeightMatrix(_T, _posLst, _basisLst, True, _n)
         else:
             # use weight that has been transformed to a primed coordinate system
             T_ij = _T
         
         # _tuple = (k = (up/down), l = (up/down))
         
         # get Basis 1 and Basis 2
         
         B1 = _basisLst[0]._transposeBasis
         B2 = _basisLst[0]._basis
        
         # print Basis 1 and Basis 2
         
         self.printComputeElement(_basisLst[0], _n)
         
         # get columns from B1 and B2
         
         colB1 = self.getColumns(B1, 2)
         colB2 = self.getColumns(B2, 2)
         
         # compute transpose(col(B1, i)).T(i,j).col(B2, j)
         
         for i in range(_n):
             for j in range(_n):
                 ret[i,j] = self.colT_T_col(colB1[i], T_ij, colB2[j], _n)
                 
         return ret
         
           
    def allocateFourthOrderElement(self, _n):
        
        ret_ij = np.array([[[[0.0]*_n]*_n]*_n]*_n)
        return ret_ij
             
    def computeTensorOuterProduct(self, _T, _posLst, _unprimed, _n):
     
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, True, _n)
        
        ret = 0
    
        for i in range(_n):
            for j in range(_n):
                for k in range(_n):
                    for l in range(_n):
                        coeff = _T[i, j, k, l]
                        print('T' + self._posNum.indice(_posLst[0], i, False)  +  self._posNum.indice(_posLst[1], j, False) + self._posNum.indice(_posLst[2], k, False) + self._posNum.indice(_posLst[3], l, False)  + ' = ', coeff)
                        self.printVector( _basisLst[0][i], False, _symbolLst[0].lower() + self._posNum.indice(_posLst[0], i, False), _n)
                        self.printVector( _basisLst[1][j], False, _symbolLst[1].lower() + self._posNum.indice(_posLst[1], j, False), _n) 
                        self.printVector( _basisLst[2][k], False, _symbolLst[2].lower() + self._posNum.indice(_posLst[2], k, False), _n)
                        self.printVector( _basisLst[3][l], False, _symbolLst[3].lower() + self._posNum.indice(_posLst[3], l, False), _n)
                        outer = np.einsum('i,j,k,l',_basisLst[0][i], _basisLst[1][j], _basisLst[2][k],_basisLst[3][l])
                        l_outer = self.convertElementToLatex(outer, _n)
                        print('outer' + self._posNum.indice(_posLst[0], i, False) + self._posNum.indice(_posLst[1], j, False) + self._posNum.indice(_posLst[2], k, False) + self._posNum.indice(_posLst[3], l, False),'\n')
                        print(l_outer, '\n')
                             
                        ret += coeff*outer
            
        return ret
    
        
    def processTransformInput(self, _posLst, _unprimed, _n):
        
        ret = computeElement()
        
        _tuple = (_posLst[0], _posLst[1])
        ret = self._transformTable[_tuple]
        
        return ret
        
    # transform tensor
    
    def transformTensor(self, _T, _posLst, _unprimed, _n):
        
        # 1st work unprimed to primed system
        # compute tensor coordinate change using matrices
       
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, True, self, False, _n)
       
        # Compute weight matrix in unprimed system
        
        T_ij = self.computeWeightMatrix(_T, _posLst, _basisLst, True, _n)
      
        self.printWeightMatrices(T_ij, _posLst, _n)
        
        # get transforms involved with coordinate change
        
        _transform = self.processTransformInput(_posLst, True, _n)
        
        # allocate return structure
        
        T1_ij = self.allocateFourthOrderElement(_n)
        symbol_1 = _transform._transposeSymbol
        symbol_2 = _transform._symbol
        
        T1_ij = self._utils.matrix_1T_TW_matrix_2(_transform._transposeBasis, T_ij, _transform._basis, True, _n)
         
        '''          
        # could transform T1_ij -> T1_ijkl
        
        # T1_ij in the primed system
        # need to do transform with primed coordinates
        # get inverse bases in the primed system
        # flip up/down values in the _posLst
        
        inversePosLst = []
        for p in _posLst:
            if p == pos.up:
                tmp = pos.down
            elif p == pos.down:
                tmp = pos.up
            inversePosLst.append(tmp)
        
        _inverseBasisLst, _inverseSymbolLst, col_1, col_2 = self.processTensorInput(inversePosLst, False, False, _n)
        
        ret = self.allocateFourthOrderElement(_n)
        W1 = self._vars._vars['W1'].value
        E1 = self._vars._vars['E1'].value
      
        
        for i in range(_n):
            for j in range(_n):
                tmp = np.dot(T1_ij[i][j],np.transpose(W1))
                ret[i][j] = np.dot(E1, tmp)
        '''    
        return T1_ij
       
        
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

    def printWeightMatrices(self, _T, _posLst, _n):
    
        for i in range(_n):
            for j in range(_n):
                self.printMatrix(_T[i][j], 'T' + self._posNum.indice(_posLst[2], i, False) + self._posNum.indice(_posLst[3], j, False), _n)
 
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
    
        # tensor coordinate down -> basis coordinate up 
        # and visa versa
    
        if _pos == pos.up:
            if not _letterFlag:
                return self._sub_num[_index + 1]
            else:
                return self._sub_let[_index]
                
        elif _pos == pos.down:
            if not _letterFlag:
                return self._sup_num[_index + 1] 
            else:
                return self._sup_let[_index]
        else:
           return None
        
       
class dictElement:
    
    value = []
    symbol = ''

class transformElement:
    
    # This class holds the transforms corresponding to a table of transforms
    # for use with an equation of the form M1^T(T)M2
    # there is no need for an explicit transpose in the computation because the transpose 
    # is taken into effect by revering the indices
    # so for this reason, the transpose attribute holds the untransposed matrix
    
    def __init__(self):
        self._transposeIndx = ''
        self._transposeSymbol = ''
        self._indx = ''
        self._symbol = ''

class computeElement:
    
    def __init__(self):
        self._transposeBasis = dictElement()
        self._basis = dictElement()
    
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
    
    def matrix_1T_TW_matrix_2(self, _matrix_1, _TW, _matrix_2, _block, _n):
        
        # computes transpose(_matrix_1)._TW._matrix_2
        # _block specifies _TW as a block matrix and the return type as a block matrix
        
        if not _block:
            ret = np.zeros(_n*_n).reshape((_n, _n))
        else:
            ret = self.allocateFourthOrderElement(_n)
      
        for i in range(_n):
            for j in range(_n):
                acc = 0.0
                for k in range(_n):
                    for l in range(_n):
                        acc += _matrix_1[k][i]*_TW[k][l]*_matrix_2[l][j] 
                ret[i,j] = acc
                 
        return ret
       
    
    
    def computeTable(self, tableFunc, _unprimed, _class):
    
        # mapping to transform T(i,j,k,l) -> T(i,j) 
        # and mapping to transform T(i,j) -> Full Tensor
        
        ret = {}
        
        indx = [(pos.up,pos.up),(pos.up, pos.down), (pos.down, pos.up), (pos.down, pos.down)]
         
        for i in indx:
            ret[i] = tableFunc(i, _unprimed, _class)  
            
        return ret
    
    def getRows(self, _basis, _n):
        
        ret = []
        tmp_vec = []
        
        tmp = _basis._transposeBasis
        for i in range(_n):
            tmp_vec.append(tmp[i])
        
        ret.append(tmp_vec)
        
        tmp_vec = []
        tmp = _basis._basis
        for i in range(_n):
            tmp_vec.append(tmp[i])
        
        ret.append(tmp_vec)
        
        return ret
    
    def getSymbols(self, _basis, _n):
        
        ret = []
        
        tmp = _basis._transposeSymbol
        ret.append(tmp)
        tmp = _basis._symbol
        ret.append(tmp)
        
        return ret
    
    def processTensorInput(self, _posLst, _unprimed, _class, _vecFlag, _n):
               
        # if _vecFlag get bases in vector form 
        # otherwise return bases in matrix form        
        
        bases = []
        
        basesLst = self.getBases(_posLst, _unprimed, _class)
        
        if _vecFlag:
            for i in basesLst:
                bases += self.getRows(i,_n)
        else:
            bases = basesLst
        
        # get symbols in vector form
        
        symbol_vecs = []
        
        for i in basesLst:
            tmp = self.getSymbols(i, _n)
            symbol_vecs += tmp
            
        return bases, symbol_vecs
   
    
    def getBases(self, _posLst, _unprimed, _class):
    
        basesLst = []
        n = len(_posLst)
        
        for i in range(0, n, 2):
            _tuple = (_posLst[i], _posLst[i+1])
            if _unprimed:
                tmp = _class._forwardTable[_tuple]
            else:
                tmp = _class._forwardTablePrimed[_tuple]
            
            basesLst.append(tmp)
            
        return basesLst
    
    def forwardTable(self, _tuple, _unprimed, _class):
        
        # function to implement Table 2 and Table 5 from the writeup
        # note there is no need to compute the transposes, they are not needed 
        # the way the algorithms are written, but need to be kept track of
        
        ret = computeElement()
        suffix = ''
        
        if not _unprimed:  # prime coordinate system
            suffix = '1'
        
        if _tuple[0] == pos.up:
            transposeIndx = 'E' + suffix
        else:
            transposeIndx = 'W' + suffix
        
                
        if _tuple[1] == pos.up:
            indx = 'E' + suffix
        else:
            indx = 'W' + suffix

        ret._transposeBasis = _class._vars._vars[transposeIndx].value
        ret._transposeSymbol = _class._vars._vars[transposeIndx].symbol
        ret._basis = _class._vars._vars[indx].value
        ret._symbol = _class._vars._vars[indx].symbol
        
        return ret
   
    
    def transformTable(self, _tuple, _unprimed, _class):
        
        ret = computeElement()
        
        if _tuple[0] == pos.up:
            transposeIndx = 'B'
        else:
            transposeIndx = 'A'
            
        if _tuple[1] == pos.up:
            indx = 'B'
        else:
            indx = 'AT'
            
        ret._transposeBasis = _class._vars._vars[transposeIndx].value
        ret._transposeSymbol = _class._vars._vars[transposeIndx].symbol
        ret._basis = _class._vars._vars[indx].value
        ret._symbol = _class._vars._vars[indx].symbol
        
        return ret

     