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
       
    def computeTensorOuterProduct(self, _T, _posLst, _unprimed, _n, _verbose):
        
        # get bases as vectors
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, True, _n)
     
        ret = 0
        eqstr = ''
        
        for i in range(_n):
            for j in range(_n):
                coeff = _T[i, j]
                if _verbose:
                    if eqstr != '':
                        eqstr += '+' + '(' + str("{:.6f}".format(coeff)) + ')'
                    else:
                        eqstr += '(' + str(coeff) + ')' 
                outer = np.einsum('i,j',_basisLst[0][i], _basisLst[1][j])
                l_outer = self._latex.convertMatrixToLatex(outer, _n)
                eqstr += l_outer
                ret += coeff*outer
        
        if _verbose:
            print(eqstr,'\n')        
        return ret
    
    def computeTensorInnerProduct(self, _T, _posLst, _unprimed, _n, _verbose):
        
        # get bases as matrices
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, False, _n)
        
        if _verbose:
            l_result = self._latex.convertMatrixToLatex(_T, _n)
            if _unprimed:
                T_symbol = 'T'
            else:
                T_symbol = 'T̅'
            print(T_symbol + '\n', l_result, '\n')
            l_result = self._latex.convertMatrixToLatex(_basisLst[0], _n)
            print('F = ',_symbolLst[0], l_result, '\n')
            l_result = self._latex.convertMatrixToLatex(_basisLst[1], _n)
            print('H = ', _symbolLst[1], l_result, '\n')
             
        ret = self._utils.matrix_1T_TW_matrix_2(_basisLst[0], _T, _basisLst[1], False, _n, _verbose)
        
        
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
    
    def transformTensor(self, _T, _posLst, _n, _verbose):
        
        transformLst = self._utils.processTransformInput(_posLst, self, _n)
        M1 = transformLst[0]._basis.value
        symbol_1 = transformLst[0]._basis.symbol
        print(symbol_1, ' = ', M1, '\n')
        M2 = transformLst[1]._basis.value
        symbol_2 = transformLst[1]._basis.symbol
        print(symbol_2, ' = ', M2, '\n\n')
        ret = self._utils.matrix_1T_TW_matrix_2(M1, _T, M2, False, _n, _verbose)
        
        return ret
    
    # change configuration for a second order tensor
    
    def changeConfig(self, _T, _inPosLst, _outPosLst, _G, _Ginv):
        
        n = len(_inPosLst)
        m = len(_outPosLst)
        
        ret = []
        
        if n != m:
            print('input and output position lists are different lengths')
            return ret
        
        ret = copy.deepcopy(_T)
        
        if _inPosLst[0] != _outPosLst[0]:
            if _inPosLst[0] == pos.up: # up to down 
                ret = np.dot(_G, ret)
            else:
                # down to up
                ret = np.dot(_Ginv, ret)
        
        if _inPosLst[1] != _outPosLst[1]:
            if _inPosLst[1] == pos.up: # up to down 
                ret = np.dot(ret, _G)
            else:
                # down to up
                ret = np.dot(ret, _Ginv)
                
        return ret
                
   
class thirdOrderTensor:
    
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
       
      
    def computeTensorOuterProduct(self, _T, _posLst, _unprimed, _n, _verbose):
        
        # put basis matrices into vectors
        
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, True, _n)
        
        ret = 0
        eqstr = ''
        
        for i in range(_n):
            for j in range(_n):
                for k in range(_n):
                    coeff = _T[i,j,k]
                    if _verbose:
                        if eqstr != '':
                            eqstr += '+' + '(' + str("{:.6f}".format(coeff)) + ')'
                        else:
                            eqstr += '(' + str(coeff) + ')' 
                 
                    outer = np.einsum('i,j,k',_basisLst[0][i], _basisLst[1][j], _basisLst[2][k])
                    l_outer = self.convertToLatex(outer, _n)
                    eqstr += l_outer
                    ret += coeff*outer
     
        if _verbose:
            print('T' + self._posNum.coordinateIndice(_posLst[0], 0, True)  +  self._posNum.coordinateIndice(_posLst[1], 1, True) + self._posNum.coordinateIndice(_posLst[2], 2, True) + ' = ')
            print(eqstr, '\n')
            
        return ret
      
    def computeWeightMatrix(self, _T, _posLst, _basis, _basis_symbol, _n, _verbose):
        
        # compute T_ij = T(i,j,k)L_il
        
        ret = self._utils.blockInnerProduct(_T, 'T', _posLst, _basis, _basis_symbol, _n, _verbose) 
        return ret
        
        
    def computeTensorInnerProduct(self, _T, _posLst, _unprimed, _n, _verbose):
        
        # put basis matrices into vectors
        
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, False, _n)
        
        # declare matrices
        
        basis_1 = _basisLst[0]
        basis_2 = _basisLst[1]
        basis_3 = _basisLst[2]
        
        # print matrices
        
        if _verbose:
            if _unprimed:
                self.printMatrix(basis_1, 'L = ' + _symbolLst[0], _n)  
                self.printMatrix(basis_2, 'F = ' + _symbolLst[1], _n)
                self.printMatrix(basis_3, 'H = ' + _symbolLst[2], _n)
            else:
                self.printMatrix(basis_1, 'L̅ = ' + _symbolLst[0], _n)  
                self.printMatrix(basis_2, 'F̅ = ' + _symbolLst[1], _n)
                self.printMatrix(basis_3, 'H̅ = ' + _symbolLst[2], _n)
                
          
        # Compute weight matrix
        
        if _unprimed: 
            T_ij = self.computeWeightMatrix(_T, _posLst, _basisLst[0], _symbolLst[0], _n, _verbose) 
        else:
            # use weight that has been transformed to a primed coordinate system
            T_ij = _T
          
        ret = []
        
        for i in range(_n):
            if _verbose:
                if _unprimed:
                    print('FᵀT'+self._posNum.coordinateIndice(pos.down, i, False)+'H term by term ')
                else:
                    print('F̅ᵀT̅'+self._posNum.coordinateIndice(pos.down, i, False)+'H̅ term by term ')
                    
            tmp = self._utils.matrix_1T_TW_matrix_2(basis_2, T_ij[i], basis_3, False, _n, _verbose)
            if _verbose:
                if _unprimed:
                    self.printMatrix(tmp, 'FᵀT'+self._posNum.coordinateIndice(pos.down, i, False)+'H', _n)
                else:
                    self.printMatrix(tmp, 'F̅ᵀT̅'+self._posNum.coordinateIndice(pos.down, i, False)+'H̅', _n)
                    
            ret.append(tmp)
       
        ret = np.array(ret)
        return ret
    
    def transformTensor(self, _T, _posLst, _n, _verbose):
        
        # transforms tensor coordinates
        
        _transformLst = self._utils.processTransformInput(_posLst, self, _n)
       
        
        M1 = _transformLst[1]._basis.value
        symbol_1 = _transformLst[1]._basis.symbol
        M2 = _transformLst[2]._basis.value
        symbol_2 = _transformLst[2]._basis.symbol

        L = self._forwardTable[_posLst[0]]._basis.value
        symbol_L = self._forwardTable[_posLst[0]]._basis.symbol
             
        if _verbose:
            self.printMatrix(M1, 'M1 = ' + symbol_1, _n)
            self.printMatrix(M2, 'M2 = ' + symbol_2, _n)
            self.printMatrix(L, 'L = ' + symbol_L  , _n)
            
        #compute T_n
        
        T_n = self._utils.blockInnerProduct(_T, 'T', _posLst, L, symbol_L, _n, _verbose)
         
        # perform transpose(M2).Tn.M3
        
        ret = []
        
        for i in range(_n):
            if _verbose:
                print(symbol_1 + 'ᵀT'+self._posNum.coordinateIndice(pos.down, i, False) + symbol_2 + ' = '  + str(np.transpose(M1)) + str(_T) + str(M2) +  ' term by term ')
                
            tmp = self._utils.matrix_1T_TW_matrix_2(M1, T_n[i], M2, False, _n, _verbose)
            ret.append(tmp)
            if _verbose:
                self.printMatrix(tmp, 'T̅' + self._posNum.coordinateIndice(pos.down, i, False), _n)
        
        T1_n = np.array(ret)
        
        # convert [T1(1, j, k), T1(2, j, k)] to T1_ijk
        # get L_prime
        
        print('Convert [T1(1, j, k), T1(2, j, k)] to T1_ijk\n')
        
        _basisLstPrime, _symbolLstPrime = self._utils.processTensorInput(_posLst, False, self, False, _n)
        
        L_prime = _basisLstPrime[0] 
        L_prime_inv = np.linalg.inv(L_prime) # need to use L_prime inverse
        
        T1_ijk = self._utils.blockInnerProduct(T1_n, 'T̅', _posLst, L_prime_inv, '[inverse(L̅)]',_n, _verbose) 
        
        return T1_n, T1_ijk
    
    # change configuration for a third order tensor
    
    def changeConfig(self, _T, _inPosLst, _outPosLst, _G, _Ginv, _n, _verbose):
        
        n = len(_inPosLst)
        m = len(_outPosLst)
        
        ret = copy.deepcopy(_T)
        
        if n != m:
            print('input and output position lists are different lengths')
            return ret
    
        if _inPosLst[0] != _outPosLst[0]:
            if _inPosLst[1] == pos.up: # up to down 
                ret = self._utils.blockInnerProduct(_T, 'T', _inPosLst, _G, 'G', _n, _verbose)
            else:
                ret = self._utils.blockInnerProduct(_T, 'T', _inPosLst, _Ginv, 'G⁻¹', _n, _verbose)
                

        if _inPosLst[1] != _outPosLst[1]:
            for i in range(n - 1):
                if _inPosLst[1] == pos.up: # up to down 
                        ret[i] = np.dot(_G, ret[i])
                else:
                    # down to up
                    ret[i] = np.dot(_Ginv, ret[i])
                    
        if _inPosLst[2] != _outPosLst[2]:
            for i in range(n - 1):
                if _inPosLst[2] == pos.up: # up to down 
                        ret[i] = np.dot(ret[i], _G )
                else:
                    # down to up
                    ret[i] = np.dot(ret[i], _Ginv )
                    
        return ret
   
    
    def printMatrix(self, _M, _label, _n):
         l_result = self._latex.convertMatrixToLatex(_M, _n)
         print(_label + ' = ', l_result, '\n')
     
    def printVector(self, _vec, _transpose, _label, _n):
         l_result = self._latex.convertVectorToLatex(_vec, _transpose, _n)
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
          print(_elem._transposeSymbol  + ' = ' + l_transposeBasis + '\n')
          l_basis = self._latex.convertMatrixToLatex(_elem._basis, _n)
          print(_elem._symbol + ' = ' + l_basis + '\n')
          return 


    def computeWeightMatrix(self, _T, _posLst, _basisLst, _symbolLst, _unprimed, _n, _verbose):
        
        # computes weight matrix using matrix operations transpose(B3).T.B4
        # _tuple = (k = (up/down), l = (up/down))
        
        # get Basis 3 and Basis 4 in matrix form
        
        F = _basisLst[2]
        symbol_F = _symbolLst[2]
        H = _basisLst[3]
        symbol_H = _symbolLst[3]
        
        # print Basis 3 and Basis 4
        
        if _verbose:
            self.printMatrix(F, 'F = ' + symbol_F, _n)
            self.printMatrix(H, 'H = ' + symbol_H, _n)
        
        ret = self.allocateFourthOrderElement(_n)
        
        for i in range(_n):
            for j in range(_n):
                TW = _T[i,j]
                if _verbose:
                    l_t_matrix = self._latex.convertMatrixToLatex(TW, _n)
                    print('T' + self._posNum.coordinateIndice(_posLst[0], i, False) + self._posNum.coordinateIndice(_posLst[1], j, False) + self._posNum.coordinateIndice(_posLst[2], 2, True) + self._posNum.coordinateIndice(_posLst[3], 3, True) + ' = ' + l_t_matrix + '\n')
                result = self._utils.matrix_1T_TW_matrix_2(F, TW, H, False, _n, False)
                
                if _verbose:
                    l_result = self._latex.convertMatrixToLatex(result, _n)
                    print('[T_block]' + self._posNum.coordinateIndice(_posLst[0], i, False) + self._posNum.coordinateIndice(_posLst[1], j, False) + ' = ' + l_result + '\n') 
                ret[i][j] = result
            
        return ret

                
    
    def computeTensorInnerProduct(self, _T, _posLst, _unprimed, _n, _verbose):
       
         # implements streamlined calculation
         # compute weight matrices
             
         ret = self.allocateFourthOrderElement(_n)
         
         print('Compute Tensor Inner Product\n')
         
         _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, False, _n)
       
        
         print('Weight Matrix Computed with Matrix Product\n')
 
         # compute weight matrix in the unprimed system
          
        
         if _unprimed: 
             T_ij = self.computeWeightMatrix(_T, _posLst, _basisLst, _symbolLst, True, _n, _verbose)
         else:
             # use weight that has been transformed to a primed coordinate system
             T_ij = _T
         
         # _tuple = (k = (up/down), l = (up/down))
         
         # get Basis 1 and Basis 2
         
         C = _basisLst[0]
         symbol_C = _symbolLst[0]
         D = _basisLst[1]
         symbol_D = _symbolLst[1]
         
        
         # print Basis 1 and Basis 2
         
         self.printMatrix(C, 'C = ' + symbol_C + ' = ', _n)
         self.printMatrix(D, 'D = ' + symbol_D + ' = ', _n)
  
         # compute transpose(C).T(i,j).D
         
         print('Compute Submatrices of transpose(C).T_block(i,j).D\n')
         
         ret = self._utils.matrix_1T_TW_matrix_2(C, T_ij, D, True, _n, _verbose)
        
         return ret
         
           
    def allocateFourthOrderElement(self, _n):
        
        ret_ij = np.array([[[[0.0]*_n]*_n]*_n]*_n)
        return ret_ij
             
    def computeTensorOuterProduct(self, _T, _posLst, _unprimed, _n, _verbose):
     
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, _unprimed, self, True, _n)
        
        ret = 0
    
        for i in range(_n):
            for j in range(_n):
                for k in range(_n):
                    for l in range(_n):
                        coeff = _T[i, j, k, l]
                        if _verbose:
                            print('T' + self._posNum.coordinateIndice(_posLst[0], i, False)  +  self._posNum.coordinateIndice(_posLst[1], j, False) + self._posNum.coordinateIndice(_posLst[2], k, False) + self._posNum.coordinateIndice(_posLst[3], l, False)  + ' = ', str("{:.6f}".format(coeff) ))
                            self.printVector( _basisLst[0][i], False, _symbolLst[0].lower() + self._posNum.basisIndice(_posLst[0], i, False), _n)
                            self.printVector( _basisLst[1][j], False, _symbolLst[1].lower() + self._posNum.basisIndice(_posLst[1], j, False), _n) 
                            self.printVector( _basisLst[2][k], False, _symbolLst[2].lower() + self._posNum.basisIndice(_posLst[2], k, False), _n)
                            self.printVector( _basisLst[3][l], False, _symbolLst[3].lower() + self._posNum.basisIndice(_posLst[3], l, False), _n)
                        outer = np.einsum('i,j,k,l',_basisLst[0][i], _basisLst[1][j], _basisLst[2][k],_basisLst[3][l])
                        l_outer = self.convertElementToLatex(outer, _n)
                        if _verbose:
                            print('outer' + self._posNum.basisIndice(_posLst[0], i, False) + self._posNum.basisIndice(_posLst[1], j, False) + self._posNum.basisIndice(_posLst[2], k, False) + self._posNum.basisIndice(_posLst[3], l, False),'\n')
                            print(l_outer, '\n')
                             
                        ret += coeff*outer
       
        return ret
    
        
    def processTransformInput(self, _posLst, _unprimed, _n):
        
        ret = computeElement()
        
        for i in _posLst:
            ret = self._transformTable[i]
        
        return ret
        
    # transform tensor
    
    def transformTensor(self, _T, _posLst, _n, _verbose):
        
        # 1st work unprimed to primed system
        # compute tensor coordinate change using matrices
       
        _basisLst, _symbolLst = self._utils.processTensorInput(_posLst, True, self, False, _n)
       
        # Compute weight matrix in unprimed system
        
        T_ij = self.computeWeightMatrix(_T, _posLst, _basisLst, _symbolLst, True, _n, False)  # no need to print the T_block calculations in the unprimed system
        
        if _verbose:
            print('Block Tensor Components in Unprimed System\n')
            self.printWeightMatrices(T_ij, _posLst, _n)
        
        # get transforms involved with coordinate change
        
        _transformLst = self._utils.processTransformInput(_posLst, self, _n)
        
        # allocate return structure
        
        M1 = _transformLst[0]._basis.value
        M2 = _transformLst[1]._basis.value
     
        
        print('Compute Submatrices of transpose(M1^-1).T_block(i,j).M2^-1\n')
        
        T_prime_ij = self._utils.matrix_1T_TW_matrix_2(M1, T_ij, M2, True, _n, _verbose)
        
        
        # Transform T_prime_ij -> T_prime_ijkl
        
        print('Transform  T_prime_ij to T_prime_ijkl\n')
        
        _basisLstPrime, _symbolLstPrime = self._utils.processTensorInput(_posLst, False, self, False, _n) 
       
        T_prime_ijkl = self.allocateFourthOrderElement(_n)
       
        F_primeT = np.transpose(_basisLstPrime[2])  
        F_primeT_inv = np.linalg.inv(F_primeT) # need to use F_prime inverse
        F_prime_inv = np.transpose( F_primeT_inv) # transpose because the matrix_1T_TW_matrix_2 automatically transposes
                                                  # could put a switch in matrix_1T_TW_matrix_2 to indicate if it should compute transposing the first matruix or not
        
        H_prime = _basisLstPrime[3]
        H_prime_inv = np.linalg.inv(H_prime) # need to use H_prime inverse
        
        for i in range(_n):
            for j in range(_n):
                if _verbose:
                    print('(' + str(i) + ',' + str(j) + ') component\n')
                T_prime_ijkl[i][j] = self._utils.matrix_1T_TW_matrix_2( F_prime_inv, T_prime_ij[i][j], H_prime_inv, False, _n, _verbose)
                if _verbose:
                    l_T_prime_ijkl = self._latex.convertMatrixToLatex(T_prime_ijkl[i][j], _n)
                    print('[T_ijkl]' + self._posNum.coordinateIndice(_posLst[0], i, False) + self._posNum.coordinateIndice(_posLst[1], j, False) + ' = ' + l_T_prime_ijkl + '\n')
            
        return T_prime_ij, T_prime_ijkl
       
    def changeConfig(self, _T, _inPosLst, _outPosLst, _G, _Ginv, _n, _verbose):
        
        n = len(_inPosLst)
        m = len(_outPosLst)
        
        ret = copy.deepcopy(_T)
        
        if n != m:
            print('input and output position lists are different lengths')
            return ret
    
        
        if _outPosLst[0] != _inPosLst[0]:
            if _outPosLst[0] == pos.down: # _inPosLst[0] == pos.up
                ret = self._utils.blockDotProduct(_G, ret, _n)
            else:
                ret = self._utils.blockDotProduct(_Ginv, ret, _n)
                
        
        if _outPosLst[1] != _inPosLst[1]:
            if _outPosLst[1] == pos.down: # _inPosLst[1] == pos.up
                ret = self._utils.blockDotProduct(ret, _G, _n)
            else:
                ret = self._utils.blockDotProduct(ret, _Ginv, _n)
                
        if _outPosLst[2] != _inPosLst[2]:
            if _outPosLst[2] == pos.down: # _inPosLst[2] == pos.up
                ret = self._utils.tensorProduct([_G], ret, _n)
            else:
                ret = self._utils.tensorProduct([_Ginv], ret, _n)
                
        if _outPosLst[3] != _inPosLst[3]:
             if _outPosLst[3] == pos.down: # _inPosLst[3] == pos.up
                 ret = self._utils.tensorProduct(ret, [_G], _n)
             else:
                 ret = self._utils.tensorProduct( ret, [_Ginv], _n)
                 
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

    def printWeightMatrices(self, _T, _posLst, _n):
    
        for i in range(_n):
            for j in range(_n):
                self.printMatrix(_T[i][j], 'T' + self._posNum.coordinateIndice(_posLst[2], i, False) + self._posNum.coordinateIndice(_posLst[3], j, False), _n)
 
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
    
    def basisIndice(self, _pos, _index, _letterFlag):
    
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
       
        
    def coordinateIndice(self, _pos, _index, _letterFlag):
        
         
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

class transformElement:
    
    # This class holds the transforms corresponding to a table of transforms
    # for use with an equation of the form M1^T(T)M2
    # there is no need for an explicit transpose in the computation because the transpose 
    # is taken into effect by revering the indices
    # so for this reason, the transpose attribute holds the untransposed matrix
    
    def __init__(self):
        self._indx = ''
        self._symbol = ''
        self._transposeFlag = False

class computeElement:
    
    def __init__(self):
        self._basis = dictElement()
        self._transposeFlag = False
    
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
    
    def __init__(self):
        
        self._posNum = posNum()
        self._latex = convertToLatex()
    
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
    
    def blockInnerProduct(self, _block_vec, _symbol, _posLst, _L, _symbol_L, _n, _verbose):
        
        # multiplies a block vector ( vector whose elements are matrices ) with a vector
        # and returns a block vector 
        
        ret = []
        eqstr = ''
        
        for j in range(_n):
            sum = 0
            for k in range(_n): 
                sum +=  _block_vec[k]*_L[k][j]
                if _verbose:
                    if eqstr == '':
                        eqstr += _symbol + '.' + _symbol_L + '(' + str(j + 1) + ') = ' 
                        eqstr += str(_symbol) + self._posNum.coordinateIndice(pos.down, j, False) + ' = ' + str(_block_vec[k]) + '(' + str("{:.6f}".format(_L[k][j])) + ')'
                    else:
                        eqstr += '+' + str(_block_vec[k]) + '(' + str("{:.6f}".format(_L[k][j])) + ')'
                 
            ret.append(sum)
            
            if _verbose:
                l_sum = self._latex.convertMatrixToLatex(sum, _n)
                print(eqstr,' = ', l_sum, '\n')
                eqstr = ''
              
        ret = np.array(ret)
        return ret
    
    def tensorProduct(self, _elem_1, _elem_2, _n):
        
        ret = []
        
        # find which input element is a list
        
        reverseFlag = False
        
        if isinstance(_elem_1, list):
            start = _elem_1
            tmp_2 = _elem_2
        else:
            start = _elem_2
            tmp_2 = _elem_1
            reverseFlag = True
            
        
    
        for k in start:
            for i in range(_n):
                for j in range(_n):
                    if reverseFlag:
                        ret.append(np.dot(tmp_2[i][j], k))
                    else:
                        ret.append(np.dot(k, tmp_2[i][j])) # dot return a regular multiplication for two scalars
        
        start = start[0]
        
        if start.shape < tmp_2.shape:
            shape_factor = tmp_2.shape
        elif start.shape == tmp_2.shape:
            shape_factor = start.shape + tmp_2.shape
        else:
            shape_factor = start.shape
       
        ret = np.array(ret)
        ret = np.reshape(ret, shape_factor)
        
        return ret
        
    def blockDotProduct(self, _elem_1, _elem_2, _n):
        
        ret = []

        for i in range(_n):
            for j in range(_n):
                acc = 0.0
                for k in range(_n):
                    acc += np.dot(_elem_1[i][k], _elem_2[k][j])
                ret.append(acc)
        
        if _elem_1.shape < _elem_2.shape:
            shape_factor = _elem_2.shape
        elif _elem_1.shape == _elem_2.shape:
            shape_factor = _elem_1.shape + _elem_2.shape
        else:
            shape_factor = _elem_1.shape
       
        ret = np.array(ret)
        ret = np.reshape(ret, shape_factor)
        return ret
           
        
    def printMatrix(self, _M, _label, _n):
         l_result = self._latex.convertMatrixToLatex(_M, 2)
         print(_label + ' = ', l_result, '\n')
    
    def matrix_1T_TW_matrix_2(self, _matrix_1, _TW, _matrix_2, _block, _n, _verbose):
        
        # computes transpose(_matrix_1)._TW._matrix_2
        # _block specifies _TW as a block matrix and the return type as a block matrix
        
        #if _verbose:
        #    print(
        
        if not _block:
            ret = np.zeros(_n*_n).reshape((_n, _n))
        else:
            ret = self.allocateFourthOrderElement(_n)
      
        eqstr = ''
      
        for i in range(_n):
            for j in range(_n): 
                if _verbose:
                    print('(i, j) = (' + str(i+1) + ',' +  str(j+1) + ')')
                acc = 0.0
                for k in range(_n):
                    for l in range(_n):
                        acc += _matrix_1[k][i]*_TW[k][l]*_matrix_2[l][j]
                        if _verbose:
                            if _block:
                                l_TW_kl = self._latex.convertMatrixToLatex(_TW[k][l], _n)
                                if eqstr:
                                    eqstr += '+' + '(' + str("{:.6f}".format(_matrix_1[k][i])) + ')' + l_TW_kl + '(' + str("{:.6f}".format(_matrix_2[l][j])) + ')'
                                else:
                                    eqstr += '+ (' + str("{:.6f}".format(_matrix_1[k][i])) + ')' + l_TW_kl + '(' + str("{:.6f}".format(_matrix_2[l][j])) + ')'
                            else:
                                eqstr += '+' + '(' + str("{:.6f}".format(_matrix_1[k][i])) + ')' + '(' + str("{:.6f}".format(_TW[k][l])) + ')(' + str("{:.6f}".format(_matrix_2[l][j])) + ')'
                ret[i,j] = acc
                
                if _verbose:
                    print(eqstr[1:] + '\n')
                    if _block:
                        l_result = self._latex.convertMatrixToLatex(acc, _n)
                        print(' = ', l_result, '\n')
                    else:
                        print(' = ', "{:.6f}".format(acc), '\n')
                eqstr = ''
                  
        return ret
       
    
    
    def computeTable(self, tableFunc, _unprimed, _class):
    
        # mapping to transform T(i,j,k,l) -> T(i,j) 
        # and mapping to transform T(i,j) -> Full Tensor
        
        ret = {}
        
        indx = [pos.up, pos.down]
         
        for i in indx:
            ret[i] = tableFunc(i, _unprimed, _class)  
            
        return ret
    
    def getRows(self, _basis, _n):
        
        ret = []
        tmp_vec = []
        
        tmp = _basis._basis.value
        for i in range(_n):
            tmp_vec.append(tmp[i])
        
        ret.append(tmp_vec)
        
        return ret
    
    def getSymbols(self, _element, _n):
        
        ret = []
        tmp = _element._basis.symbol
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
            for i in basesLst:
                tmp = i._basis.value
                bases.append(tmp)
                
        
        # get symbols in vector form
        
        
        symbol_vecs = []
        for i in basesLst:
            tmp = self.getSymbols(i, _n)
            symbol_vecs += tmp
            
        return bases, symbol_vecs
   
    def processTransformInput(self, _posLst, _class, _n):
        
        ret = []
        
        for i in _posLst:
            tmp = _class._transformTable[i]
            ret.append(tmp)
        
        return ret
    
    def getBases(self, _posLst, _unprimed, _class):
    
        basesLst = []
        
        for i in _posLst:
            if _unprimed:
                tmp = _class._forwardTable[i]
            else:
                tmp = _class._forwardTablePrimed[i]
            basesLst.append(tmp)
        
        return basesLst
    
    def forwardTable(self, _pos, _unprimed, _class):
        
        # function to implement Table 2 and Table 5 from the writeup
        # note there is no need to compute the transposes, they are not needed 
        # the way the algorithms are written, but need to be kept track of
        # _pos can be pos.up or pos.down
        
        ret = computeElement()
        suffix = ''
        
        if not _unprimed:  # prime coordinate system
            suffix = '1'
        
        if _pos == pos.up:
            indx = 'E' + suffix
        else:
            indx = 'W' + suffix
        
        ret._basis = _class._vars._vars[indx]
        
        return ret
   
    
    def transformTable(self, _pos, _unprimed, _class):
        
        ret = computeElement()
        ret._transposeFlag = False
        
        if _pos == pos.up:
            indx = 'B'
        else:
            indx = 'AT'
            ret._transposeFlag = True
                  
        ret._basis = _class._vars._vars[indx]
        
        return ret

            
                
            
        

        
     