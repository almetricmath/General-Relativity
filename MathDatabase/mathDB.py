# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:11:17 2023

@author: Al
"""

import sympy as sp
from sympy.printing.latex import latex as sp_latex
from pathlib import Path
import pickle
import os
import string

def I(n):
    return sp.eye(n)

# class to compute symbolic natrices

class computeMatrices:
    
    def __init__(self, _logFilePtr):
        self._logFilePtr = _logFilePtr 
        self._latex = convertToLatex()
    
    
    def computeTransform(self, _params, _vec):
        
        _n = len(_params)
        ret = sp.Matrix(0, 0, [])
       
        for i in range(_n):
            ret = ret.row_insert(i, sp.Matrix([sp.diff(_vec, _params[i])]))
        
        return ret
        
    
    def A_matrix(self, _coords):
        
        params = _coords._params
        vec = _coords._vec
        ret = self.computeTransform(params, vec)
        ret_latex = self._latex.convertMatrixToLatex(ret)
        
        if self._logFilePtr:
            self._logFilePtr.write('Computing A matrix for ' + _coords._name + '\n')
            self._logFilePtr.write(ret_latex + '\n\n')
        
        return ret
        
        
    
    def B_matrix(self, _coords):
        
        # use this method if for some reason the A matrix is singular
        
        inv_params = _coords._inv_params
        inv_vec = _coords._inv_vec
        ret = self.computeTransform(inv_params, inv_vec)
        ret_latex = self._latex.convertMatrixToLatex(ret)
        if self._logFilePtr:
            self._logFilePtr.write('Computing B matrix for ' + _coords._name + '\n')
            self._logFilePtr.write(ret_latex + '\n' + '\n')
       
        # substitute to get both A and B matrices in terms of the same parameters
       
        vec = _coords._vec
        sub_str = dict(zip(inv_params, vec))
        ret = ret.subs(sub_str) 
        ret = sp.simplify(ret)
        ret_latex = self._latex.convertMatrixToLatex(ret)
        if self._logFilePtr:
            self._logFilePtr.write('B matrix After simplification ' + ret_latex + '\n' + '\n')
       
        # handle sqrt(r**2) = r
        params = _coords._params
        sub_str = map(lambda x: sp.sqrt(x**2), params)
        sub_str = dict(zip(sub_str, params))
        ret = ret.subs(sub_str)
        ret = sp.simplify(ret)
        ret_latex = self._latex.convertMatrixToLatex(ret)
        if self._logFilePtr:
            self._logFilePtr.write('B matrix After sqrt simplification ' + ret_latex + '\n' + '\n')
       
        return ret
        
    def close(self):
        if self._logFilePtr:
            self._logFilePtr.close()


# class with static methods to compute polar matrices from cartesian coordinates


class polarFromCartesian:
    
    def __init__(self, _r, _theta, _x, _y):
        self._name = 'polarFromCartesian'
        self._params = sp.Array([_r, _theta])
        self._vec = sp.Array([self._params[0]*sp.cos(self._params[1]), self._params[0]*sp.sin(self._params[1])]) # definitions of x and y
        self._inv_params = sp.Array([_x, _y])
        self._inv_vec = sp.Array([sp.sqrt(self._inv_params[0]**2 + self._inv_params[1]**2), sp.atan2(_y, _x)])
    

class polarSqrtFromPolar:
    
    def __init__(self, _r_bar, _theta_bar, _r, _theta):
        self._name = 'polarSqrtFromCartesian'
        self._params = sp.Array([_r_bar, _theta_bar])
        self._vec = sp.Array([sp.sqrt(self._params[0]), sp.sqrt(self._params[1])])
        self._inv_params = sp.Array([_r, _theta])
        self._inv_vec = sp.Array([_r**2, _theta**2])


class polar1FromPolar:
    
    def __init__(self, _r_bar, _theta_bar, _r, _theta):
        self._name = 'polar1FromPolar'
        self._params = sp.Array([_r_bar, _theta_bar])
        self._vec = sp.Array([_r_bar + _theta_bar, _r_bar - _theta_bar])
        self._inv_params = sp.Array([_r, _theta])
        self._inv_vec = sp.Array([(_r + _theta)/2, (_r - _theta)/2])
    
  
class polarSqrt1FromPolar:
    
    def __init__(self, _r_bar, _theta_bar, _r, _theta):
        self._name = 'polarSqrt1FromPolar'
        self._params = sp.Array([_r_bar, _theta_bar])
        self._vec = sp.Array([sp.sqrt(_r_bar) + sp.sqrt(_theta_bar), sp.sqrt(_r_bar) - sp.sqrt(_theta_bar)])
        self._inv_params = sp.Array([_r, _theta])
        self._inv_vec = sp.Array([((_r + _theta)/2)**2, ((_r - _theta)/2)**2])
    

class transformRecord:
    
    def __init__(self, _A, _B, _E, _W):
        self._A = _A
        self._B = _B
        self._E = _E
        self._W = _W
        
    def printRecord(self, _key):
        latex = convertToLatex()
        
        A_latex = latex.convertMatrixToLatex(self._A)
        print('A',str(_key),' = \n', A_latex, '\n')
        E_latex = latex.convertMatrixToLatex(self._E)
        print('E',str(_key),' = \n', E_latex, '\n')
        B_latex = latex.convertMatrixToLatex(self._B)
        print('B',str(_key),' = \n', B_latex, '\n')
        W_latex = latex.convertMatrixToLatex(self._W)
        print('W',str(_key),' = ', W_latex, '\n\n')
        
class coordinateRecord:
    
    def __init__(self, _class):
        self._name = _class._name
        self._params = _class._params
        self._vec = _class._vec
        self._inv_params = _class._inv_params
        self._inv_vec = _class._inv_vec
        
    def printRecord(self):
        print('name = ', self._name, '\n')
        print('_params = ', self._params, '\n')
        print('_vec = ', self._vec, '\n')
        print('_inv_params = ', self._inv_params, '\n')
        print('_inv_vec = ', self._inv_vec, '\n')

# develop a loop to process results given an input coordinate system

def createProcessLoop(_coords, msg, _logFileName:string):
    
    if len(_logFileName) != 0:
        logPtr = open(_logFileName, 'a')
    else:
        logPtr = None
    
    # process loop to create a transform record from the coordinate class
    
    latex = convertToLatex()
    
    cM = computeMatrices(logPtr)
    A = cM.A_matrix(_coords)
    A_latex = latex.convertMatrixToLatex(A)
    print('A', msg, A_latex, '\n')
    B = cM.B_matrix(_coords)
    B_latex = latex.convertMatrixToLatex(B)
    print('B', msg, B_latex, '\n')

    E = I(2)
    W = I(2)
    E_bar = A*E
    E_bar_latex = latex.convertMatrixToLatex(E_bar)
    print('E', msg, E_bar_latex, '\n')
    W_bar = sp.transpose(B)*W
    W_bar_latex = latex.convertMatrixToLatex(W_bar)
    print('W', msg,  W_bar_latex, '\n')

    tmpRec = transformRecord(A, B, E_bar, W_bar)
    
    cM.close()
    
    return tmpRec

class mathDB:
    
    def __init__(self, db_name): 
        
        self._db_name = db_name
        self._transformTable = {}
        self._coordinateTable = {}
        self._dictionary = {}
        filename = Path(db_name)
        filename.touch(exist_ok=True)
        self._fp = open(filename, 'rb+')
        
        _size = os.stat(self._fp.fileno()).st_size
        if _size > 0:
            self._fp.seek(0)
            self._dictionary = pickle.load(self._fp)
            self._transformTable = self._dictionary['transformTable']
            self._coordinateTable = self._dictionary['coordinateTable']
            
    def getTransformTable(self):
        return self._transformTable
    
    def insertIntoTransformTable(self, _toKey, _fromKey, _record):
        
        tmpKey = (_toKey, _fromKey )
        
        self._transformTable[tmpKey] = _record
     
    def insertIntoCoordinateTable(self, _toKey, _fromKey, _class):
        
        tmpKey = (_toKey, _fromKey )
        
        self._coordinateTable[tmpKey] = _class        
        
    def getCoordinateTable(self):
        return self._coordinateTable
        
        
    def close(self):
        
        self._dictionary['transformTable'] = self._transformTable
        self._dictionary['coordinateTable'] = self._coordinateTable
        self._fp.seek(0)
        pickle.dump(self._dictionary, self._fp)
        self._fp.close()
        return
    
class convertToLatex:

    def convertMatrixToLatex(self, _matrix):
        
        _n = _matrix.shape[0]
        _m = _matrix.shape[1]
        
        ret = '\\bmatrix{'
        
           
        for i in range(_n):
            for j in range(_m):
                ret += self.convertElementToLatex(_matrix[i, j])
                if j != _n - 1:
                    ret += '&'
                    
            if i != _n - 1:
                ret += '\\\\'
    
        ret += '}'
        return ret
    
    def convertVectorToLatex(self, _vec, transposeFlag, _n):
    
        ret = '\\bmatrix{'
        
        for i in range(_n):
            ret += self.convertElementToLatex(_vec[i])
            if i != _n - 1:
                if not transposeFlag:
                    ret += '&'
                else:
                    ret += '\\\\'
        
        ret += '\\\\}'
        return ret
    
    def convertElementToLatex(self, _elem):
        
        # formats matrix and vector symbolic elements to latex
        
        ret = sp_latex(_elem)
        
        return ret
        
        
