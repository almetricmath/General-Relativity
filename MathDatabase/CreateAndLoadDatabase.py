FetchFromDatabase.py                                                                                0000777 0001750 0001750 00000006417 14476137207 012536  0                                                                                                    ustar   al                              al                                                                                                                                                                                                                     # -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:09:32 2023

@author: Al
"""

import sympy as sp
from sympy import * #noqa
from sympy.printing.latex import latex as sp_latex

import mathDB

class transformRecord:
    
    def __init__(self, _A, _B):
        self._A = _A
        self._B = _B


mathData = mathDB.mathDB('math.db')
latex = mathDB.convertToLatex()

transformTable = mathData.getTransformTable()

# get transforms from cartesian to the various basis sets

key = ('polar', 'cartesian')
if key in transformTable:
   A_polar = transformTable[key]._A
   E_polar = transformTable[key]._E
   B_polar = transformTable[key]._B
   W_polar = transformTable[key]._W
   # output matrices in latex form
   
   print("Polar in terms of Cartesian coordinates \n\n")
   
   A_latex = latex.convertMatrixToLatex(A_polar)
   print("A = ", A_latex)
   E_latex = latex.convertMatrixToLatex(E_polar)
   print("E = ", E_latex)
   B_latex = latex.convertMatrixToLatex(B_polar)
   print("B = ", B_latex)
   W_latex = latex.convertMatrixToLatex(W_polar)
   print("W = ", W_latex, '\n\n')
else:
    print(key, " not in database")


key = ('polarsqrt', 'polar')
if key in transformTable:
   A_polarSqrt = transformTable[key]._A
   E_polarSqrt = transformTable[key]._E
   B_polarSqrt = transformTable[key]._B
   W_polarSqrt = transformTable[key]._W

   # output matrices in latex form
   print("PolarSqrt in terms of Polar coordinates\n\n")
   
   A_latex = latex.convertMatrixToLatex(A_polarSqrt)
   print("A = ", A_latex)
   E_latex = latex.convertMatrixToLatex(E_polarSqrt)
   print("E = ", E_latex)
   B_latex = latex.convertMatrixToLatex(B_polarSqrt)
   print("B = ", B_latex)
   W_latex = latex.convertMatrixToLatex(W_polarSqrt)
   print("W = ", W_latex, '\n\n')
   
else:
    print(key, " not in database")

key = ('polar1', 'polar')
if key in transformTable:
    A_polar1 = transformTable[key]._A
    E_polar1 = transformTable[key]._E
    B_polar1 = transformTable[key]._B
    W_polar1 = transformTable[key]._W
    # output matrices in latex form
    print("Polar1 in terms of Polar coordinates\n\n")
    
    A_latex = latex.convertMatrixToLatex(A_polar1)
    print("A = ", A_latex)
    E_latex = latex.convertMatrixToLatex(E_polar1)
    print("E = ", E_latex)
    B_latex = latex.convertMatrixToLatex(B_polar1)
    print("B = ", B_latex)
    W_latex = latex.convertMatrixToLatex(W_polar1)
    print("W = ", W_latex, '\n\n')
   
    
else:
    print(key, " not in database")
    

key = ('polarSqrt1', 'polar')
if key in transformTable:
    A_polarSqrt1 = transformTable[key]._A
    E_polarSqrt1 = transformTable[key]._E
    B_polarSqrt1 = transformTable[key]._B
    W_polarSqrt1 = transformTable[key]._W
    # output matrices in latex form
    print("PolarSqrt1 in terms of Polar coordinates\n\n")
    
    A_latex = latex.convertMatrixToLatex(A_polarSqrt1)
    print("A = ", A_latex)
    E_latex = latex.convertMatrixToLatex(E_polarSqrt1)
    print("E = ", E_latex)
    B_latex = latex.convertMatrixToLatex(B_polarSqrt1)
    print("B = ", B_latex)
    W_latex = latex.convertMatrixToLatex(W_polarSqrt1)
    print("W = ", W_latex, '\n\n')
   
    
else:
    print(key, " not in database")


mathData.close()









                                                                                                                                                                                                                                                 mathDB.py                                                                                           0000777 0001750 0001750 00000004552 14476137247 010375  0                                                                                                    ustar   al                              al                                                                                                                                                                                                                     # -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:11:17 2023

@author: Al
"""
from sympy.printing.latex import latex as sp_latex
from pathlib import Path
import pickle
import os



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
            
    def getTransformTable(self):
        return self._transformTable
    
    def insertIntoTransformTable(self, _toKey, _fromKey, _record):
        
        tmpKey = (_toKey, _fromKey )
        
        self._transformTable[tmpKey] = _record
        
        
    def close(self):
        
        self._dictionary['transformTable'] = self._transformTable
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
        
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      