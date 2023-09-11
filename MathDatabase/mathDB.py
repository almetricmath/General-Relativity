# -*- coding: utf-8 -*-
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
        
        
