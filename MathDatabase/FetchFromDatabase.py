# -*- coding: utf-8 -*-
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









