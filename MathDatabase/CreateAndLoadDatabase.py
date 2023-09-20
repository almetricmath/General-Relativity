# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:29:33 2023

@author: Al
"""

import sympy as sp
from sympy import * #noqa
from sympy.printing.latex import latex as sp_latex
import mathDB as mathDB
import sys


def I(n):
    return sp.eye(n)

# class to compute symbolic natrices

class computeMatrices:
    
    @staticmethod
    def A_matrix(_coords):
        
        params = _coords._params
        vec = _coords._vec
        _n = len(params)
        ret = sp.Matrix(0, 0, [])
       
        for i in range(_n):
            ret = ret.row_insert(i, sp.Matrix([sp.diff(vec, params[i])]))
        
        return ret
        
    @staticmethod
    def B_matrix(_A_matrix: sp.Matrix):
        
        ret = sp.Matrix(0, 0, [])
        
        if _A_matrix.shape != (0, 0):
            ret = sp.trigsimp(_A_matrix.inv())
        else:
            print("Error - A matrix not computed")
        
        return ret
        
        

# class with static methods to compute polar matrices from cartesian coordinates


class polarFromCartesian:
    
    def __init__(self, _r, _theta):
        self._params = sp.Array([r, theta])
        self._vec = sp.Array([self._params[0]*sp.cos(self._params[1]), self._params[0]*sp.sin(self._params[1])]) # definitions of x and y
    

class polarSqrtFromPolar:
    
    def __init__(self, _r_bar, _theta_bar):
        self._params = sp.Array([_r_bar, _theta_bar])
        self._vec = sp.Array([sp.sqrt(self._params[0]), sp.sqrt(self._params[1])])


class polar1FromPolar:
    
    @staticmethod
    def A_matrix(_r1, _theta1):
        ret = Matrix([[1, 1],[1, -1 ]]) #noqa
        return ret 
    @staticmethod 
    def B_matrix(_r1, _theta1):
        ret = Matrix([[1/2, 1/2],[1/2, -1/2 ]]) #noqa
        return ret

class polarSqrt1FromPolar:
    
    @staticmethod
    def A_matrix(_r1, _theta1):
        ret = Matrix([[1/(2*sp.sqrt(_r1)), 1/(2*sp.sqrt(_r1))],[1/(2*sp.sqrt(_theta1)),-1/(2*sp.sqrt(_theta1))]]) #noqa
        return ret 
    @staticmethod 
    def B_matrix(_r1, _theta1):
        ret = Matrix([[sp.sqrt(_r1), sp.sqrt(_r1)],[sp.sqrt(_theta1), -1*sp.sqrt(_theta1)]]) #noqa
        return ret

class transformRecord:
    
    def __init__(self, _A, _B, _E, _W):
        self._A = _A
        self._B = _B
        self._E = _E
        self._W = _W


mathData = mathDB.mathDB('math.db')
latex = mathDB.convertToLatex()

# cartesian to cartesian

A_cart = I(2)
B_cart = I(2)
E_cart = I(2)
W_cart = I(2)

tmpRec = transformRecord(A_cart, B_cart, E_cart, W_cart)
mathData.insertIntoTransformTable('cartesian', 'cartesian', tmpRec)

# polar to polar

A_polar = I(2)
B_polar = I(2)
E_polar = I(2)
W_polar = I(2)

tmpRec = transformRecord(A_polar, B_polar, E_polar, W_polar)
mathData.insertIntoTransformTable('polar', 'polar', tmpRec)

#  cartesian to polar 

r = symbols('r') #noqa
theta = symbols('theta') #noqa

pFc = polarFromCartesian(r, theta)
cM = computeMatrices()
A_polar_cart = cM.A_matrix(pFc)
A_polar_cart_latex = latex.convertMatrixToLatex(A_polar_cart)
print("A(polar, cartesian) = ", A_polar_cart_latex, '\n')
B_polar_cart = cM.B_matrix(A_polar_cart)
B_polar_cart_latex = latex.convertMatrixToLatex(B_polar_cart)
print("B(polar, cartesian) = ", B_polar_cart, '\n')

E_cart = I(2)
W_cart = I(2)
E_polar = A_polar_cart*E_cart
E_polar_latex = latex.convertMatrixToLatex(E_polar)
print("E(polar) = ", E_polar_latex, '\n')
W_polar = sp.transpose(B_polar_cart)*W_cart
W_polar_latex = latex.convertMatrixToLatex(W_polar)
print("W(polar) = ", W_polar_latex, '\n')

tmpRec = transformRecord(A_polar_cart, B_polar_cart, E_polar, W_polar)
mathData.insertIntoTransformTable('polar', 'cartesian', tmpRec)

# polarSqrt from polar

r_bar = symbols('\\bar{r}') #noqa
theta_bar = symbols('\\bar{\\theta}') #noqa

E_polar = I(2)
W_polar = I(2)

pSqrtFp = polarSqrtFromPolar(r_bar, theta_bar)
A_polarSqrt_polar = cM.A_matrix(pSqrtFp)
A_polarSqrt_polar_latex = latex.convertMatrixToLatex(A_polarSqrt_polar)
print("A(polarSqrt, polar) = ", A_polarSqrt_polar_latex, '\n')
B_polarSqrt_polar = cM.B_matrix(A_polarSqrt_polar)
B_polarSqrt_polar_latex = latex.convertMatrixToLatex(B_polarSqrt_polar)
print("B(polarSqrt, polar) = ", B_polarSqrt_polar_latex, '\n')

E_polarSqrt = A_polarSqrt_polar*E_polar
E_polarSqrt_latex = latex.convertMatrixToLatex(E_polarSqrt)
print("E(polarSqrt) = ", E_polarSqrt_latex, '\n')
W_polarSqrt = sp.transpose(B_polarSqrt_polar)*W_polar
W_polarSqrt_latex = latex.convertMatrixToLatex(W_polarSqrt)
print("W(polarSqrt) = ", W_polarSqrt_latex, '\n')


# polar to polarSqrt1





mathData.close()
sys.exit(0)








