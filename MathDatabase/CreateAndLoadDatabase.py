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
    
    def computeTransform(_params, _vec):
        
        _n = len(_params)
        ret = sp.Matrix(0, 0, [])
       
        for i in range(_n):
            ret = ret.row_insert(i, sp.Matrix([sp.diff(_vec, _params[i])]))
        
        return ret
        
    
    @staticmethod
    def A_matrix(_coords):
        
        params = _coords._params
        vec = _coords._vec
        ret = computeMatrices.computeTransform(params, vec)
        
        return ret
        
        
    @staticmethod
    def B_matrix(_coords):
        
        # use this method if for some reason the A matrix is singular
        
        inv_params = _coords._inv_params
        inv_vec = _coords._inv_vec
        ret = computeMatrices.computeTransform(inv_params, inv_vec)
        # substitute to get both A and B matrices in terms of the same parameters
       

        vec = _coords._vec
        sub_str = dict(zip(inv_params, vec))
        ret = ret.subs(sub_str) 
        ret = sp.simplify(ret)
        
        # handle sqrt(r**2) = r
        params = _coords._params
        sub_str = map(lambda x: sp.sqrt(x**2), params)
        sub_str = dict(zip(sub_str, params))
        ret = ret.subs(sub_str)
        ret = sp.simplify(ret)
        
        return ret
        
        
# class with static methods to compute polar matrices from cartesian coordinates


class polarFromCartesian:
    
    def __init__(self, _r, _theta, _x, _y):
        self._params = sp.Array([r, theta])
        self._vec = sp.Array([self._params[0]*sp.cos(self._params[1]), self._params[0]*sp.sin(self._params[1])]) # definitions of x and y
        self._inv_params = sp.Array([_x, _y])
        self._inv_vec = sp.Array([sp.sqrt(self._inv_params[0]**2 + self._inv_params[1]**2), sp.atan2(_y, _x)])
    

class polarSqrtFromPolar:
    
    def __init__(self, _r_bar, _theta_bar, _r, _theta):
        self._params = sp.Array([_r_bar, _theta_bar])
        self._vec = sp.Array([sp.sqrt(self._params[0]), sp.sqrt(self._params[1])])
        self._inv_params = sp.Array([_r, _theta])
        self._inv_vec = sp.Array([_r**2, _theta**2])


class polar1FromPolar:
    
    def __init__(self, _r_bar, _theta_bar, _r, _theta):
        self._params = sp.Array([r_bar, theta_bar])
        self._vec = sp.Array([_r_bar + _theta_bar, _r_bar - _theta_bar])
        self._inv_params = sp.Array([_r, _theta])
        self._inv_vec = sp.Array([(_r + _theta)/2, (_r - _theta)/2])
    
  
class polarSqrt1FromPolar:
    
    def __init__(self, _r_bar, _theta_bar, _r, _theta):
        self._params = sp.Array([r_bar, theta_bar])
        self._vec = sp.Array([sp.sqrt(_r_bar) + sp.sqrt(_theta_bar), sp.sqrt(_r_bar) - sp.sqrt(_theta_bar)])
        self._inv_params = sp.Array([_r, _theta])
        self._inv_vec = sp.Array([((_r + _theta)/2)**2, ((_r - _theta)/2)**2])
    
    
class transformRecord:
    
    def __init__(self, _A, _B, _E, _W):
        self._A = _A
        self._B = _B
        self._E = _E
        self._W = _W

# develop a loop to process results given an input coordinate system

def processLoop(_coords, msg):
    
    cM = computeMatrices()
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
    return tmpRec



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
x = symbols('x') #noqa
y = symbols('y') #noqa

print('polar from cartesian matrices \n')

pFc = polarFromCartesian(r, theta, x, y)
rec = processLoop(pFc, '(polar, cartesian) = ')
mathData.insertIntoTransformTable('polar', 'cartesian', rec)

# polarSqrt from polar

r_bar = symbols('\\bar{r}') #noqa
theta_bar = symbols('\\bar{\\theta}') #noqa

E_polar = I(2)
W_polar = I(2)

print('polarSqrt from polar matrices \n')

pSqrtFp = polarSqrtFromPolar(r_bar, theta_bar, r, theta)
rec = processLoop(pSqrtFp, '(polarSqrt, polar) = ')
mathData.insertIntoTransformTable('polarSqrt', 'polar', rec)

# polar1 from polar

print('polar1 from polar\n')

p1Fp = polar1FromPolar(r_bar, theta_bar, r, theta)
rec = processLoop(p1Fp, '(polar1, polar) = ')
mathData.insertIntoTransformTable('polar1', 'polar', rec)

# polar to polarSqrt1

print('polarSqrt1 from polar\n')

pSqrt1Fp = polarSqrt1FromPolar(r_bar, theta_bar, r, theta)
rec = processLoop(pSqrt1Fp, '(polarSqrt1, polar) = ')
mathData.insertIntoTransformTable('polarSqrt1', 'polar', rec)

mathData.close()
sys.exit(0)











