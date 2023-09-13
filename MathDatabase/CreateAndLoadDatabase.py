# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:29:33 2023

@author: Al
"""

import sympy as sp
from sympy import * #noqa
import mathDB as mathDB
import sys


def I(n):
    return sp.eye(n)

# class to compute symolic natrices

class computeMatrices:
    
    def __init__(self):
        self._A_matrix = sp.Matrix(0, 0, [])
        self._B_matrix = sp.Matrix(0, 0, [])
        
    def A_matrix(self, params: sp.Array, vec: sp.Array):
        
        _n = len(params)
        
        for i in range(_n):
            self._A_matrix = self._A_matrix.row_insert(i, sp.Matrix([sp.diff(vec, params[i])]))
        
        return self._A_matrix
        

    def B_matrix(self, params: sp.Array, vec: sp.Array):
        
        if self._A_matrix.shape == (0, 0):
            self.A_matrix(params, vec)
        else:
            self._B_matrix = sp.trigsimp(self._A_matrix.inv())
        
        return self._B_matrix
        
        
            
    
# class with static methods to compute polar matrices from cartesian coordinates


class polarFromCartesian:
    
    def __init__(self, r, theta):
        self._params = sp.Array([r, theta])
        self._vec = sp.Array([self._params[0]*sp.cos(self._params[1]), self._params[0]*sp.sin(self._params[1])]) # definitions of x and y

    
# # class with static methods to compute polar sqrt matrices from polar coordinates

class polarSqrtFromPolar:
    
    @staticmethod
    def A_matrix(_r1, _theta1):
        ret = Matrix([[1/(2*sp.sqrt(_r1)),0],[0,1/(2*sp.sqrt(_theta1)) ]]) #noqa
        return ret 
    @staticmethod 
    def B_matrix(_r1, _theta1):
        ret = Matrix([[2*sp.sqrt(_r1), 0],[0, 2*sp.sqrt(_theta1)]]) # noqa
        return ret

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

# cartesian to cartesian

A = I(2)
B = I(2)
E_cart = I(2)
W_cart = I(2)

tmpRec = transformRecord(A, B, E_cart, W_cart)
mathData.insertIntoTransformTable('cartesian', 'cartesian', tmpRec)

# polar to polar

A = I(2)
B = I(2)
E_polar = I(2)
W_polar = I(2)


tmpRec = transformRecord(A, B, E_polar, W_polar)
mathData.insertIntoTransformTable('polar', 'polar', tmpRec)



r = symbols('r') #noqa
theta = symbols('theta') #noqa

pFc = polarFromCartesian(r, theta)
cM = computeMatrices()
A = cM.A_matrix(pFc._params , pFc._vec)
B = cM.B_matrix(pFc._params , pFc._vec)
E_cart = I(2)
W_cart = I(2)
E_polar = A*E_cart
W_polar = sp.transpose(B)*W_cart
tmpRec = transformRecord(A, B, E_polar, W_polar)
mathData.insertIntoTransformTable('polar', 'cartesian', tmpRec)
sys.exit(0)

'''
mathData = mathDB.mathDB('math.db')



# cartesian to polar

r = symbols('r') #noqa
theta = symbols('theta') #noqa

cM = computeMatrices() = polarFromCartesian.A_matrix(r, theta)
E_polar = A*E_cart
B = polarFromCartesian.B_matrix(r, theta)
W_polar = sp.transpose(B)*W_cart

tmpRec = transformRecord(A, B, E_polar, W_polar)
mathData.insertIntoTransformTable('polar', 'cartesian', tmpRec)

# polar to polarsqrt

r_bar = symbols('\\bar{r}') #noqa
theta_bar = symbols('\\bar{\\theta}') #noqa
E_polar = I(2)
W_polar = I(2)

A = polarSqrtFromPolar.A_matrix(r_bar, theta_bar)
E_polarSqrt = A*E_polar
B = polarSqrtFromPolar.B_matrix(r_bar, theta_bar)
W_polarSqrt = sp.transpose(B)*W_polar

tmpRec = transformRecord(A, B, E_polarSqrt, W_polarSqrt)
mathData.insertIntoTransformTable('polarsqrt', 'polar', tmpRec)

# polar to polar1

E_polar = I(2)
W_polar = I(2)

A = polar1FromPolar.A_matrix(r_bar, theta_bar)
E_polar1 = A*E_polar
B = polar1FromPolar.B_matrix(r_bar, theta_bar)
W_polar1 = sp.transpose(B)*W_polar

tmpRec = transformRecord(A, B, E_polar1, W_polar1)
mathData.insertIntoTransformTable('polar1', 'polar', tmpRec)

# polar to polarSqrt1

E_polar = I(2)
W_polar = I(2)

A = polarSqrt1FromPolar.A_matrix(r_bar, theta_bar)
E_polar1 = A*E_polar
B = polarSqrt1FromPolar.B_matrix(r_bar, theta_bar)
W_polar1 = sp.transpose(B)*W_polar

tmpRec = transformRecord(A, B, E_polar1, W_polar1)
mathData.insertIntoTransformTable('polarSqrt1', 'polar', tmpRec)
'''
mathData.close()

