# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 15:58:37 2024

@author: Al
"""

from sympy import * #noqa
import sympy as sp
import numpy as np
import mathDB
import sys


class christoffelSymbols():
    
    # This class computes Christoffel Symbols of the Second Kind
    
    def __init__(self, _data, _key):
        # _data is mathDB class
        transformTable =  _data.getTransformTable()
        
        if _key in transformTable:
            self._transRec = transformTable[_key]
        else:
            raise KeyError(f'{key} not in transformTable\n')
            
        coordinateTable = _data.getCoordinateTable()
        if _key in coordinateTable:
            self._coords = coordinateTable[_key]
        else:
            raise KeyError(f'{key} not in coordTable\n')
            
        # set up variables for the results
        
        self._Gamma_p = {}

    def compute(self):
        
        params = self._coords._params
        E = self._transRec._E
        W = self._transRec._W
         
        for p in params:
            dE_dp = sp.diff(E, p)
            dE_dp = sp.simplify(dE_dp)
            Gamma_p = dE_dp*sp.transpose(W)
            Gamma_p = sp.simplify(Gamma_p)
            self._Gamma_p[str(p)] = Gamma_p 
            
    def getGamma(self):
        return self._Gamma_p
        
class covariantDerivative:
    
    def __init__(self, _data, _key):
        # _data is mathDB class
        # only need coordinate table
       
        coordinateTable = _data.getCoordinateTable()
        if _key in coordinateTable:
            self._coords = coordinateTable[_key]
        else:
            raise KeyError(f'{key} not in coordTable\n')
        
        # compute Christoffel Symbols
        
        try:
            Gamma = christoffelSymbols(mathData, key)
        except KeyError as e:
            print(str(e))
            
        # create variables for the results
        
        Gamma.compute()
        self._Gamma_vals = Gamma.getGamma() 
        self._dv_dp = {}
        self._vDotGamma = {} 
        self._covar_p = {}
        
    def compute(self, _v_in):
        
        params = self._coords._params

        for p in params:
            self._dv_dp[str(p)] = sp.diff(_v_in, p)
            self._vDotGamma[str(p)] = np.dot(_v_in, self._Gamma_vals[str(p)])
            self._covar_p[str(p)] = self._dv_dp[str(p)] + self._vDotGamma[str(p)]
        
        # put the row vectors togeter to create a covariant derivative matrix
        
        matrix = []
        for p in self._covar_p.values():
            matrix.append(p)
            
        self._covar_matrix = sp.Matrix(matrix)
    
    def printVars(self):
        
        params = self._coords._params
        n =len(params)
        
        for p in params:
            dv_dp_latex = latex.convertVectorToLatex(self._dv_dp[str(p)], False, n)
            if '\overline' in str(p):
                expr = '\\frac{d\\overline{\\bf{v}}}{d' + str(p) + '}'
                expr1 = '\\overline{\\bf{v}}'
                expr2 = '\\overline{p}'
            else:
                expr = '\\frac{d\\bf{v}}{d' + str(p) + '}'
                expr1 = '\\bf{v}'
                expr2 = 'p'
                
            print(f'{str(p)} component \n')
            print(f'{expr} = {dv_dp_latex} \n')
            vDotGamma_latex = latex.convertVectorToLatex(self._vDotGamma[str(p)], False, n)
            print(f'v \. \\Gamma_{p} = {vDotGamma_latex} \n')
            covar_p_latex = latex.convertVectorToLatex(self._covar_p[str(p)], False, n)
            print(f'\\nabla_{str(p)} {str(expr1)}  = {covar_p_latex} \n')
            
        
        covar_matrix_latex = latex.convertMatrixToLatex(self._covar_matrix)
        print(f'\\nabla_{expr2} {expr1} = {covar_matrix_latex} \n')

    def getDv_dp(self):
        return self._dv_dp
    
    def getCovar_p(self):
        return self._covar_p
    
    def getCovarMatrix(self):
        return self._covar_matrix
        

mathData = mathDB.mathDB('math.db')
latex = mathDB.convertToLatex()

# Compute Covariant Derivative in Polar Sqrt Coordinates 

transformTable = mathData.getTransformTable()

# get polarSqrt from polar coordinates 

key = ('polarSqrt', 'polar')
if key in transformTable:
    transRec = transformTable[key] 
    A = transRec._A
    B = transRec._B
else:
    print(f'{key} not in transformTable\n')

coordTable = mathData.getCoordinateTable()
if key in coordTable:
    coords = coordTable[key]
else:
    print(f'{key} not in coordTable\n')


vec = coords._vec
inv_params = coords._inv_params

# Declare a vector field in polar coordinates

r = symbols('r') #noqa
theta = symbols('\\theta') #noqa

v = sp.Array([r**3*theta, r*theta**2])

# Compute Covariant Derivative in the polar sqrt coordinate system

# Transfor vector field to the polar sqrt coordinate systtem

v1 = np.dot(v, B)

# substitute (r, theta) coordinates with (r1, theta1) coordinates

sub_str = dict(zip(inv_params, vec))
v1 = sp.Array(v1)
v1 = v1.subs(sub_str) 
v1 = sp.simplify(v1)
v1_latex = latex.convertVectorToLatex(v1, False, 2)
expr = '\\overline{\\bf v}'
print( f'{expr} = {v1_latex}\n')

try:
     covar_p_prime = covariantDerivative(mathData, key)
except KeyError as e:
     print(str(e))
     sys.exit(-1)

covar_p_prime.compute(v1)
covar_p_prime.printVars()
    
# Compute Covariant Derivative in Polar Coordinates 

# Declare a vector field in polar coordinates

key = ('polar', 'cartesian')
try:
     covar_p = covariantDerivative(mathData, key)
except KeyError as e:
     print(str(e))
     sys.exit(-1)

covar_p.compute(v)
covar_p.printVars()
# Test transform of covariance derivative from polar coordinates to polar sqrt coordinates

covar_p_matrix = covar_p.getCovarMatrix()

test = A*covar_p_matrix*B
sub_str = dict(zip(inv_params, vec))
test = test.subs(sub_str)
test = sp.simplify(test)
test_latex = latex.convertMatrixToLatex(test)
print(f'test = {test_latex}\n')





