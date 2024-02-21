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
            return
        
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
                
                expr = '\\frac{\\partial{\\overline{\\bf{v}}}}{\\partial{' + str(p) + '}}'
                expr1 = '\\overline{\\bf{v}}'
                expr2 = '\\overline{p}'
            else:
                expr = '\\frac{\\partial{\\bf{v}}}{\\partial{' + str(p) + '}}'
                expr1 = '\\bf{v}'
                expr2 = 'p'
                
            print(f'{str(p)} component \n')
            print(f'{expr} = {dv_dp_latex} \n')
            vDotGamma_latex = latex.convertVectorToLatex(self._vDotGamma[str(p)], False, n)
            expr = '\\bf{v} \. \\Gamma_'
            print(f'{expr}{p} = {vDotGamma_latex} \n')
            covar_p_latex = latex.convertVectorToLatex(self._covar_p[str(p)], False, n)
            expr3 = '{' + str(p) + '}'
            print(f'\\nabla_{expr3} {str(expr1)}  = {covar_p_latex} \n')
            
        
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
v_latex = latex.convertVectorToLatex(v, False, 2)
expr = '\\bf{v}'
print('vector field in polar coordinates \n')
print(f'{expr} = {v_latex} \n')

# Compute Covariant Derivative in the polar sqrt coordinate system

print('Transformation Matrices \n')

A_latex = latex.convertMatrixToLatex(A)
print(f'A = {A_latex} \n')
B_latex = latex.convertMatrixToLatex(B)
print(f'B = {B_latex} \n')

# Transform vector field to the polar sqrt coordinate systtem

v1 = np.dot(v, B)

# substitute (r, theta) coordinates with (r1, theta1) coordinates

sub_str = dict(zip(inv_params, vec))
v1 = sp.Array(v1)
v1 = v1.subs(sub_str) 
v1 = sp.simplify(v1)
v1_latex = latex.convertVectorToLatex(v1, False, 2)
expr = '\\overline{\\bf{v}}'
print('vector field in polar sqrt coordinates\n')
print( f'{expr} = {v1_latex}\n')

try:
     covar_p_prime = covariantDerivative(mathData, key)
except KeyError as e:
     print(str(e))
     sys.exit(0)

covar_p_prime.compute(v1)
print('Covariant Derivative in the polar sqrt system\n')
covar_p_prime.printVars()
    
# Compute Covariant Derivative in Polar Coordinates 

# Declare a vector field in polar coordinates

key = ('polar', 'cartesian')

try:
     covar_p = covariantDerivative(mathData, key)
except KeyError as e:
     print(str(e))
     sys.exit(0)

covar_p.compute(v)
print('Covariant Derivative in the polar system\n')
covar_p.printVars()

# Test matrix transform of covariance derivative from polar coordinates to polar sqrt coordinates

print('Test of matrix transform of the Covariant Derivative from polar to polar sqrt coordinates \n')

covar_p_matrix = covar_p.getCovarMatrix()
covar_p_matrix_latex = latex.convertMatrixToLatex((covar_p_matrix))

test = A*covar_p_matrix*B   # Perform transform on matrix representation of covariant derivative
sub_str = dict(zip(inv_params, vec))
test = test.subs(sub_str)
test = sp.simplify(test)
test_latex = latex.convertMatrixToLatex(test)
expr = 'A [\\nabla_p \\bf{v}] B'
print(f'{expr} = {test_latex}\n')


# Test equation (69)

print('Test equation (69) \n')
r_prime = symbols('\\overline{r}') #noqa
theta_prime = symbols('\\overline{\\theta}') #noqa

# r_prime component

print('r_prime compnent\n')

dB_dr_prime = sp.diff(B, r_prime)
dB_dr_prime_latex = latex.convertMatrixToLatex(dB_dr_prime)
expr = '\\frac{\\partial{B}}{\\partial{\\overline{r}}}'
print(f'{expr} = {dB_dr_prime_latex} \n')

dA_dr_prime = sp.diff(A, r_prime)
dA_dr_prime_latex = latex.convertMatrixToLatex(dA_dr_prime)
expr = '\\frac{\\partial{A}}{\\partial{\\overline{r}}}'
print(f'{expr} = {dA_dr_prime_latex} \n')

test_r_prime = dB_dr_prime + np.dot(B, np.dot(dA_dr_prime, B))
test_r_prime_latex = latex.convertMatrixToLatex(test_r_prime)
expr = '\\frac{\\partial{B}}{\\partial{\\overline{r}}} + B \\frac{\\partial{A}}{\\partial{\\overline{r}}} B'
print(f'{expr} = {test_r_prime_latex} \n')

# theta_prime component

print('theta_prime compnent\n')

dB_dTheta_prime = sp.diff(B, theta_prime)
dB_dTheta_prime_latex = latex.convertMatrixToLatex(dB_dTheta_prime)
expr = '\\frac{\\partial{B}}{\\partial{\\overline{\\theta}}}'
print(f'{expr} = {dB_dTheta_prime_latex} \n')

dA_dTheta_prime = sp.diff(A, theta_prime)
dA_dTheta_prime_latex = latex.convertMatrixToLatex(dA_dTheta_prime)
expr = '\\frac{\\partial{A}}{\\partial{\\overline{\\theta}}}'
print(f'{expr} = {dA_dTheta_prime_latex} \n')


test_theta_prime = dB_dTheta_prime + np.dot(B, np.dot(dA_dTheta_prime, B))
test_theta_prime_latex = latex.convertMatrixToLatex(test_theta_prime)
expr = '\\frac{\\partial{B}}{\\partial{\\overline{\\theta}}} + B \\frac{\\partial{A}}{\\partial{\\overline{\\theta}}} B'
print(f'{expr} = {test_theta_prime_latex} \n')