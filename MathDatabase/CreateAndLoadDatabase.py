# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:29:33 2023

@author: Al
"""

import sympy as sp
from sympy import * #noqa
import mathDB as mathDB
import sys


    

# develop a loop to process results given an input coordinate system

def processLoop(_coords, msg):
    
    latex = mathDB.convertToLatex()
    
    cM = mathDB.computeMatrices()
    A = cM.A_matrix(_coords)
    A_latex = latex.convertMatrixToLatex(A)
    print('A', msg, A_latex, '\n')
    B = cM.B_matrix(_coords)
    B_latex = latex.convertMatrixToLatex(B)
    print('B', msg, B_latex, '\n')

    E = mathDB.I(2)
    W = mathDB.I(2)
    E_bar = A*E
    E_bar_latex = latex.convertMatrixToLatex(E_bar)
    print('E', msg, E_bar_latex, '\n')
    W_bar = sp.transpose(B)*W
    W_bar_latex = latex.convertMatrixToLatex(W_bar)
    print('W', msg,  W_bar_latex, '\n')

    tmpRec = mathDB.transformRecord(A, B, E_bar, W_bar)
    return tmpRec



mathData = mathDB.mathDB('math.db')


# cartesian to cartesian

A_cart = mathDB.I(2)
B_cart = mathDB.I(2)
E_cart = mathDB.I(2)
W_cart = mathDB.I(2)

tmpRec = mathDB.transformRecord(A_cart, B_cart, E_cart, W_cart)
mathData.insertIntoTransformTable('cartesian', 'cartesian', tmpRec)

# polar to polar

A_polar = mathDB.I(2)
B_polar = mathDB.I(2)
E_polar = mathDB.I(2)
W_polar = mathDB.I(2)

tmpRec = mathDB.transformRecord(A_polar, B_polar, E_polar, W_polar)
mathData.insertIntoTransformTable('polar', 'polar', tmpRec)

#  cartesian to polar 

r = symbols('r') #noqa
theta = symbols('theta') #noqa
x = symbols('x') #noqa
y = symbols('y') #noqa

print('polar from cartesian matrices \n')

pFc = mathDB.polarFromCartesian(r, theta, x, y)
rec = processLoop(pFc, '(polar, cartesian) = ')
mathData.insertIntoTransformTable('polar', 'cartesian', rec)
rec = mathDB.coordinateRecord(pFc)
mathData.insertIntoCoordinateTable('polar', 'cartesian', rec)

# polarSqrt from polar

r_bar = symbols('\\bar{r}') #noqa
theta_bar = symbols('\\bar{\\theta}') #noqa

E_polar = mathDB.I(2)
W_polar = mathDB.I(2)

print('polarSqrt from polar matrices \n')

pSqrtFp = mathDB.polarSqrtFromPolar(r_bar, theta_bar, r, theta)
rec = processLoop(pSqrtFp, '(polarSqrt, polar) = ')
mathData.insertIntoTransformTable('polarSqrt', 'polar', rec)
rec = mathDB.coordinateRecord(pSqrtFp)
mathData.insertIntoCoordinateTable('polarSqrt', 'polar', rec)

# polar1 from polar

print('polar1 from polar\n')

p1Fp = mathDB.polar1FromPolar(r_bar, theta_bar, r, theta)
rec = processLoop(p1Fp, '(polar1, polar) = ')
mathData.insertIntoTransformTable('polar1', 'polar', rec)
rec = mathDB.coordinateRecord(p1Fp)
mathData.insertIntoCoordinateTable('polar1', 'polar', rec)

# polar to polarSqrt1

print('polarSqrt1 from polar\n')

pSqrt1Fp = mathDB.polarSqrt1FromPolar(r_bar, theta_bar, r, theta)
rec = processLoop(pSqrt1Fp, '(polarSqrt1, polar) = ')
mathData.insertIntoTransformTable('polarSqrt1', 'polar', rec)
rec = mathDB.coordinateRecord(pSqrt1Fp)
mathData.insertIntoCoordinateTable('polarSqrt1', 'polar', rec)

mathData.close()
sys.exit(0)











