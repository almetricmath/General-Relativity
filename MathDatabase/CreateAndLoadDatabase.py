# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 18:29:33 2023

@author: Al
"""

import sympy as sp
from sympy import * #noqa
import mathDB as mathDB
import sys


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
rec = mathDB.createProcessLoop(pFc, '(polar, cartesian) = ')
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
rec = mathDB.createProcessLoop(pSqrtFp, '(polarSqrt, polar) = ')
mathData.insertIntoTransformTable('polarSqrt', 'polar', rec)
rec = mathDB.coordinateRecord(pSqrtFp)
mathData.insertIntoCoordinateTable('polarSqrt', 'polar', rec)

# polar1 from polar

print('polar1 from polar\n')

p1Fp = mathDB.polar1FromPolar(r_bar, theta_bar, r, theta)
rec = mathDB.createProcessLoop(p1Fp, '(polar1, polar) = ')
mathData.insertIntoTransformTable('polar1', 'polar', rec)
rec = mathDB.coordinateRecord(p1Fp)
mathData.insertIntoCoordinateTable('polar1', 'polar', rec)

# polar to polarSqrt1

print('polarSqrt1 from polar\n')

pSqrt1Fp = mathDB.polarSqrt1FromPolar(r_bar, theta_bar, r, theta)
rec = mathDB.createProcessLoop(pSqrt1Fp, '(polarSqrt1, polar) = ')
mathData.insertIntoTransformTable('polarSqrt1', 'polar', rec)
rec = mathDB.coordinateRecord(pSqrt1Fp)
mathData.insertIntoCoordinateTable('polarSqrt1', 'polar', rec)

mathData.close()
sys.exit(0)