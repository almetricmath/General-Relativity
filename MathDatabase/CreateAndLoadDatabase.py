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

#  cartesian to polar 

r = symbols('r') #noqa
theta = symbols('theta') #noqa
x = symbols('x') #noqa
y = symbols('y') #noqa

print('polar from cartesian matrices \n')

E_cartesian = sp.eye(2)
W_cartesian = sp.eye(2)

pFc = mathDB.polarFromCartesian(r, theta, x, y)
table = mathDB.createProcessLoop(pFc, E_cartesian, W_cartesian, '(polar, cartesian) = ', 'Output.txt') 
mathData.insertIntoTransformTable('polar', 'cartesian', table)
rec = mathDB.coordinateRecord(pFc)
mathData.insertIntoCoordinateTable('polar', 'cartesian', rec)

# polarSqrt from polar

E_polar = table._E
W_polar = table._W

r_bar = symbols('\\overline{r}') #noqa
theta_bar = symbols('\\overline{\\theta}') #noqa


print('polarSqrt from polar matrices \n')

pSqrtFp = mathDB.polarSqrtFromPolar(r_bar, theta_bar, r, theta)
rec = mathDB.createProcessLoop(pSqrtFp, E_polar, W_polar, '(polarSqrt, cartesian) = ', 'Output.txt')
mathData.insertIntoTransformTable('polarSqrt', 'polar', rec)
rec = mathDB.coordinateRecord(pSqrtFp)
mathData.insertIntoCoordinateTable('polarSqrt', 'polar', rec)

# polar1 from polar

print('polar1 from polar\n')

p1Fp = mathDB.polar1FromPolar(r_bar, theta_bar, r, theta)
rec = mathDB.createProcessLoop(p1Fp, E_polar, W_polar, '(polar1, cartesian) = ', 'Output.txt') 
mathData.insertIntoTransformTable('polar1', 'polar', rec)
rec = mathDB.coordinateRecord(p1Fp)
mathData.insertIntoCoordinateTable('polar1', 'polar', rec)

# polar to polarSqrt1

print('polarSqrt1 from polar\n')

pSqrt1Fp = mathDB.polarSqrt1FromPolar(r_bar, theta_bar, r, theta)
rec = mathDB.createProcessLoop(pSqrt1Fp, E_polar, W_polar, '(polarSqrt1, cartesian) = ', 'Output.txt') 
mathData.insertIntoTransformTable('polarSqrt1', 'polar', rec)
rec = mathDB.coordinateRecord(pSqrt1Fp)
mathData.insertIntoCoordinateTable('polarSqrt1', 'polar', rec)

mathData.close()
sys.exit(0)