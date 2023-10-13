# -*- coding: utf-8 -*-
"""
Created on Tue May 16 17:09:32 2023

@author: Al
"""

#import sympy as sp
from sympy import * #noqa
#from sympy.printing.latex import latex as sp_latex
import mathDB


mathData = mathDB.mathDB('math.db')
latex = mathDB.convertToLatex()

transformTable = mathData.getTransformTable()

# get transforms from cartesian to the various basis sets

key = ('polar', 'cartesian')
if key in transformTable:
    transRec = transformTable[key] 
    print(str(key), 'transform matrices')
    transRec.printRecord(key)   
else:
    print(str(key), " not in database\n")    

 
# print coordinate class

coordTable = mathData.getCoordinateTable()
coord = coordTable[key]
print(str(key), 'coordinates\n')
coord.printRecord()

key = ('polarSqrt', 'polar')
if key in transformTable:
    transRec = transformTable[key] 
    print(str(key), 'transform matrices\n')
    transRec.printRecord(key)       
else:
    print(key, " not in database")

coord = coordTable[key]
print(str(key), 'coordinates\n')
coord.printRecord()

key = ('polar1', 'polar')
if key in transformTable:
    transRec = transformTable[key] 
    print(str(key), 'transform matrices')
    transRec.printRecord(key)       
else:
    print(key, " not in database")
    
coord = coordTable[key]
print(str(key), 'coordinates\n')
coord.printRecord()

key = ('polarSqrt1', 'polar')
if key in transformTable:
   transRec = transformTable[key] 
   print(str(key), 'transform matrices')
   transRec.printRecord(key)         
else:
    print(key, " not in database")

coord = coordTable[key]
print(str(key), 'coordinates\n')
coord.printRecord()

mathData.close()










