# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 18:08:28 2022

@author: Al Bernstein
"""

import numpy as np
from matplotlib import pyplot as plt
import draw as dw
_dw = dw.draw()

class CoordinateGrid2D:
    
    def __init__(self, _xlow, _xhigh, _ylow, _yhigh,_title):
        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111)
        self._ax.set_xlim(_xlow, _xhigh)
        self._ax.set_ylim(_ylow, _yhigh)
        self._ax.set_xlabel('x',fontsize=20)
        self._ax.set_ylabel('y',rotation=0, fontsize=20)
        plt.title(_title, fontsize = 20)
        self._coordTrans =  coordinateTransforms()
      
    def setXvals(self, _x:np.ndarray):
        self._x_input = _x
    
    def setYvals(self, _y:np.ndarray):
        self._y_input = _y
      
    def setPlotTitle(self, _title):
        plt.title(_title, fontsize = 20)
    
    def createRotatedGrid(self, xfn, yfn, _angle, _color):
       
        y_label = []
        y_pos = []
        x_label = []
        x_pos = []
                
        # hold y value constant and generate x values
        
        for i in self._y_input:
            x_tmp = xfn(coords = (self._x_input, [i]))
            y_tmp = yfn(coords = (self._x_input, [i]))
            x_array, y_array = self._coordTrans.rotateCoords(x_tmp, y_tmp, _angle)
            
            # compute endpoints
            
            end_dx = x_array[-1] - x_array[-2]
            end_dy = y_array[-1] - y_array[-2]
            x_end = x_array[-1] + end_dx
            y_end = y_array[-1] + end_dy
            
            self.drawGridPoints(x_array, y_array,_color)
            self._ax.plot(x_array, y_array,c=_color,label = str(i))
            y_label.append(i)
            y_pos.append((x_end, y_end))
            _dw.drawVecXY( [x_array[-1], y_array[-1]], [x_end, y_end],self. _ax, _color)
           
        for i in range(len(y_label)):
             plt.annotate(r'$\bar{y} $ =%.2f'%y_label[i],y_pos[i] ,xytext=(0, 0), textcoords="offset pixels", fontsize = 14) 
    
        # hold x constant and generate y values

        y_label = []
        y_pos = []
        x_label = []
        x_pos = []
        

        for i in self._x_input:
            x_tmp = xfn(coords = ([i], self._y_input))
            y_tmp = yfn(coords = ([i], self._y_input))
            x_array, y_array = self._coordTrans.rotateCoords(x_tmp, y_tmp, _angle)
            
            # compute endpoints
            
            end_dx = x_array[-1] - x_array[-2]
            end_dy = y_array[-1] - y_array[-2]
            x_end = x_array[-1] + end_dx
            y_end = y_array[-1] + end_dy
        
            self._ax.plot(x_array, y_array,c='black',label = str(i))
            #i_rot = self._coordTrans.rotateCoords(i, 0, _angle)
            x_label.append(i)
            x_pos.append((x_end, y_end))   
            _dw.drawVecXY( [x_array[-1], y_array[-1]], [x_end, y_end],self. _ax, 'black')
        
        for i in range(len(x_label)): 
            plt.annotate(r'$\bar{x}$ = %.2f'%x_label[i], x_pos[i], xytext=(-20, 0), textcoords="offset pixels", fontsize = 14)

    def createGrid(self, xfn, yfn, x_color, y_color, _x_line, _y_line, _barFlag):
        y_label = set({})
        x_label = set({})
        
        # hold y value constant and generate x values
        
        y_label = []
        y_pos = []
        x_label = []
        x_pos = []
        
        for i in self._y_input:
            x_array = xfn(coords = (self._x_input, [i]))
            y_array = yfn(coords = (self._x_input, [i]))
            
            # compute endpoints
            
            end_dx = x_array[-1] - x_array[-2]
            end_dy = y_array[-1] - y_array[-2]
            x_end = x_array[-1] + end_dx
            y_end = y_array[-1] + end_dy
            
            self.drawGridPoints(x_array, y_array, 'black')
            self._ax.plot(x_array, y_array,c=y_color,label = str(i))
            y_label.append(i)
            y_pos.append((x_end, y_end))
            _dw.drawVecXY( [x_array[-1], y_array[-1]], [x_end, y_end],self. _ax, y_color)
           
            if _barFlag:
                s = 'r\'$\bar{'+str(_y_line) + '}$'
            else:
                s = str(_y_line)           
                
        for i in range(len(y_label)):
             plt.annotate(s + '=%.2f'%y_label[i],y_pos[i] ,xytext=(0, -20), textcoords="offset pixels", fontsize = 14) 
    
        # hold x constant and generate y values
        
        for i in self._x_input:
            x_array = xfn(coords = ([i], self._y_input))
            y_array = yfn(coords = ([i], self._y_input))
            
            # compute endpoints
            
            end_dx = x_array[-1] - x_array[-2]
            end_dy = y_array[-1] - y_array[-2]
            x_end = x_array[-1] + end_dx
            y_end = y_array[-1] + end_dy
            
            self._ax.plot(x_array, y_array,c=x_color, label = str(i))
            x_label.append(i)
            x_pos.append((x_end, y_end))
            _dw.drawVecXY( [x_array[-1], y_array[-1]], [x_end, y_end],self. _ax, x_color)
        
        if _barFlag:
            s = 'r\'$\bar{' + str(_x_line) + '}$'
        else:
            s = str(_x_line)
        
        for i in range(len(x_label)): 
            plt.annotate(s+'= %.2f'%x_label[i], x_pos[i], xytext=(-20, 0), textcoords="offset pixels", fontsize = 14)
              
            
    def drawGridPoints(self,  _x_array, _y_array, _color):
        _dw.drawXYPoint([_x_array, _y_array], self._ax, _color, 40)
            
            
class coordinateTransforms:
    
        # define rectangular coordinate transform
        
        # Rectangular Grid
        
        def x_rect(self, **kwargs):
            _coords = kwargs['coords']
            _x, _y = _coords
            if len(_x) < len(_y):
                return np.full(len(_y),_x)
            return _x
        
        
        def y_rect(self, **kwargs):
            _coords = kwargs['coords']
            _x, _y = _coords
            if len(_y) < len(_x):
                return np.full(len(_x),_y)
            return _y
    
        # inputs x and y points and an  angle in radians
    
        def rotateCoords(self,  _x_array, _y_array, _angle):
            
            # this displays the x_prime, y_prime grid lines in (x,y) coordinates
            
            x_ret = _x_array*np.cos(_angle) - _y_array*np.sin(_angle)
            y_ret = _x_array*np.sin(_angle) + _y_array*np.cos(_angle)
            return (x_ret, y_ret)
       
        def x_parabolic(self,  **kwargs):
            _coords = kwargs['coords']
            _rho, _tau = _coords
            
            ret = _rho*_tau
            return ret
         
        def y_parabolic(self,  **kwargs):
            _coords = kwargs['coords']
            _rho, _tau = _coords
            
            if len(_tau) < len(_rho):
                _tau =  np.full(len(_rho),_tau)
            else:
                _rho = np.full(len(_tau),_rho)
    
            ret = (1/2)*(_tau*_tau - _rho*_rho)
            return ret

grid = CoordinateGrid2D(-30, 30, -30, 30, "Parabolic Grid") 
transform = coordinateTransforms()
x =  np.linspace(-5,5,10) 
y = np.linspace(-5,5,10)
grid.setXvals(x)
grid.setYvals(y)
#grid.createRotatedGrid(transform.x_rect, transform.y_rect, np.pi/3, 'black')
grid.createGrid(transform.x_parabolic, transform.y_parabolic, 'green', 'red','rho', 'tau', False)
