import numpy as np
import draw as dw
import GreatCircle as gc
import coordUtils as cu
from matplotlib import pyplot as plt
from enum import Enum


class switch(Enum):
     Cartesian = 1
     ThetaPhi = 2

class sp_angle(Enum):
    theta = 0
    phi = 1


class parallelTransport:
    
    def __init__(self):
        
        self._dw = dw.draw()
        self._gc = gc.greatCircle()
        self._cu = cu.coordUtils()
    
    def stepXYZToThetaPhi(self, _u0_sp, _u1_sp, _u0_next_sp, _ax):
        
        # input vectors in spherical coordinates
        
        # draw next point on path 
        self._dw.drawThetaPhiPoint(_u0_sp, _ax, 'red')
        
        # draw an arc between u0_next and u1
        a_coords = self._gc.arc(_u0_next_sp, _u1_sp, 1.0, 7)
        # a_coords in spherical coordinates
        self._dw.drawThetaPhiArc(a_coords[0], _ax, 1, 'red')
        
        # plot midpoint of arc between u0_next and u1
        midpoint = a_coords[1]
        self._dw.drawThetaPhiPoint(midpoint, _ax, 'blue')
        
        a2_coords = self._gc.twice_arc(_u0_sp, midpoint, 1.0, 14)
        
        self._dw.drawThetaPhiArc(a2_coords, _ax, 1, 'red')
        
        # draw parallel transported vector

        u1_new =  a2_coords[-1]
        
        self._dw.drawVecThetaPhi(_u0_next_sp, u1_new, _ax)
        
        return u1_new 
        
        
    def step(self, _u0_sp, _u1_sp, _u0_next_sp, _ax, _switch):

        # input vectors in spherical coordinates (r, theta, phi)
        
        # draw next point on path 
        if _switch == switch.Cartesian:
            self._dw.drawXYZPoint(_u0_next_sp, _ax, 'black')
        else:
            self._dw.drawThetaPhiPoint(_u0_next_sp, _ax, 'black')
    
        # draw an arc between u0_next and u1
        
        a_coords = self._gc.arcTheta(_u0_next_sp, _u1_sp, 7)
        
        if _switch == switch.Cartesian:
            self._dw.drawXYZArc(a_coords, _ax, 1, 'red')
        else:
            self._dw.drawThetaPhiArc(a_coords, _ax, 1, 'red')
            
            
        # plot midpoint of arc between u0_next and u1
            
        midpoint = a_coords[3]
        
        if _switch == switch.Cartesian:
            self._dw.drawXYZPoint(midpoint, _ax, 'blue')
        else:
            self._dw.drawThetaPhiPoint(midpoint, _ax, 'blue')

        
        a2_coords = self._gc.twice_arcTheta(_u0_sp, midpoint, 14)
        if _switch == switch.Cartesian:
            self._dw.drawXYZArc(a2_coords, _ax, 1, 'red')
        else:
            self._dw.drawThetaPhiArc(a2_coords, _ax, 1, 'red')
        
        # draw parallel transported vector
        
        u1_new =  a2_coords[-1]
        if _switch == switch.Cartesian:
            self._dw.drawVecXYZ(_u0_next_sp,u1_new, _ax)
        else:
            self._dw.drawVecThetaPhi(_u0_next_sp, u1_new, _ax)
            
        
        return u1_new 
    
    def plotChunk(self, _u0_sp, _u1_sp, _phi, _theta, _ax):
    
        # input vectors in Spherical Coordinates
        
        if len(_phi) != len(_theta):
            raise Exception("length not the same")
        
        #_u0 = self._cu.SphericalToCartesian(_u0_sp)
        #_u1 = self._cu.SphericalToCartesian(_u1_sp)
        self._dw.drawVecThetaPhi(_u0_sp, _u1_sp, _ax)
        
        u0 = [_u0_sp[1:]]
        u1 = [_u1_sp[1:]]
        
        for i in range(1, len(_phi)):
            u0_next_sp = np.array([1, _theta[i], _phi[i]])
            #u0_next = self._cu.SphericalToCartesian(u0_next_sp)
            
            u1_new = self.stepXYZToThetaPhi(_u0_sp, _u1_sp, u0_next_sp, _ax)
            _u0_sp = u0_next_sp
            _u1_sp = u1_new
            
            #u0_sp = self._cu.CartesianToSpherical(_u0)
            #u1_sp = self._cu.CartesianToSpherical(_u1)
            u0.append(_u0_sp[1:])
            u1.append(_u1_sp[1:])
            
    
        return [u0, u1]



    # analytic solution for a constant longitude path
    
    def analyticThetaLongitude(self, v0_theta, v0_phi, theta_0, _theta):
        
        # inputs are spherical coordinates
        
        ret = [ v0_theta, v0_phi*np.sin(theta_0)/np.sin(_theta) ]
        
        return ret
    
    def analyticThetaLatitude(self, v0_theta, v0_phi, theta_0, phi):
    
        ret = 0
        k = np.cos(theta_0)
        
        tmp_theta = v0_theta*np.cos(k*phi) +  np.sin(theta_0)*v0_phi*np.sin(k*phi)
        tmp_phi = v0_phi*np.cos(k*phi) - np.sin(k*phi)*v0_theta/np.sin(theta_0)
        tmp = np.array([tmp_theta, tmp_phi])
        ret = tmp    
        
        return ret

    
# the metric for the 2-D sphere
        
def g(_theta):
    return np.array([[1, 0], [0, np.sin(_theta)]])



fig = plt.figure()
ax1 = fig.add_subplot(111)# initial vector
_u0_sp = np.array([1, np.pi/6, 0])
du = 0.05
_u1_sp = _u0_sp + np.array([0, -du, 0])
#_u1_sp = _u0_sp + np.array([0, 0, du])
_theta = [np.pi/6]*50
_phi = np.linspace(0, 2*np.pi-0.01, 50)
#plt.xlim(-0.01, 0.06)
plt.ylim(0.58, 0.47)
plt.xlabel(r'$\varphi$', fontsize=40, ha='left')
ax1.xaxis.set_label_coords(1.0, 0.0)
ax1.set_ylabel(r'$\theta$', rotation=0, ha='left', fontsize=40, labelpad = 40)
plt.rcParams['xtick.labelsize']=30
plt.rcParams['ytick.labelsize']=30
#plt.ylabel(r'$\theta$', fontsize=20)
plt.title('Parallel Transport of a Contravariant Vector on a 30 Degree Latitude Circle', fontsize = 40)
txt = 'Figure 8'
fig.text(.5, 0.01, txt, ha='center', fontsize = 40)

pt = parallelTransport()

v_theta = []

theta_0 = _theta[0]
vec_phi = []
vec_theta = []
vec_length = []
angle = []
tmp_vec = []

u0, u1 = pt.plotChunk(_u0_sp, _u1_sp, _phi, _theta, ax1)
plt.show()

v = []

for i in range(len(u0)):
    v.append(u1[i] - u0[i])

v_an = []

for th in _phi:
    tmp = pt.analyticThetaLatitude(-du, 0, np.pi/6, th)
    v_an.append(tmp)
    
error = []
for i in range(len(v_an)):
    error.append(v_an[i] - v[i])

# output table of values as a file
    
with open('results.dat','w') as f:
    f.write('phi\t analytic v_phi\t analytic v_theta\t Schilds v_phi\t Schilds v_theta\t \
            error_phi\t error_theta\n')
    for i in range(len(_phi)):
        f.write(str(_phi[i]) + '\t' + str(v_an[i][1]) + '\t' + str(v_an[i][0]) + '\t' + \
              str(v[i][1]) + '\t' + str(v[i][0]) + '\t' + str(error[i][1]) + \
              '\t' + str(error[i][0]) + '\n')
    f.close()

