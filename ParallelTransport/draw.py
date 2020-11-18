import coordUtils as cu
import numpy as np

class draw:
    
    def __init__(self):
        
        self._cu = cu.coordUtils()

    def drawThetaPhiPoint(self, _v_xyz, _axis, _color):
        
        # _v_xyz in cartesian coordinates (x, y, z)
        
        _v_sp = self._cu.CartesianToSpherical(_v_xyz)
        
        _axis.scatter(_v_sp[2], _v_sp[1], c = _color, s = 75)

    def drawXYZPoint(self, _v_sp, _axis, _color):
        
    # _v_sp in spherical coordinates (r, theta, phi)

        vec = self._cu.SphericalToCartesian(_v_sp)
        _axis.scatter(vec[0], vec[1], vec[2], c = _color, s = 75)
        
        
    def drawThetaPhiArc(self, _vecs_xyz, _axis, _lwidth, _color):
    
        # _vecs in xyz coordinates
        # transform to spherical coordinates
    
        _vecs_sp = []
        for vec in _vecs_xyz:
            tmp = self._cu.CartesianToSpherical(vec)
            _vecs_sp.append(tmp)
                
        vecs_t = np.transpose(_vecs_sp)
        
        # plot theta, phi points
        
        _axis.plot(vecs_t[2], vecs_t[1], color = _color, linewidth = _lwidth, linestyle='dotted')
        # plot endpoints
        self.drawThetaPhiPoint(_vecs_xyz[0], _axis, _color)
        self.drawThetaPhiPoint(_vecs_xyz[-1], _axis, _color)

    def drawXYZArc(self, _vecs_sp, _axis, _lwidth, _color):
        
        # _vecs in Spherical Coordinates
        
        vecs_xyz = self._cu.SphericalToCartesianVec(_vecs_sp)
        vecs_t = np.transpose(vecs_xyz)
        
        #plot xyz points
        
        _axis.plot(vecs_t[0], vecs_t[1], vecs_t[2], color = _color, linewidth = _lwidth, linestyle='dotted')
        self.drawXYZPoint(_vecs_sp[0], _axis, _color)
        self.drawXYZPoint(_vecs_sp[-1], _axis, _color)
        
    def drawVecThetaPhi(self, _u0_xyz, _u1_xyz, _axis):
    
        # _u0_xyz and _u1_xyz are in cartesian coordinates
        
        _u0_sp = self._cu.CartesianToSpherical(_u0_xyz)
        _u1_sp = self._cu.CartesianToSpherical(_u1_xyz)
         
        _axis.annotate(s='', xy=(_u1_sp[2],_u1_sp[1]), xytext=(_u0_sp[2],_u0_sp[1]), arrowprops=dict(arrowstyle='-|>, head_width=0.5, head_length=1', lw = 5))

    def drawVecXYZ(self, _u0, _u1, _axis):
    
        # _u0 and _u1 are in spherical coordinates
        
        u0 = self._cu.SphericalToCartesian(_u0)
        u1 = self._cu.SphericalToCartesian(_u1)
        
        x = [u0[0], u1[0]]
        y = [u0[1], u1[1]]
        z = [u0[2], u1[2]]
        
        #_axis.annotate(s='', xy=(_u1_sp[2],_u1_sp[1]), xytext=(_u0_sp[2],_u0_sp[1]), arrowprops=dict(arrowstyle='-|>, head_width=0.5, head_length=1', lw = 5))
        _axis.plot(x, y, z, color = 'black', linewidth = 5)

    # generate a piece of sphere or sphere centered at origin
        
    def sphere(self, r, theta_0, theta_1, phi_0, phi_1, num):
    
        sphere = []
        
        theta = np.linspace(theta_0, theta_1, num)
        phi = np.linspace(phi_0, phi_1, num)
        
        for i in theta:
            for j in phi:
                tmp = self._cu.SphericalToCartesian(np.array([r, i, j]))
                sphere.append(tmp)
        
        x = [_x[0] for _x in sphere]
        y = [_x[1] for _x in sphere]
        z = [_x[2] for _x in sphere]
        
        return [x, y, z]
    

    def latitudeCircle(self, _r, _theta_0, _num):
        
        _phi = 0
        dphi = 2*np.pi/(_num - 1)
        circle = []
        for i in range(_num):
            tmp = self._cu.SphericalToCartesian(np.array([_r, _theta_0, _phi]))
            circle.append(tmp)
            _phi += dphi
            
        x = [_x[0] for _x in circle]
        y = [_x[1] for _x in circle]
        z = [_x[2] for _x in circle]
        
        return [x, y, z]
    
    def plotLatitudeCircle(self, _r, _theta_0, _num, _color, _axis):
        s_coords = self.latitudeCircle(_r, _theta_0, _num)
        _axis.plot(s_coords[0], s_coords[1], s_coords[2], color = _color, linewidth = 3.5)   
        

    def plotSphere(self, r, _theta_0, _theta_1, _phi_0, _phi_1, _num, _color, _axis ):
    
    # plot section of sphere or complete sphere
    
        s_coords = self.sphere(r, _theta_0, _theta_1, _phi_0, _phi_1, _num)
        _axis.plot(s_coords[0], s_coords[1], s_coords[2], color = _color)   
    
