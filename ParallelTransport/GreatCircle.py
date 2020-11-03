import numpy as np
import coordUtils as cu

class greatCircle:
    
    def __init__(self):
        self._cu = cu.coordUtils()
        
    def _beta(self, c):
    
        with np.errstate(divide='ignore'):
            ret = c/np.sqrt(1 - c**2)
        return ret
    
    
    def _arccot(self, x):
        
        ret = np.arctan2(1, x)
        return ret
    
        
    # Theta version of great circle equations as a function of phi
    
    def greatCircleTheta(self, _phi, _c, _k):
        
        # theta(phi)
        
        b_tmp = self._beta(_c)
        
        ret = -(1/b_tmp)*np.sin(_phi - _k)
        
        return self._arccot(ret)
   
    # cartesian vector great circle equation 
    
    def greatCircle(self, _u, _v, r, t):
        return r * np.cos(t)*_u + r * np.sin(t)*_v
    
     
    def computeParameters(self, _v_sp_0, _v_sp_1 ):
        
        # _v_sp_0 and _v_sp_1 are two points on the great
        # circle in spherical coordinates
        
        # find great circle parameters
        
        eps = 1.0e-4
        
        if _v_sp_0[0] != _v_sp_1[0]:
            raise Exception("radius not the same")

        ct = cu.coordUtils()
        
        u0 = ct.SphericalToCartesian(_v_sp_0)
        u1 = ct.SphericalToCartesian(_v_sp_1)
        
        _n = ct.planeNormal(u0, u1)
        
        for i in range(len(_n)):
            if np.abs(_n[i]) < 1.0e-10:
                _n[i] = 0.0
        
        _c = _n[2]
        
        try:
            k = np.arctan(-_n[0]/_n[1])
        except ZeroDivisionError:
            k = self._arccot(-_n[1]/_n[0])
        
        # see if parameters are correct
        
        theta_0 = self.greatCircleTheta(_v_sp_0[2], _c, k )
        theta_1 = self.greatCircleTheta(_v_sp_1[2], _c, k )
        
        if np.abs(theta_0 - _v_sp_0[1]) > eps  or np.abs(theta_1 - _v_sp_1[1]) > eps:
            _n = -_n
            _c = _n[2]
        
            try:
                k = np.arctan(-_n[0]/_n[1])
            except ZeroDivisionError:
                k = self._arccot(-_n[1]/_n[0])
            
        return [_c, k]
    
    #define arc between v0_sp and v1_sp
    
    def arcTheta(self, _v0_sp, _v1_sp, _num):
        
        #theta(phi)
        
        if np.abs(_v0_sp[2] - _v1_sp[2]) < 1e-7:
            # circle that goes through the poles
            ret = []
            theta = np.linspace(_v0_sp[1], _v1_sp[1], _num)
            for ang in theta:
                tmp = np.array([1, ang, _v0_sp[2]])
                ret.append(tmp)
        else:
            phi = np.linspace(_v0_sp[2], _v1_sp[2], _num)
            params = self.computeParameters(_v0_sp, _v1_sp)
            _c = params[0]
            _k = params[1]

            ret = []
        
            for ang in phi:
                tmp = np.array([1, self.greatCircleTheta(ang, _c, _k), ang])
                ret.append(tmp)
            
        return ret
    
    # draw an arc 2x the length going through the midpoint
  
    def twice_arcTheta(self, _v0_sp, _v1_sp, _num):
        
        # input vectors in spherical coordinates
        # theta(phi)
        
        if np.abs(_v0_sp[2] - _v1_sp[2]) < 1e-7:
            # circle that goes through the poles
            ret = []
            theta = np.linspace(_v0_sp[1], 2*_v1_sp[1], _num)
            for ang in theta:
                tmp = np.array([1, ang, _v0_sp[2]])
                ret.append(tmp)
        else:
            params = self.computeParameters(_v0_sp, _v1_sp)
            _c = params[0]
            _k = params[1]

        

        angle = 2 * ( _v1_sp[2] - _v0_sp[2] )
        ang = np.linspace(_v0_sp[2], angle, _num)


        ret = []
        
        for t in ang:
            tmp = np.array([1, self.greatCircleTheta(t, _c, _k), t])
            ret.append(tmp)
            
        return ret
    
    
    def arc(self, _u, _v, r, num):

        u_v = self._cu.createOrthonormal(_u, _v)
        angle = self._cu.angleBetween(_u, _v)
        
        ang = np.linspace(0, angle, num)
        
        ret = []
        
        for t in ang:
            ret.append(self.greatCircle(u_v[0], u_v[1], r, t))
        
        return ret

    # draw an arc 2x the length going through the midpoint
    
    def twice_arc(self, _u, _v, r, num):
                
        u_v = self._cu.createOrthonormal(_u, _v)
        angle = self._cu.angleBetween(_u, _v)
            
        ang = np.linspace(0, angle, num)
        ang = 2*ang
            
        ret = []
            
        for t in ang:
            ret.append(self.greatCircle(u_v[0], u_v[1], r, t))
                
        return ret
        
    
 


