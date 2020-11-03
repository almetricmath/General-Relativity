import numpy as np


class coordUtils:

    def SphericalToCartesian(self, v):
         
        r = v[0]
        theta = v[1]
        phi = v[2]
        
        x = r * np.sin(theta) * np.cos(phi)
        y = r * np.sin(theta) * np.sin(phi)
        z = r * np.cos(theta)
        
        ret = np.array([x, y, z])
        return ret
    
    def SphericalToCartesianVec(self, v_sp):
        
        vec_xyz = []
        
        for v in v_sp:
            tmp = self.SphericalToCartesian(v)
            vec_xyz.append(tmp)
    
        return vec_xyz
    
    def CartesianToSpherical(self, v):
        
        x = v[0]
        y = v[1]
        z = v[2]
        
        r = np.sqrt(x**2 + y**2 + z**2)
        phi = np.arctan2(y, x)
        if phi < 0 and np.abs(phi) > 1e-7:
            phi = phi + 2*np.pi
        theta = np.arctan2(np.sqrt(x**2 + y**2), z)
        
        ret = np.array([r, theta, phi])
        return ret
        

    # determine normalized normal to plane
        
    def planeNormal(self, _v1, _v2):
    
        if self.angleBetween(_v1, _v2) != 0:
            n = np.cross(_v1, _v2)
            n_mag = np.linalg.norm(n)
            return n/n_mag
        else:
            print(' vectors are linearly dependent')
            return np.array([0, 0, 0])
    
    # creates an orthonormal coordinate system
    # from 2 linearly independent vectors
        
    def createOrthonormal(self, _v1, _v2):
        
        _v1_mag = np.linalg.norm(_v1)
        _u = _v1/_v1_mag  # first orthonormal basis vector
        
        _n = self.planeNormal(_v1, _v2) # second orthonormal basis vector
        _v = self.planeNormal(_n, _u)
        
        return [_u, _v, _n]
        
    def angleBetween(self, _v1, _v2):
        
        mag_v1 = np.linalg.norm(_v1)
        mag_v2 = np.linalg.norm(_v2)
        tmp = np.dot(_v1, _v2)/(mag_v1*mag_v2)
        ret = np.arccos(tmp)
        
        return ret
    
