import sympy as sp
import mpmath as mp

r = sp.Symbol('r',positive=True, real=True)
theta = sp.Symbol('theta', real=True)
phi = sp.Symbol('phi', real=True)

x = sp.Symbol('x')
y = sp.Symbol('y')
z = sp.Symbol('z')



def __x(_r, _theta, _phi):
    return _r*sp.sin(_theta)*sp.cos(_phi)
 
def __y(_r, _theta, _phi):
    return _r*sp.sin(_theta)*sp.sin(_phi)

def __z(_r, _theta, _phi):
    return _r*sp.cos(_theta)

def __r(_x, _y, _z):
    return sp.sqrt(x**2+y**2+z**2)

def __theta(_x, _y, _z):
    return sp.atan(sp.sqrt((_x**2+_y**2))/z)

def __phi(_x, _y, _z):
    return sp.atan(_y/_x)


#test = pi.evalf(25)

# compute vector derivative

def dvec(_v, _x):
    
    ret = []
    
    _vars = outer(_v, _x)
    for i in _vars:
        tmp = []
        for _j in i:
            tmp.append(sp.diff(_j[0], _j[1]))
        ret.append(tmp)
    
    return sp.Matrix(ret)
    

# compute outer product

def outer(v1, v2):
    ret = []
    
    for i in v1:
        tmp = []
        for j in v2:
            tmp.append((i,j))
        ret.append(tmp)
    
    return ret


sp.init_printing()
# compute A matrix

sp_vec = [__x(r,theta,phi),__y(r,theta,phi),__z(r,theta,phi)]
sp_coords = [r, theta, phi]
A = dvec(sp_vec, sp_coords).T
car_vec = [__r(x, y, z), __theta(x, y, z), __phi(x, y, z)]
car_coords = [x, y, z]
B = sp.simplify(dvec(car_vec, car_coords))
# need to put B back into spherical coordinates
# create an array of substitutions
b11 = __x(r,theta,phi)/r
b12 = __y(r,theta,phi)/r
b13 =  __z(r,theta,phi)/r
b21 = sp.cos(phi)*sp.cos(theta)/r
b22 = sp.sin(phi)*sp.cos(theta)/r
b23 = -sp.sin(theta)/r
b31 = -sp.sin(phi)/(r*sp.sin(theta))
b32 = sp.cos(phi)/(r*sp.sin(theta))
b33 = 0

B_sp = sp.Matrix([[b11,b12,b13],[b21,b22,b23],[b31,b32,b33]])

# Test A and B matrix

test = sp.trigsimp(B_sp.T*A)


# compute the metric
G = sp.simplify(A*A.T)

# compuite the inverse matric

Ginv = sp.simplify(B_sp*B_sp.T)

# Compute Christoffel Symbols

# Ti_r,k

dAdr = sp.diff(A, r)
T_r = sp.simplify(dAdr*B_sp.T)

#check T_r

dAdr_r = sp.simplify(T_r*A)



# compute error

error_r = dAdr_r - dAdr

print('error in r derivative\n')
print('error_r\n', error_r)


# Ti_theta, k

dAdtheta = sp.diff(A, theta)
T_theta = sp.simplify(dAdtheta*B_sp.T)
dAdtheta_r = sp.simplify(T_theta*A)

# compute error

error_theta = dAdtheta_r - dAdtheta

print('error in theta derivative\n')
print('error_theta\n', error_theta)


# Ti_phi, k

dAdphi = sp.diff(A, phi)
T_phi = sp.simplify(dAdphi*B_sp.T)
dAdphi_r = sp.simplify(T_phi*A)

# compute error

error_phi = dAdphi_r - dAdphi

print('error in theta derivative\n')
print('error_phi\n', error_phi)

# Covariant Derivative





