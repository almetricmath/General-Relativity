import sympy as sp
import mpmath as mp

r = sp.Symbol('r',positive=True, real=True)
theta = sp.Symbol('theta', real=True)

x = sp.Symbol('x')
y = sp.Symbol('y')


def __x(_r, _theta):
    return _r*sp.cos(_theta)
 
def __y(_r, _theta):
    return _r*sp.sin(_theta)


def __r(_x, _y):
    return sp.sqrt(x**2+y**2)

def __theta(_x, _y):
    return sp.atan(_y/_x)

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

p_vec = [__x(r,theta),__y(r,theta)]
p_coords = [r, theta]
A = dvec(p_vec, p_coords).T

pinv_vec = [__r(x, y), __theta(x, y)]
pinv_coords = [x, y]
B = sp.simplify(dvec(pinv_vec, pinv_coords))
# need to put B back into spherical coordinates
# create an array of substitutions



b11 = __x(r,theta)/r
b12 = __y(r,theta)/r
b21 = -__y(r,theta)/r**2
b22 = __x(r,theta)/r**2

B_p = sp.Matrix([[b11,b12],[b21,b22]])

# Test A and B matrix

test = sp.trigsimp(B_p.T*A)

# compute the metric
G = sp.simplify(A*A.T)

# compuite the inverse matric

Ginv = sp.simplify(B_p*B_p.T)


# Compute Christoffel Symbols

# Ti,r

dAdr = sp.diff(A, r)
T_r = sp.simplify(dAdr*B_p.T)
dAdr_r = sp.simplify(T_r*A)

# check results

print('check T_r')

de1dr = T_r[0,0]*A[0,:]+T_r[0,1]*A[1,:]
error1 = de1dr - dAdr[0,:]
print('de1dr error = ',error1)

de2dr = T_r[1,0]*A[0,:]+T_r[1,1]*A[1,:]
error2 = de2dr - dAdr[1,:]
print('de2dr error = ', error2)

# Ti, _theta, k

dAdtheta = sp.diff(A, theta)
T_theta = sp.simplify(dAdtheta*B_p.T)
dAdtheta_r = sp.simplify(T_theta*A)

# check result

print('check T_theta')

de1dtheta = T_theta[0,0]*A[0,:] + T_theta[0,1]*A[1,:]
error1 = de1dtheta - dAdtheta[0,:]
print('error1 = ', error1)

de2dtheta = T_theta[1,0]*A[0,:] + T_theta[1,1]*A[1,:]
error2 = de2dtheta - dAdtheta[1,:]
print('error2 = ', error2)


