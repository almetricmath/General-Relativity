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
print('de1dr error = \n')
sp.pprint(error1)
print('\n')

de2dr = T_r[1,0]*A[0,:]+T_r[1,1]*A[1,:]
error2 = de2dr - dAdr[1,:]
print('de2dr error = \n')
sp.pprint(error2)
print('\n')

# Ti, _theta, k

dAdtheta = sp.diff(A, theta)
T_theta = sp.simplify(dAdtheta*B_p.T)
dAdtheta_r = sp.simplify(T_theta*A)

# check result

print('check T_theta')

de1dtheta = T_theta[0,0]*A[0,:] + T_theta[0,1]*A[1,:]
error1 = de1dtheta - dAdtheta[0,:]
print('error1 = \n')
sp.pprint(error1)
print('\n')

de2dtheta = T_theta[1,0]*A[0,:] + T_theta[1,1]*A[1,:]
error2 = de2dtheta - dAdtheta[1,:]
print('error2 = \n')
sp.pprint(error2)
print('\n')


# Convert Christoffel Symbols of the Second Kind, T_r
# to Christoffel Symbols of the First Kind T1_r

T1_r = T_r





# Covariant Derivative 

# vector

vr = sp.Symbol('v_r', real=True)
v_theta = sp.Symbol('v_theta', real=True)

v = sp.Matrix([r*(sp.cos(theta)**2),-r*sp.sin(theta)**2])
#v = sp.Matrix([vr, v_theta])

# r component

dvdr = sp.diff(v, r)
testr = sp.simplify(T_r.T*v)
cov_r = sp.simplify(dvdr + testr)
#print('covariant derivative wrt r\n')
#sp.pprint(cov_r)

# theta component

dvdtheta = sp.diff(v,theta)
testtheta = sp.simplify(T_theta.T*v)
cov_theta = sp.simplify(dvdtheta + testtheta)
#print('covariant derivative wrt theta\n')
#sp.pprint(cov_theta)

# Christoffel Symbol of the First Kind - equation (49) - Curvilinear Calculus


dG1a = dvec(G[:,0].T, p_coords).T
dG2a = dvec(G[0,:], p_coords)
dG3a = sp.diff(G, r)

T1 = sp.simplify((1/2)*(dG1a+dG2a-dG3a))
print('Christoffel Symbol of the First Kind, Tr\n')
sp.pprint(T1)

dG1b = dvec(G[:,1].T, p_coords).T
dG2b = dvec(G[1,:], p_coords)
dG3b = sp.diff(G, theta)

T2 = sp.simplify((1/2)*(dG1b+dG2b-dG3b))
print('Christoffel Symbol of the First Kind, Ttheta\n')
sp.pprint(T2)



'''
print('dG1 = \n')
sp.pprint(dG1)

print('dG2 = \n')
sp.pprint(dG2)

print('dG3 = \n')
sp.pprint(dG3)

dG1 = dvec(G[:,1], p_coords).T
dG2 = dvec(G[1,:], p_coords)
dG3 = sp.diff(G, r)

T2 = dG1+dG1-dG3

print('Christoffel Symbol of the First Kind - p = 2\n')
sp.pprint(T2)
'''