# Surface mass balance functions (for upper and lower surfaces)
# Note: I've written the expressions below so that they work on both UFL
#       objects and numpy arrays 
import numpy as np
from params import L, H, t_e
from scipy.special import erf
from ufl import sin

def step(t):
    return (t+abs(t))/(2*t)

def smb_s(x,t,m0,lamda):
    k = 2*np.pi/lamda   # k0 from 1e-2 to 1e2
    m = m0*sin(k*x)
    return m*step(4*t_e-t)

def smb_h(x,t,m0,lamda):
    # Surface mass balance functions (at upper surface)
    return 0*x 