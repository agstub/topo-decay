# Surface mass balance functions (for upper and lower surfaces)
# Note: I've written the expressions below so that they work on both UFL
#       objects and numpy arrays because I also compute a linear solution
#       by convolving m with a Green's function!
import numpy as np
from params import L, H, t_e
from scipy.special import erf

# m0 = 5 / 3.154e7               # max basal melt(+) or freeze(-) rate (m/yr)
# stdev = 10*H/3                  # standard deviation for Gaussian basal melt anomaly

def step(t):
    return (t+abs(t))/(2*t)

def smb_s(x,t,m0,stdev):
    m = m0*(np.exp(1)**(-x**2/(stdev**2)))
    return m*step(8*t_e-t)

def smb_h(x,t,m0,stdev):
    # Surface mass balance functions (at upper surface)
    a = m0*np.sqrt(np.pi)*stdev*erf(L/(2*stdev)) / L 
    return a*step(8*t_e-t)