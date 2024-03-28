from params import delta,eps,ratio
import numpy as np

def R(k):
    # relaxation function for floating ice
    R1 =  np.exp(4*k) + (4*k)*np.exp(2*k) - 1 
    D = k*(np.exp(4*k) -2*(1+2*k**2)*np.exp(2*k) + 1)
    f0 = D/R1
    f = 1/(eps+f0)
    return f

def B(k):
    # buoyancy transfer function for floating ice
    B1 =  2*(k+1)*np.exp(3*k) + 2*(k-1)*np.exp(k)
    D = k*(np.exp(4*k) -2*(1+2*k**2)*np.exp(2*k) + 1)
    f0 = D/B1
    f =1/(eps+f0)
    return f

def t_p(k):
    # expression for the dominant relaxation timescale
    chi = (1-delta)*R(k)
    mu = np.sqrt(4*delta*(B(k))**2 + chi**2)
    Lp = -0.5*(delta+1)*R(k)+0.5*mu
    return -ratio/Lp


def t_m(k):
    # expression for the faster timescale associated with
    # adjustment towards hydrostatic state
    chi = (1-delta)*R(k)
    mu = np.sqrt(4*delta*(B(k))**2 + chi**2)
    Lm = -0.5*(delta+1)*R(k)-0.5*mu 
    return -ratio/Lm