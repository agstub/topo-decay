from params import delta,eps,ratio
import numpy as np

def R(k,Exp=lambda x: np.exp(1)**x):
    # relaxation function for floating ice
    R1 =  Exp(4*k) + (4*k)*Exp(2*k) - 1 
    D = k*(Exp(4*k) -2*(1+2*k**2)*Exp(2*k) + 1)
    f0 = D/R1
    f = 1/(eps+f0)
    return f

def B(k,Exp=lambda x: np.exp(1)**x):
    # buoyancy transfer function for floating ice
    B1 =  2*(k+1)*Exp(3*k) + 2*(k-1)*Exp(k)
    D = k*(Exp(4*k) -2*(1+2*k**2)*Exp(2*k) + 1)
    f0 = D/B1
    f =1/(eps+f0)
    return f

def t_p(k):
    # expression for the dominant relaxation timescale
    chi = (1-delta)*R(k)
    mu = (4*delta*(B(k))**2 + chi**2)**0.5
    Lp = -0.5*(delta+1)*R(k)+0.5*mu
    return -ratio/Lp


def t_m(k):
    # expression for the faster timescale associated with
    # adjustment towards hydrostatic state
    chi = (1-delta)*R(k)
    mu = (4*delta*(B(k))**2 + chi**2)**0.5
    Lm = -0.5*(delta+1)*R(k)-0.5*mu 
    return -ratio/Lm

def h_exact(t,k):
    # analytic solution for sinusoidal perturbations at surface 
    a = 0.5*(1-delta)*R(k)/B(k)
    mu = np.sqrt(4*delta*B(k)**2 + ((delta-1)*R(k))**2)
    b = mu/(2*B(k))
    return 0.5*((1-a/b)*np.exp(-t/t_p(k)) + (1+a/b)*np.exp(-t/t_m(k)))


def s_exact(t,k):
    # analytic solution for sinusoidal perturbations at base
    a = 0.5*(1-delta)*R(k)/B(k)
    mu = np.sqrt(4*delta*B(k)**2 + ((delta-1)*R(k))**2)
    b = mu/(2*B(k))
    return 0.5*((1-a/b)*np.exp(-t/t_m(k)) + (1+a/b)*np.exp(-t/t_p(k)))