import numpy as np
from scipy.optimize import curve_fit
from params import H,t_e

def get_decay_rate(h_max,t):
    # calculate decay rates from the elevation timeseries
    n_samp = 1
    rates = np.zeros(n_samp)
    
    # set cutoff (t>t_cut) to estimate the decay rate 
    t_cut = 0.0*t.max()

    h_decay = h_max[t>t_cut]
    t_d = t[t>t_cut]

    t_0 = t_d[0]
    t_d -= t_0

    func = lambda x, a, b, c: a * np.exp(-b * x) + c

    y_data = h_decay
    x_data = t_d/t_e

    popt, pcov = curve_fit(func, x_data, y_data,method='dogbox')

    a,b,c = popt

    t_decay = 1/b

    h_pred = lambda x: func(x,*popt)
    return t_decay,h_pred