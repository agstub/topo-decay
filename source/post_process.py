import numpy as np
from scipy.optimize import curve_fit
from params import H,t_e

def get_decay_rate(h_max,t):
    # calculate decay rates using a range of samples
    # from the elevation timeseries
    n_samp = 1
    rates = np.zeros(n_samp)
    
    # cutoff (t>t_cut) used to estimate the decay rate to avoid
    # initial adjustment from non-hydrostatic inititical condition
    t_cut = [0] #np.linspace(0.0*t.max(),0.5*t.max(),n_samp)

    for j in range(rates.size):
        t_off = t_cut[j]

        h_decay = h_max[t>t_off]
        t_d = t[t>t_off]

        t_0 = t_d[0]
        t_d -= t_0

        func = lambda x, a, b, c: a * np.exp(-b * x) + c

        y_data = h_decay
        x_data = t_d/t_e

        popt, pcov = curve_fit(func, x_data, y_data,method='dogbox')

        a,b,c = popt

        rates[j] = 1/b

    t_decay = np.mean(rates)
    h_pred = lambda x: func(x,*popt)
    return t_decay,h_pred