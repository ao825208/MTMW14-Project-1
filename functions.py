"""
MTMW14 - Project 1 - The ocean recharge oscillator model (ROM)
Student No. 30825208
Functions that define specific parameters used in schemes.
"""

import numpy as np       

# Thermocline depth equation (h)

def thermocline_depth(T, h, r, a, b, wind_stress):
   """Function that defines the equation for the west Pacific ocean
      thermocline depth (h)."""
   return -r*h - a*b*T - a*wind_stress

# SST anomaly equation (T)

def SST_anomaly(T, h, R, gamma, e, b, wind_stress, heating):
   """Function that defines the equation for the east Pacific Sea Surface
   Temperature anomaly (T)."""
   return R*T + gamma*h - e*(h + b*T)**3 + gamma*wind_stress + heating

# Coupling parameter on an annual cycle (mu)

def mu_D(mu0, mu_ann, t, tau_nondim):
    """Function that defines the coupling parameter that varies on an annual
    cycle."""
    return mu0*(1 + mu_ann*np.cos((2*np.pi*t/tau_nondim) - (5*np.pi/6)))

# Wind stress forcing

def wind_stress_E(f_ann, t, tau_nondim, f_ran, W, tau_cor, nd_dt):
    """Function that defines the wind stress forcing for the recharge
    oscillator model."""
    return (f_ann*np.cos(2*np.pi*t/tau_nondim)) + f_ran*W*tau_cor/nd_dt