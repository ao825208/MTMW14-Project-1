"""
MTMW14 - Project 1 - The ocean recharge oscillator model (ROM)
Student No. 30825208
Time-stepping schemes used to determine solutions to the coupled equations.
"""

import numpy as np
from functions import *

# Runga-Kutta (Task A, B, C)

def runge_kutta(T, h, nd_dt, nt, r, a, b, wind_stress, heating, gamma, R, e):
    """Function that solves the runge-kutta time step scheme for the coupled 
    ocean equations for thermocline depth and SST anomaly ordinary differential
    equation."""
    
    for i in range(0, nt):
        k1 = thermocline_depth(T[i], h[i], r, a, b, wind_stress)
        l1 = SST_anomaly(T[i], h[i], R, gamma, e, b, wind_stress, heating)
        
        k2 = thermocline_depth((T[i] + l1*(nd_dt/2)), (h[i] + k1*(nd_dt/2)),\
                               r, a, b, wind_stress)
        l2 = SST_anomaly((T[i] + l1*(nd_dt/2)), (h[i] + k1*(nd_dt/2)), R,\
                         gamma, e, b, wind_stress, heating)
        
        k3 = thermocline_depth((T[i] + l2*(nd_dt/2)), (h[i] + k2*(nd_dt/2)),\
                               r, a, b, wind_stress)
        l3 = SST_anomaly((T[i] + l2*(nd_dt/2)), (h[i] + k2*(nd_dt/2)), R,\
                         gamma, e, b, wind_stress, heating)
        
        k4 = thermocline_depth((T[i] + l3*nd_dt), (h[i] + k3*nd_dt), r, a, b,\
                               wind_stress)
        l4 = SST_anomaly((T[i] + l3*nd_dt), (h[i] + k3*nd_dt), R, gamma, e,\
                         b, wind_stress, heating)
        
        h[i+1] = h[i] + nd_dt*(1/6)*(k1 + 2*k2 + 2*k3 + k4)
        T[i+1] = T[i] + nd_dt*(1/6)*(l1 + 2*l2 + 2*l3 + l4)
    
    return h, T

# Runga-Kutta (Task D)

def runge_kutta_D(T, h, t, nd_dt, nt, r, a, c, wind_stress, heating, gamma,\
                  b0, e, mu0, mu_ann, tau_nondim):
    """Function that solves the runge-kutta time step scheme for the coupled 
    ocean equations for thermocline depth and SST anomaly ordinary differential
    equation. This model now considers the variation in the coupling parameter
    on an annual cycle."""
    
    for i in range(0, nt):
        mu = mu_D(mu0, mu_ann, t[i], tau_nondim)
        
        b = b0*mu                   # measure of thermocline slope
        R = gamma*b - c             # Bjerknes positive feedback process
        
        k1 = thermocline_depth(T[i], h[i], r, a, b, wind_stress)
        l1 = SST_anomaly(T[i], h[i], R, gamma, e, b, wind_stress, heating)
        
        k2 = thermocline_depth((T[i] + l1*(nd_dt/2)), (h[i] + k1*(nd_dt/2)),\
                               r, a, b, wind_stress)
        l2 = SST_anomaly((T[i] + l1*(nd_dt/2)), (h[i] + k1*(nd_dt/2)), R,\
                         gamma, e, b, wind_stress, heating)
        
        k3 = thermocline_depth((T[i] + l2*(nd_dt/2)), (h[i] + k2*(nd_dt/2)),\
                               r, a, b, wind_stress)
        l3 = SST_anomaly((T[i] + l2*(nd_dt/2)), (h[i] + k2*(nd_dt/2)), R,\
                         gamma, e, b, wind_stress, heating)
        
        k4 = thermocline_depth((T[i] + l3*nd_dt), (h[i] + k3*nd_dt), r, a, b,\
                               wind_stress)
        l4 = SST_anomaly((T[i] + l3*nd_dt), (h[i] + k3*nd_dt), R, gamma, e,\
                         b, wind_stress, heating)
        
        h[i+1] = h[i] + nd_dt*(1/6)*(k1 + 2*k2 + 2*k3 + k4)
        T[i+1] = T[i] + nd_dt*(1/6)*(l1 + 2*l2 + 2*l3 + l4)
        t[i+1] = t[i] + nd_dt
        
    return h, T, t

# Runge-Kutta (Task E, F, G)

def runge_kutta_E(T, h, t, nd_dt, nt, r, a, c, wind_stress, heating, gamma,\
                  b0, e, mu0, mu_ann, tau_nondim, f_ann, f_ran, tau_cor):
    """Function that solves the runge-kutta time step scheme for the coupled 
    ocean equations for thermocline depth and SST anomaly ordinary differential
    equation. This model now considers wind stress forcing."""
    
    for i in range(0, nt):
        mu = mu_D(mu0, mu_ann, t[i], tau_nondim)
        
        b = b0*mu                   # measure of thermocline slope
        R = gamma*b - c             # Bjerknes positive feedback process
        
        W = np.random.uniform(-1, 1, nt + 1)    # White noise
        
        wind_stress = wind_stress_E(f_ann, t[i], tau_nondim, f_ran, W[i],\
                                    tau_cor, nd_dt)
        
        k1 = thermocline_depth(T[i], h[i], r, a, b, wind_stress)
        l1 = SST_anomaly(T[i], h[i], R, gamma, e, b, wind_stress, heating)
        
        k2 = thermocline_depth((T[i] + l1*(nd_dt/2)), (h[i] + k1*(nd_dt/2)),\
                               r, a, b, wind_stress)
        l2 = SST_anomaly((T[i] + l1*(nd_dt/2)), (h[i] + k1*(nd_dt/2)), R,\
                         gamma, e, b, wind_stress, heating)
        
        k3 = thermocline_depth((T[i] + l2*(nd_dt/2)), (h[i] + k2*(nd_dt/2)),\
                               r, a, b, wind_stress)
        l3 = SST_anomaly((T[i] + l2*(nd_dt/2)), (h[i] + k2*(nd_dt/2)), R,\
                         gamma, e, b, wind_stress, heating)
        
        k4 = thermocline_depth((T[i] + l3*nd_dt), (h[i] + k3*nd_dt), r, a, b,\
                               wind_stress)
        l4 = SST_anomaly((T[i] + l3*nd_dt), (h[i] + k3*nd_dt), R, gamma, e,\
                         b, wind_stress, heating)
        
        h[i+1] = h[i] + nd_dt*(1/6)*(k1 + 2*k2 + 2*k3 + k4)
        T[i+1] = T[i] + nd_dt*(1/6)*(l1 + 2*l2 + 2*l3 + l4)
        t[i+1] = t[i] + nd_dt
        
    return h, T, t