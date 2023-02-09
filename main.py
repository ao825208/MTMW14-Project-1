"""
MTMW14 - Project 1 - The ocean recharge oscillator model (ROM)
Student No. 30825208
Main code to run all tasks in the assignment.
"""

import numpy as np
import matplotlib.pyplot as plt
from functions import *
from schemes import *

# Recharge Oscillator Model (Tasks A, B, C)

def ROM(b0 = 2.5, gamma = 0.75, c = 1, r = 0.25, a = 0.125, e = 0, mu = 2/3, \
        wind_stress = 0, heating = 0, t_min = 0, t_max = 41, nt = 1000):
    """Recharge oscillator model with initial parameters defined from theory 
    from Jin (1997a).
    
    Call signature:
        ROM(b0 = 2.5, gamma = 0.75, c = 1, r = 0.25, a = 0.125, e = 0,
            mu = 2/3, wind_stress = 0, heating = 0, t_min = 0, t_max = 41,
            nt = 1000)

    Parameters:
        b0 : float
            measure of thermocline slope
        gamma : float
            feedback of the thermocline gradient on the SST gradient
        c : int
            damping rate of SST anomalies
        r : float
            damping of the upper ocean heat content
        a : float
            enchanced easterly wind stress relation to the recharge of ocean
            heat content
        e : float
            nonlinearity of ROM
        mu : float
            coupling coefficient
        wind_stress : float
            random wind stress forcing
        heating : float
            random heating
        t_min : int
            minimum value of time
        t_max : int
            maximum value of time
        nt : int
            number of time steps"""
    
    # Arrays
    h = np.zeros(nt + 1)        # array to store values of thermocline depth
    T = np.zeros(nt + 1)        # array to store values of SST anomaly
    
    # Non-dimensionalised values
    T_nondim = 7.5
    h_nondim = 150
    t_nondim = 2

    # Time steps
    dt = ((t_max - t_min)/nt)   # time step
    nd_dt = dt/t_nondim         # non-dimensionalised dt

    # Initial conditions
    h[0] = 0/h_nondim
    T[0] = 1.125/T_nondim
    
    # Time axis setup
    t_axis = np.linspace(t_min, t_max, nt + 1)
    
    # Additional parameters
    b = b0*mu                   # measure of thermocline slope
    R = gamma*b - c             # Bjerknes positive feedback process
    
    # Defining runga-kutta for redimensionalisation
    rk = runge_kutta(T, h, nd_dt, nt, r, a, b, wind_stress, heating,\
                     gamma, R, e)

    # Redimensionalise thermocline depth (h) and SST anomaly (T)
    h_rk = rk[0]*h_nondim
    T_rk = rk[1]*T_nondim
    
    # Creating a figure
    fig = plt.figure(figsize = (20, 7))
    
    # Setting figure rows and columns
    rows = 1
    columns = 2
    
    # Graph of SST anomaly and thermocline depth against time
    fig.add_subplot(rows, columns, 1)
    plt.plot(t_axis, h_rk, color = 'blue', label = 'Thermocline depth, hw (m)') 
    plt.plot(t_axis, T_rk, color = 'red', label = 'SST anomaly, Te (K)')
    plt.ylabel('Te (K), hw (m)')
    plt.xlabel('time (months)')
    plt.title('SST anomaly (Te) and Thermocline depth (hw) vs. time (t)')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Graph of SST anomaly against thermocline depth
    fig.add_subplot(rows, columns, 2)
    plt.plot(T_rk, h_rk)
    plt.xlabel('Te (K)')
    plt.ylabel('hw (m)')
    plt.title('SST anomaly (Te) vs. Thermocline depth (hw)')
    plt.tight_layout()

# Task A - Linear models (one and five periods)
ROM()
ROM(t_max = 41*5)               # stability test

# Task B - Sub-critical and super=critical models
ROM(mu = 0.6, t_max = 41*5)     # sub-critical (mu < 2/3)
ROM(mu = 0.7, t_max = 41*5)     # super-critical (mu > 2/3)

# Task C - Nonlinearity set-up
ROM(e = 0.1, t_max = 41*5)
ROM(e = 0.1, mu = 0.7, t_max = 41*5)
ROM(e = 0.1, mu = 0.75, t_max = 41*5)

# Task D - Self-excitation hypotheses

# Recharge oscillator model (Task D)

def ROM_D(b0 = 2.5, gamma = 0.75, c = 1, r = 0.25, a = 0.125, e = 0.1,\
          wind_stress = 0, heating = 0, t_min = 0, t_max = 41, nt = 1000,\
              mu0 = 0.75, mu_ann = 0.2, tau_nondim = 12/2):
    """Recharge oscillator model with initial parameters defined from theory 
    from Jin (1997a), now considering the variation of the coupling parameter,
    mu, on an annual cycle.
    
    Call signature:
        ROM_D(b0 = 2.5, gamma = 0.75, c = 1, r = 0.25, a = 0.125, e = 0,
            wind_stress = 0, heating = 0, t_min = 0, t_max = 41, nt = 1000,
            mu0 = 0.75, mu_ann = 0.2, tau_nondim = 12/6)

    New parameters:
        mu0 : float
            initial value for the coupling parameter
        mu_ann: float
            annual value for coupling parameter
        tau_nondim: int
            annual cycle (non-dimensionalised)"""
    
    # Arrays
    h = np.zeros(nt + 1)        # array to store values of thermocline depth
    T = np.zeros(nt + 1)        # array to store values of SST anomaly
    t = np.zeros(nt + 1)        # array to store values of time
    
    # Non-dimensionalised values
    T_nondim = 7.5
    h_nondim = 150
    t_nondim = 2

    # Time steps
    dt = ((t_max - t_min)/nt)   # time step
    nd_dt = dt/t_nondim         # non-dimensionalised dt

    # Initial conditions
    h[0] = 0/h_nondim
    T[0] = 1.125/T_nondim
    t[0] = 0/t_nondim
    t[1] = t[0] + nd_dt
    
    # Time axis setup
    t_axis = np.linspace(t_min, t_max, nt + 1)
    
    # Defining runga-kutta for redimensionalisation
    rk_D = runge_kutta_D(T, h, t, nd_dt, nt, r, a, c, wind_stress, heating,\
                         gamma, b0, e, mu0, mu_ann, tau_nondim)

    # Redimensionalise thermocline depth (h) and SST anomaly (T) and time (t)
    h_rk_D = rk_D[0]*h_nondim
    T_rk_D = rk_D[1]*T_nondim
    
    # Creating a figure
    fig = plt.figure(figsize = (20, 7))
    
    # Setting figure rows and columns
    rows = 1
    columns = 2
    
    # Graph of SST anomaly and thermocline depth against time
    fig.add_subplot(rows, columns, 1)
    plt.plot(t_axis, h_rk_D, color = 'blue',\
             label = 'Thermocline depth, hw (m)') 
    plt.plot(t_axis, T_rk_D, color = 'red', label = 'SST anomaly, Te (K)')
    plt.ylabel('Te (K), hw (m)')
    plt.xlabel('time (months)')
    plt.title('SST anomaly (Te) and Thermocline depth (hw) vs. time (t)')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Graph of SST anomaly against thermocline depth
    fig.add_subplot(rows, columns, 2)
    plt.plot(T_rk_D, h_rk_D)
    plt.xlabel('Te (K)')
    plt.ylabel('hw (m)')
    plt.title('SST anomaly (Te) vs. Thermocline depth (hw)')
    plt.tight_layout()
    
ROM_D(t_max = 41*5)

# Task E - Stochastic initiation hypotheses

# Recharge oscillator model (Task E, F)

def ROM_E(b0 = 2.5, gamma = 0.75, c = 1, r = 0.25, a = 0.125, e = 0,\
          wind_stress = 0, heating = 0, t_min = 0, t_max = 41, nt = 10000,\
              mu0 = 0.75, mu_ann = 0.2, tau_nondim = 12/2, f_ann = 0.02,\
                  f_ran = 0.2, tau_cor = (1/30)/2):
    """Recharge oscillator model with initial parameters defined from theory 
    from Jin (1997a), now considering the variation of the wind stress forcing
    over multiple time steps.
    
    Call signature:
        ROM_E(b0 = 2.5, gamma = 0.75, c = 1, r = 0.25, a = 0.125, e = 0,
            wind_stress = 0, heating = 0, t_min = 0, t_max = 41, nt = 10000,
            mu0 = 0.75, mu_ann = 0.2, tau_nondim = 12/6, f_ann = 0.02,
            f_ran = 0.2, tau_cor = 1/30)

    New parameters:
        f_ann : float
            annual forcing
        f_ran: float
            random forcing
        tau_cor: float
            daily cycle"""
    
    # Arrays
    h = np.zeros(nt + 1)        # array to store values of thermocline depth
    T = np.zeros(nt + 1)        # array to store values of SST anomaly
    t = np.zeros(nt + 1)        # array to store values of time
    
    # Non-dimensionalised values
    T_nondim = 7.5
    h_nondim = 150
    t_nondim = 2

    # Time steps
    dt = 1/30                   # time step
    nd_dt = dt/t_nondim         # non-dimensionalised dt

    # Initial conditions
    h[0] = 0/h_nondim
    T[0] = 1.125/T_nondim
    t[0] = 0/t_nondim
    t[1] = t[0] + nd_dt
    
    # Time axis setup
    t_axis = np.linspace(t_min, t_max, nt + 1)
    
    # Defining runga-kutta for redimensionalisation
    rk_E = runge_kutta_E(T, h, t, nd_dt, nt, r, a, c, wind_stress, heating,\
                         gamma, b0, e, mu0, mu_ann, tau_nondim, f_ann, f_ran,\
                             tau_cor)

    # Redimensionalise thermocline depth (h) and SST anomaly (T) and time (t)
    h_rk_E = rk_E[0]*h_nondim
    T_rk_E = rk_E[1]*T_nondim
    
    # Creating a figure
    fig = plt.figure(figsize = (20, 7))
    
    # Setting figure rows and columns
    rows = 1
    columns = 2
    
    # Graph of SST anomaly and thermocline depth against time
    fig.add_subplot(rows, columns, 1)
    plt.plot(t_axis, h_rk_E, color = 'blue',\
             label = 'Thermocline depth, hw (m)') 
    plt.plot(t_axis, T_rk_E, color = 'red', label = 'SST anomaly, Te (K)')
    plt.ylabel('Te (K), hw (m)')
    plt.xlabel('time (months)')
    plt.title('SST anomaly (Te) and Thermocline depth (hw) vs. time (t)')
    plt.legend(bbox_to_anchor=(1, 1))
    plt.tight_layout()

    # Graph of SST anomaly against thermocline depth
    fig.add_subplot(rows, columns, 2)
    plt.plot(T_rk_E, h_rk_E)
    plt.xlabel('Te (K)')
    plt.ylabel('hw (m)')
    plt.title('SST anomaly (Te) vs. Thermocline depth (hw)')
    plt.tight_layout()
    
ROM_E(t_max = 41*5)
ROM_E(t_max = 41*5, f_ran = 0) # Considering wind stress depending on f_ann
ROM_E(t_max = 41*5, f_ann = 0) # Considering wind stress depending on f_ran

# Task F - Introducing nonlinearity
ROM_E(e = 0.1, t_max = 41*5)

# Task G - Chaotic behaviour

# Recharge oscillator model (Task G)

def ROM_G(b0 = 2.5, gamma = 0.75, c = 1, r = 0.25, a = 0.125, e = 0,\
          wind_stress = 0, heating = 0, t_min = 0, t_max = 41, nt = 10000,\
              mu0 = 0.75, mu_ann = 0.2, tau_nondim = 12/2, f_ann = 0.02,\
                  f_ran = 0.2, tau_cor = (1/30)/2):
    """Recharge oscillator model with initial parameters defined from theory 
    from Jin (1997a), now considering an ensemble of forecasts as an attempt 
    to simulate chaos.
    
    Call signature:
        ROM_G(b0 = 2.5, gamma = 0.75, c = 1, r = 0.25, a = 0.125, e = 0,
            wind_stress = 0, heating = 0, t_min = 0, t_max = 41, nt = 10000,
            mu0 = 0.75, mu_ann = 0.2, tau_nondim = 12/6, f_ann = 0.02,
            f_ran = 0.2, tau_cor = 1/30)"""
    
    # Arrays
    h = np.zeros(nt + 1)        # array to store values of thermocline depth
    T = np.zeros(nt + 1)        # array to store values of SST anomaly
    t = np.zeros(nt + 1)        # array to store values of time
    
    # Non-dimensionalised values
    T_nondim = 7.5
    h_nondim = 150
    t_nondim = 2

    # Time steps
    dt = 1/30                   # time step
    nd_dt = dt/t_nondim         # non-dimensionalised dt

    # Initial conditions
    h[0] = 0/h_nondim
    T[0] = 1.125/T_nondim
    t[0] = 0/t_nondim
    t[1] = t[0] + nd_dt
    
    # Time axis setup
    t_axis = np.linspace(t_min, t_max, nt + 1)
    
    # Implementing an ensemble model
    size = 10                   # ensemble size

    # Lists 
    T_ens = np.zeros([nt + 1, size])        # array to store ensemble of T
    h_ens = np.zeros([nt + 1, size])        # array to store ensemble of h
    
    # Loop to go through runga-kutta scheme for all perturbations of variables
    for ie in range(size):
        T_pert = np.random.uniform(-1, 1)       # perturbations of T
        h_pert = np.random.uniform(-2.5, 2.5)   # perturbations of h
        
        T_init = (1.125 + T_pert)/7.5
        T[0] = T_init
        
        h_init = (0 + h_pert)/150 
        h[0] = h_init
        
        rk_G = runge_kutta_E(T, h, t, nd_dt, nt, r, a, c, wind_stress,\
                             heating, gamma, b0, e, mu0, mu_ann, tau_nondim,\
                                 f_ann, f_ran, tau_cor)
        
        T_ens[:,ie] = rk_G[0]
        h_ens[:,ie] = rk_G[1]
        
    # Re-dimensionalise h and T
    New_T = T_ens*T_nondim
    New_h = h_ens*h_nondim
    
    # Creating a figure
    fig = plt.figure(figsize = (20, 7))
    
    # Setting figure rows and columns
    rows = 1
    columns = 2
    
    # Graph of SST anomaly (T) against time
    fig.add_subplot(rows, columns, 1)
    plt.plot(t_axis, New_T)
    plt.ylabel('Temperature (K)')
    plt.xlabel('time (months)')
    plt.title('SST anomaly (Te) vs. time (t)')
    plt.tight_layout()
    
    # Graph of SST anomaly (T) against thermocline depth (h)
    fig.add_subplot(rows, columns, 2)
    plt.plot(New_T, New_h)
    plt.xlabel('Te (K)')
    plt.ylabel('hw (m)')
    plt.title('SST anomaly (Te) vs. Thermocline depth (hw)')
    plt.tight_layout()

ROM_G(e = 0.1, t_max = 41*5)

# Increasing stochastic forcing to simulate chaos
ROM_G(e = 0.1, t_max = 41*5, f_ann = 1)
ROM_G(e = 0.1, t_max = 41*5, f_ann = 1, f_ran = 0.8)
ROM_G(e = 0.1, t_max = 41*5, f_ann = 2, f_ran = 0.8)