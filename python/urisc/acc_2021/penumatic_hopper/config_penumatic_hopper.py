""" ideally has all config parameters for the problem """

import numpy as np 
import os,sys 

LINE_WIDTH = 100 
PLOT_FIGS = True 
SAVE_SOLN = True

x0 = np.array([0.5, 0., 0., 0.])
MAX_ITER = 1000
horizon = 200 


plan_dt = 1.e-2 
control_dt = 1.e-3 
sim_dt = 1.e-5

initial_covariance = 1.e-3 * np.eye(4)
process_noise = 1.e-3*np.eye(4)
# process_noise[1,1] = 1.e-2 
measurement_noise = 1.e-3*np.eye(4)
sensitivity =  .2

