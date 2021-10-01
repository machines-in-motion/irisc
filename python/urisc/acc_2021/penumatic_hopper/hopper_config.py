""" ideally has all config parameters for the problem """
from pickle import FALSE
import numpy as np 
import os,sys 

LINE_WIDTH = 100 
PLOT_FIGS = True 
SAVE_SOLN = True

x0 = np.array([0.5, 0., 0., 0.])
MAX_ITER = 1000
horizon = 300 
dt = 1.e-2
time_array = dt*np.arange(horizon+1)


# SIMULATION PARAMETERS 

WHICH_CONTROLLER = "ddp"
WITH_NOISE = [False, False] # process and measurement noise flags  
solutions_path = os.path.abspath('solutions/')