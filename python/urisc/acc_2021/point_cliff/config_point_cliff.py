import numpy as np 

plan_dt = 1.e-2 
control_dt = 1.e-3 
sim_dt = 1.e-4

horizon = 100


horizon = 100 
x0 = np.zeros(4)
initial_covariance = 1.e-3 * np.eye(4)
process_noise = 1.e-3*np.eye(4)
# process_noise[1,1] = 1.e-2 
measurement_noise = 1.e-3*np.eye(4)
sensitivity = 1.e-2


MAX_ITER = 100 
PLOT_SOLN = True
SAVE_SOLN = True 
LINE_WIDTH = 100