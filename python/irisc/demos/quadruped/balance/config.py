# contains config parameters for solo balancing 

import numpy as np 

timeStep = 1.e-2
horizon = 500 
measurementModel= "FullStateUniform" 
q0 = np.array(
       [0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
       + 2 * [0.0, 0.8, -1.6]
       + 2 * [0.0, -0.8, 1.6])
v0 = np.zeros(18)
x0 = np.hstack([q0, v0])    
initial_covariance = 1.e-7 * np.eye(36)
process_noise = 1.e-7*np.eye(36)
process_noise[1,1] = 1.e-2 
measurement_noise = 1.e-5*np.eye(36)
sensitivity = -.01 
