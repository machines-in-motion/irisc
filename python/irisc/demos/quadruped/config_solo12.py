import numpy as np 


class BalanceConfig(object):
    def __init__(self, whichMeasurement=None):
        self.timeStep = 1.e-2
        self.horizon = 500 
        self.measurementModel= whichMeasurement
        self.q0 = np.array(
                    [0.2, 0.0, 0.25, 0.0, 0.0, 0.0, 1.0]
                    + 2 * [0.0, 0.8, -1.6]
                    + 2 * [0.0, -0.8, 1.6])
        self.v0 = np.zeros(18)
        self.x0 = np.hstack([self.q0, self.v0])    
        self.initial_covariance = 1.e-7 * np.eye(36)
        self.process_noise = 1.e-7*np.eye(36)
        # process_noise[1,1] = 1.e-2 
        self.measurement_noise = 1.e-5*np.eye(36)
        self.sensitivity = -.01 





class JumpConfig(object):
    timeStep = 1.e-2 
    jumpHeight = 1. 
    horizon = 500 # 1 meter height might take some time to land 
    