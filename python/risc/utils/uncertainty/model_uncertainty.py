"defines a single time step uncertainty model "

import numpy as np 




class ModelUncertainty: 
    def __init__(self, process_model, measurement_model): 
        """ A class the combines model uncertainty for both process and measurements """

        self.pModel = process_model
        self.mModel = measurement_model 
        self.Omega = None 
        self.invOmega = None 
        self.Gamma = None 
        self.invGamma = None 
        

