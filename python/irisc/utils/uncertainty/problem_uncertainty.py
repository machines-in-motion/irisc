""" A class that holds process and measurement uncertainty models for the shooting problem """
import numpy as np 


class ProblemUncertainty: 
    def __init__(self, x0, initialUncertainty, modelUncertainties):
        """ A class that models process and measurement uncertainties for 
        the entire shooting problem as added Gaussian noise 
        Args:
            x0: initial state mean 
            winitialUncertainty: initial state covariance 
            initialUncertainty: list contaning 
            """
        self.x0 = x0 
        self.P0 = initialUncertainty 
        self.runningModels = modelUncertainties 
        self.runningDatas = []

        for uModel in self.runningModels: 
            self.runningDatas += [uModel.createData()]

    def calcDiff(self):
        """ calculates system approximations along a given trajectory """
        raise NotImplementedError("calcDiff is not Implemented for Problem Uncertainty")

        
