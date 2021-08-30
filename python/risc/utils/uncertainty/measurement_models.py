import numpy as np 


class AbstractMeasurementModel:
    def __init__(self, state_model):
        """ defines a full state measurement model 
        y_{t+1} = g(x_t, u_t) + \gamma_{t+1]
        Args: 
        state_model:  state model defined in crocoddyl 
        m_covariance: exactly \gamma_{t+1} in the above equation
        """
        self.state = state_model


    def calc(self, x, u): 
        raise NotImplementedError("calc method is not implemented for AbstractMeasurementModel")
    
    def calcDiff(self, x, u, recalc=False): 
        raise NotImplementedError("calcDiff method is not implemented for AbstractMeasurementModel")


class FullStateMeasurement(AbstractMeasurementModel):
    def __init__(self, state_model, m_covariance):
        super(FullStateMeasurement, self).__init__(state_model)
        self.Gamma = m_covariance
        self.H = np.eye(self.state.ndx) 
        try:
            self.invGamma = np.linalg.inv(m_covariance)
        except:
            raise BaseException("measurement covariance is not Positive Definite") 

    def calc(self, x, u): 
        """This whole thing here might not make sense unless 
        I assume calc as returning some disturbed measurement sample""" 
        pass 
        
    def calcDiff(self, x, u, recalc=False): 
        """ might change if g(x_t, u_t) is some nonlinear function, for now it is just the identity """
        if recalc:
            self.calc(x,u)

