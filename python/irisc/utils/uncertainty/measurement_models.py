import numpy as np 


class AbstractMeasurementModel:
    def __init__(self, action_model):
        """ defines a full state measurement model 
        y_{t+1} = g(x_t, u_t) + \gamma_{t+1]
        Args: 
        state_model:  state model defined in crocoddyl 
        m_covariance: exactly \gamma_{t+1} in the above equation
        """
        self.model = action_model
        self.state = self.model.state

    def calc(self, x, u): 
        raise NotImplementedError("calc method is not implemented for AbstractMeasurementModel")
    
    def calcDiff(self, x, u, recalc=False): 
        raise NotImplementedError("calcDiff method is not implemented for AbstractMeasurementModel")


class FullStateMeasurement(AbstractMeasurementModel):
    def __init__(self, integrated_action, m_covariance):
        """
        Args: 
        filter: in different models this will be a matrix multiplying the noise vector, i.e. C_t * gamma_t  
        """
        super(FullStateMeasurement, self).__init__(integrated_action)
        self.Gamma = m_covariance
        self.H = np.eye(self.state.ndx) 
        self.ny = self.state.ndx 
        self.filter = np.eye(self.state.ndx)

    def calc(self, x, u): 
        """This whole thing here might not make sense unless 
        I assume calc as returning some disturbed measurement sample""" 
        y = np.zeros(self.ny)
        return y

    def deviation(self, y, xn): 
        return self.state.diff(xn, y)
    
    def integrate(self, y, dy):
        return self.state.integrate(y, dy)
        
    def calcDiff(self, x, u, recalc=False): 
        """ might change if g(x_t, u_t) is some nonlinear function, for now it is just the identity """
        if recalc:
            self.calc(x,u)

