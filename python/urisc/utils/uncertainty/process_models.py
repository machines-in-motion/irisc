import numpy as np 


class AbstractProcessModel:
    def __init__(self, action_model):
        """ defines uncertainty in the process model as  
        y_{t+1} = f(x_t, u_t) + \gamma_{t+1]
        Args: 
        action_model: crocoddyl integrated action model 
        state_model: state model defined in crocoddyl 
        """
        self.model = action_model 
        self.state = self.model.state 

    def calc(self, x, u): 
        raise NotImplementedError("calc method is not implemented for AbstractProcessModel")
    
    def calcDiff(self, x, u, recalc=False): 
        raise NotImplementedError("calcDiff method is not implemented for AbstractProcessModel")


class FullStateProcess(AbstractProcessModel):
    def __init__(self, integrated_action, p_covariance):
        """ adds noise without a transformation on the full state """
        super(FullStateProcess, self).__init__(integrated_action)
        self.Omega = p_covariance
        self.filter = np.eye(self.state.ndx)

    def calc(self, x, u): 
        """This whole thing here might not make sense unless 
        I assume calc as returning some disturbed measurement sample""" 
        pass 
        
    def calcDiff(self, x, u, recalc=False): 
        """ might change if g(x_t, u_t) is some nonlinear function, for now it is just the identity """
        if recalc:
            self.calc(x,u)

