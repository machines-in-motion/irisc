import numpy as np 


class AbstractProcessModel:
    def __init__(self, action_model, state_model):
        """ defines uncertainty in the process model as  
        y_{t+1} = f(x_t, u_t) + \gamma_{t+1]
        Args: 
        action_model: crocoddyl integrated action model 
        state_model: state model defined in crocoddyl 
        """
        self.model = action_model 
        self.state = state_model


    def calc(self, x, u): 
        raise NotImplementedError("calc method is not implemented for AbstractProcessModel")
    
    def calcDiff(self, x, u, recalc=False): 
        raise NotImplementedError("calcDiff method is not implemented for AbstractProcessModel")


class FullStateProcess(AbstractProcessModel):
    def __init__(self, robot_model, state_model, p_covariance):
        """ adds noise without a transformation on the full state """
        super(FullStateProcess, self).__init__(robot_model, state_model)
        self.Omega = p_covariance
        try:
            self.invOmega = np.linalg.inv(p_covariance)
        except:
            raise BaseException("process covariance is not Positive Definite") 

    def calc(self, x, u): 
        """This whole thing here might not make sense unless 
        I assume calc as returning some disturbed measurement sample""" 
        pass 
        
    def calcDiff(self, x, u, recalc=False): 
        """ might change if g(x_t, u_t) is some nonlinear function, for now it is just the identity """
        if recalc:
            self.calc(x,u)

