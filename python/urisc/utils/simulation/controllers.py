import numpy as np 


class AbstractController:
    def __init__(self, action_models, xs, us):
        self.action_models = action_models
        self.states = []
        for m in self.action_models: 
            self.states += [m.state]
        self.xs = xs 
        self.us = us
        self.T = len(self.xs)

    def interpolate_xs(self, t, d):
        pass 

    def interpolate_us(self, t, d):
        pass

    def compute_error(self, t, xs, xa):
        """ computes error between reference state and
        actual state, accounts lie group stuff 
        Args:
            t: control horizon index 
            xs: reference state 
            xa: actual state 
        """
        pass 
    
        
class DDPController(AbstractController):
    def __init__(self, action_models, xs, us, K):
        super().__init__(action_models, xs, us)
        self.K = K 

    def __call__(self, t, d, x):
        """ takes a feedback state, control index, and simulation index
        and returns a control signal u """ 
        pass 

    



def load_ddp_controller(path, action_models):
    pass 