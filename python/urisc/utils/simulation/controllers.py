import numpy as np 

class AbstractController:
    def __init__(self, action_models, xs, us):
        self.action_models = action_models
        self.states = []
        for m in self.action_models: 
            self.states += [m.state]
        self.xs = xs 
        self.us = us
        self.horizon = len(self.xs) -1

    def interpolate_xs(self, t, d):
        if t == self.horizon:
            # terminal state, no further interpolation 
            return self.xs[t]
        else: 
            dx = self.states[t].diff(self.xs[t], self.xs[t+1])
            return self.states[t].integrate(self.xs[t], d*dx)  

    def compute_error(self, t, xs, xa):
        """ computes error between reference state and
        actual state, accounts lie group stuff 
        Args:
            t: control horizon index 
            xs: reference state 
            xa: actual state 
        """
        return self.states[t].diff(xs, xa) 
    
        
class DDPController(AbstractController):
    def __init__(self, action_models, xs, us, K, dt=1.e-2):
        super().__init__(action_models, xs, us)
        self.K = K 
        self.dt = dt

    def __call__(self, t, d, x):
        """ takes a feedback state, control index, and simulation index
        and returns a control signal u """ 
        xdes = self.interpolate_xs(t,d)
        err = self.compute_error(t, xdes, x)
        u = self.us[t] -self.K[t].dot(err)
        return u

    



def load_ddp_controller(path, action_models):
    pass 