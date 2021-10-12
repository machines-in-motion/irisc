import numpy as np 
import scipy.linalg as scl 
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
    
    def __call__(self, t, d, x, chi):
        raise NotImplementedError("call method not implemented for Abstract Controller")
    
        
class DDPController(AbstractController):
    def __init__(self, action_models, xs, us, K, dt=1.e-3):
        super().__init__(action_models, xs, us)
        self.K = K 
        self.dt = dt


    def __call__(self, t, d, x, chi):
        """ takes a feedback state, control index, and simulation index
        and returns a control signal u """ 
        xdes = self.interpolate_xs(t,d)
        err = self.compute_error(t, xdes, x)
        u = self.us[t] -self.K[t].dot(err)
        return u

    def reset(self):
        pass 
    


class RiskSensitiveController(AbstractController):
    def __init__(self, action_models, xs, us, K, V, v, sensitivity, dt=1.e-3):
        super().__init__(action_models, xs, us)
        self.dt = dt 
        self.K = K 
        self.V = V
        self.v = v 
        self.sigma = sensitivity
        self.K_opt = []
        self.k_opt = []

    def reset(self):
        self.K_opt = []
        self.k_opt = []
        


    def __call__(self, t, d, x, chi):
        xdes = self.interpolate_xs(t,d)
        err = self.compute_error(t, xdes, x)
        Lb = scl.cho_factor(np.eye(self.action_models[t].state.ndx) + self.sigma*chi.dot(self.V[t]),lower=True)
        k_right = chi.dot(self.v[t])
        k_pre = scl.cho_solve(Lb, k_right)
        k = self.sigma*self.K[t].dot(k_pre)
        K_right = np.eye(self.action_models[t].state.ndx) 
        K_pre = scl.cho_solve(Lb, K_right)
        K = self.K[t].dot(K_pre) 
        u = self.us[t] + k - K.dot(err)
        if d < 1.e-8:
            #store stuff 
            self.K_opt +=[K]
            self.k_opt += [k] 

        return u 
        
        