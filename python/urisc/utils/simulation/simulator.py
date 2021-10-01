""" 
ConSim Simulator for Articulated 3D Robots 
One Dimensional Hopping Simulator """

import numpy as np 


class AbstractSimulator:
    def __init__(self, dynamics, controller, estimator, sim_dt):
        self.dynamics = dynamics
        self.dt = sim_dt 
        self.controller = controller
        self.estimator = estimator 
        self.horizon = self.controller.horizon

    def step(self):
        raise NotImplementedError("method step not implemented for AbstractSimulator")




class HopperSimulator(AbstractSimulator):
    def __init__(self, dynamics, controller, estimator, process_noise, 
                measurement_noise, x0, horizon, sim_dt=1.e-4):
        """ 1D penumatic hopper simulator, using stiff visco-elastic contacts 
        Args: 
            model:
            controller:
            estimator:
            terrain:
            x0:
            horizon: 
            sim_dt: 
        """
        super().__init__(dynamics, controller, estimator, sim_dt)
        self.horion = horizon
        self.x0 = x0 
        self.xsim = [] # true states 
        self.usim = [] # control inputs 
        self.fsim = [] # contact forces  
        self.ysim = [] # observations 
        self.xhsim = [] # estimated states 
        self.g = 9.81 
        self.mass  = dynamics.mass  
        self.inv_m = 1./self.mass  

        # few simulation parameters 
        self.k = 1.e+5 
        self.b = 300.  
        self.env = 0.
        self.x0[0] += self.env
        self.controller_dt = self.controller.dt  
        self.n_steps = int(self.controller_dt/self.dt)
        self.process_noise = process_noise 
        self.measurement_noise = measurement_noise
        # 
        if self.process_noise is not None:
            self.pnoise_flag = True 
        else:
            self.pnoise_flag = False 
        if self.measurement_noise is not None:
            self.mnoise_flag = True 
        else: 
            self.mnoise_flag = False 

    def step(self, x, u):
        """ computes one simulation step """
        dv = np.zeros(2)
        xnext = np.zeros(4)
        vnet = x[2]-x[3] 
        if vnet > 0.:
            fc = 0. 
        else:
             
            xc = x[0] - x[1] - self.dynamics.d0 # contact point height 
            if xc > self.env:
                fc = 0. 
            else:
                err = self.env - xc 
                fc = self.k*err - self.b*vnet
        
        dv[0] = self.inv_m*fc - self.g  
        dv[1] = u[0] 
        xnext[:2] = x[:2] + self.dt*x[2:] + .5*dv*self.dt**2 
        xnext[2:] = x[2:] + self.dt*dv
        return xnext, fc 


    def simulate(self): 
        self.xsim += [self.x0.copy()]
        xi = self.x0.copy()
        for t in range(self.horizon):
            for i in range(self.n_steps):
                ui = self.controller(t, i/self.n_steps, xi)
                xi, fi = self.step(xi, ui) 
                self.fsim += [fi]
            self.xsim += [xi]
            self.usim += [ui]
            
