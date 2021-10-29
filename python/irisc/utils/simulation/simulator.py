""" 
ConSim Simulator for Articulated 3D Robots 
One Dimensional Hopping Simulator """

from types import DynamicClassAttribute
import numpy as np 


class AbstractSimulator:
    def __init__(self, dynamics, controller, estimator, sim_dt):
        self.dynamics = dynamics
        self.sim_dt = sim_dt 
        self.controller = controller
        self.estimator = estimator 
        self.horizon = self.controller.horizon

    def step(self):
        raise NotImplementedError("method step not implemented for AbstractSimulator")




class HopperSimulator(AbstractSimulator):
    def __init__(self, dynamics, controller,estimator, x0, horizon,  plan_dt, control_dt, sim_dt):
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
        
        self.g = 9.81 
        self.mass  = dynamics.mass  
        self.inv_m = 1./self.mass  

        # few simulation parameters 
        self.k = 1.e+5 
        self.b = 300.  
        self.env = 0.
        self.x0[0] += self.env

        self.plan_dt = plan_dt
        self.controller_dt = control_dt # sim_dt
        self.n_control_steps = int(self.plan_dt/self.controller_dt)
        self.n_sim_steps = int(self.controller_dt/self.sim_dt)
        if self.n_sim_steps == 0:
            self.n_sim_steps = 1
        self.horion = horizon
        self.x0 = x0 
        self.process_models = self.estimator.processModels 
        self.uncertainty_models = self.estimator.uncertaintyModels 
        self.process_datas = self.estimator.processDatas
        self.uncertainty_datas = self.estimator.uncertaintyDatas

        self.xsim = [] # true states 
        self.usim = [] # control inputs 
        self.fsim = [] # contact forces  
        self.ysim = [] # observations 
        self.xhsim = [self.estimator.xhat[0]] # estimated states 
        self.chi = [self.estimator.chi[0]]


    def reset(self):
        self.xsim = [] # true states 
        self.usim = [] # control inputs 
        self.fsim = [] # contact forces  
        self.ysim = [] # observations 
        self.xhsim = [self.estimator.xhat[0]] # estimated states 
        self.chi = [self.estimator.chi[0]]
        self.controller.reset()
        self.estimator.reset()


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
        dv = self.sim_dt*dv 
        xnext[:2] = x[:2] + self.sim_dt*x[2:] + .5*self.sim_dt*dv
        xnext[2:] = x[2:] + dv
        return xnext, fc 


    def simulate(self, env=None): 
        """ env here is dictionary that defines an environment chainge at a certain time """
        self.xsim += [self.x0.copy()]
        xi = self.x0.copy()
        xh = self.x0.copy()
        chi = self.chi[0].copy()
        if env is not None:
            new_env = env["height"]
            env_time = env["time"]
        for t in range(self.horizon):
            if env is not None:
                if t == env_time:
                    self.env = new_env
            for i in range(self.n_control_steps):
                ui = self.controller(t, i/self.n_control_steps, xh, chi)
                for _ in range(self.n_sim_steps):
                    xi, fi = self.step(xi, ui) 
                
                if self.estimator is not None:
                    umodel = self.uncertainty_models[t][i]
                    udata = self.uncertainty_datas[t][i]
                    xi = umodel.sample_process(udata, xi, ui)
                    yi = umodel.sample_measurement(udata, xi, ui) 
                    self.estimator.update(t,i,yi,xi,ui)
                    xh = self.estimator.xhat[-1].copy()
                    chi = self.estimator.chi[-1].copy()
                else:
                    xh = xi.copy()
            self.fsim += [fi]

            self.usim += [ui]            
            self.xsim += [xi]
            self.xhsim += [self.estimator.xhat[-1]] # estimated states 
            self.chi += [self.estimator.chi[-1]]


class PointMassSimulator(AbstractSimulator):
    def __init__(self, dynamics, controller, estimator, x0, horizon,  plan_dt, control_dt, sim_dt):
        super().__init__(dynamics, controller, estimator, sim_dt)

        
        self.plan_dt = plan_dt
        self.controller_dt = control_dt # sim_dt
        self.n_control_steps = int(self.plan_dt/self.controller_dt)
        self.n_sim_steps = int(self.controller_dt/self.sim_dt)
        if self.n_sim_steps == 0:
            self.n_sim_steps = 1
        self.horion = horizon
        self.x0 = x0 
        self.process_models = self.estimator.processModels 
        self.uncertainty_models = self.estimator.uncertaintyModels 
        self.process_datas = self.estimator.processDatas
        self.uncertainty_datas = self.estimator.uncertaintyDatas

        self.xsim = [] # true states 
        self.usim = [] # control inputs 
        self.fsim = [] # contact forces  
        self.ysim = [] # observations 
        self.xhsim = [self.estimator.xhat[0]] # estimated states 
        self.chi = [self.estimator.chi[0]]

        self.withEstimation = 1

    def step(self, x, u):
        dv = self.sim_dt*self.dynamics(x, u)
        xnext = np.zeros(4)
        xnext[:2] = x[:2] + self.sim_dt*x[2:] + .5*self.sim_dt*dv
        xnext[2:] =  x[2:] + dv 
        return xnext


    def simulate(self):
        self.xsim += [self.x0.copy()]
        xi = self.x0.copy()
        xh = self.x0.copy()
        chi = self.chi[0].copy()
        for t in range(self.horizon):
            for i in range(self.n_control_steps):
                ui = self.controller(t, i/self.n_control_steps, xh, chi)
                for _ in range(self.n_sim_steps):
                    xi = self.step(xi, ui)
                if self.estimator is not None:
                    umodel = self.uncertainty_models[t][i]
                    udata = self.uncertainty_datas[t][i]
                    xi = umodel.sample_process(udata, xi, ui)
                    yi = umodel.sample_measurement(udata, xi, ui) 
                    self.estimator.update(t,i,yi,xi,ui)
                    xh = self.estimator.xhat[-1].copy()
                    chi = self.estimator.chi[-1].copy()
                else:
                    xh = xi.copy()

            self.usim += [ui]            
            self.xsim += [xi]
            self.xhsim += [self.estimator.xhat[-1]] # estimated states 
            self.chi += [self.estimator.chi[-1]]