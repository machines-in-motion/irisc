""" here we will have two different dynamics 
Nominal for Optimization 
Sloch for actual simulation 
model inspired from "Direct Policy Optimization Using Deterministic Sampling and Collocation" by Howell et. al. 
link: https://ieeexplore.ieee.org/abstract/document/9387078 
"""



from types import DynamicClassAttribute
import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 

class NominalRocket2D:
    def __init__(self):
        self.l = 3. # length of rocket assumed to be rectangle 
        self.w = 1. # width of the rectangle 
        self.d = .5*self.l # distance to thruster from COM 
        self.inertia = (1./12.) * self.w * self.l**3 # rectangle inertia around center of mass 
        self.inv_inertia = 1./self.inertia
        self.mass = 100. 
        self.inv_m = 1./self.m
        self.g = 9.81 

    def nonlinear_dynamics(self, x, u):
        acc = np.zeros(3)
        acc[0] = - self.inv_m*np.sin(x[2]+u[0])*u[1]
        acc[1] = self.inv_m*np.cos(x[2]+ u[0])*u[1] - self.g 
        acc[2] = - self.inv_inertia*self.d*np.sin(u[0])*u[1] # thrust torque at center of mass 
        return acc

    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u """
        dfdx = np.zeros([3,3])
        dfdu = np.zeros([3,2])
        # 
        dfdx[0,2] = -self.inv_m*u[1]*np.cos(x[2] + u[0]) 
        dfdx[1,2] = -self.inv_m*u[1]*np.sin(x[2] + u[0])
        # 
        b = self.inv_inertia * self.d
        dfdu[0,0] = -self.inv_m*u[1]*np.cos(x[2] + u[0])
        dfdu[1,0] = -self.inv_m*u[1]*np.sin(x[2] + u[0])
        dfdu[2,0] = -b*u[1]*np.cos(u[0])
        dfdu[0,1] = -self.inv_m*np.sin(x[2] + u[0])
        dfdu[1,1] = self.inv_m*np.cos(x[2] + u[0])
        dfdu[2,1] = - b*np.sin(u[0])
        return dfdx, dfdu

    def com_transform_3d(self, x, u):
        # homogeneous transfrom for the com 
        m = np.eye(4)
        return m 

    def thrust_transform_3d(self, x, u): 
        m = np.eye(4)
        return m


class DifferentialActionModelRocket2D(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, t, T, isTerminal=False):
        self.dynamics = NominalRocket2D()
        nq = 3 
        nv = 3 
        nx = nv + nq 
        ndx = nx 
        nu = 2
        state =  crocoddyl.StateVector(nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, ndx)

        self.isTerminal = isTerminal

        self.w = []
        # running cost only on controls 
        self.w += [1.e+1]
        self.w += [1.e+1]
        # terminal cost is desired state 

    def _running_cost(self, t, x, u):
        cost = self.w[0]*u[0]**2 + self.w[1]*u[1]**2   
        return cost 

    def _terminal_cost(self, x):
        cost = self.w[2]*x[0]**2 # zero x pos 
        cost += self.w[3]*(x[1]-self.dynamics.d)**2 # y position hover at desired height 
        cost += self.w[4]*x[2]**2 # zero orientation 
        cost += self.w[5]*x[3:].dot(x[3:]) # zero velocities for everything 
        return cost  

    def calc(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        
        if self.isTerminal:
            data.cost = self._terminal_cost(x)
            data.xout = np.zeros(self.state.ndx)


    def calcDiff(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)

        

        if self.isTerminal:
            pass
        else:
            Fx, Fu = self.dynamics.derivatives(x,u)
        
        data.Fx = Fx.copy()
        data.Fu = Fu.copy()


