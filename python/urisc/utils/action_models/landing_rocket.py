""" here we will have two different dynamics 
Nominal for Optimization 
Sloch for actual simulation 
model inspired from "Direct Policy Optimization Using Deterministic Sampling and Collocation" by Howell et. al. 
link: https://ieeexplore.ieee.org/abstract/document/9387078 
"""



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



    def _running_cost(self, t, x, u):
        pass 

    def _terminal_cost(self, t, x, u):
        pass 


    def calc(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)


    def calcDiff(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)

