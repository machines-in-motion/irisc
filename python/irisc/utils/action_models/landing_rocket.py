""" A 2 dimensional rocket model with thrust vectoring, 
rocket is assumed to be rectanglular 
center of mass is assumed to be right at the center of the rectangle 
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
        self.mass = 10. 
        self.inv_m = 1./self.mass
        self.g = 9.81 

    def nonlinear_dynamics(self, x, u):
        acc = np.zeros(3)
        acc[0] = - self.inv_m*np.sin(x[2]+u[0])*u[1]
        acc[1] = self.inv_m*np.cos(x[2]+ u[0])*u[1] - self.g 
        acc[2] = - self.inv_inertia*self.d*np.sin(u[0])*u[1] # thrust torque at center of mass 
        return acc

    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u """
        dfdx = np.zeros([3,6])
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
        self.t = t 
        self.T = T 
        self.w = []
        # running cost only on controls 
        self.w += [1.e-1]  # w[0]
        self.w += [1.e-1]  # w[1]
        # running cost state weights 
        self.w += [1.e+3] # x pos  w[2] 
        self.w += [1.e+3] # y pos  w[3]
        self.w += [1.e+2] # orientation  w[4] 
        self.w += [1.e+2] # velcities   w[5]
        # terminal cost 
        self.w += [1.e+3] # w[6]
        self.w += [1.e+3] # w[7]
        self.w += [1.e+3] # w[8]
        self.w += [1.e+3] # w[9]

    def _running_cost(self, t, x, u):
        cost = self.w[0]*u[0]**2 + self.w[1]*u[1]**2   
        cost += self.w[2]*x[0]**2 # zero x pos 
        cost += self.w[3]*x[1]**2 # y position hover at desired height 
        cost += self.w[4]*x[2]**2 # zero orientation 
        cost += self.w[5]*x[3:].dot(x[3:]) # zero velocities for everything 
        return cost  
        

    def _terminal_cost(self, x):
        cost = self.w[6]*x[0]**2 # zero x pos 
        cost += self.w[7]*x[1]**2 # y position hover at desired height 
        cost += self.w[8]*x[2]**2 # zero orientation 
        cost += self.w[9]*x[3:].dot(x[3:]) # zero velocities for everything 
        return cost  

    def calc(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        
        if self.isTerminal:
            data.cost = self._terminal_cost(x)
            data.xout = np.zeros(self.state.ndx)
        else:
            data.cost = self._running_cost(self.t, x, u)
            data.xout = self.dynamics.nonlinear_dynamics(x,u)


    def calcDiff(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        # 
        Fx = np.zeros([self.state.ndx,self.state.ndx]) 
        Fu = np.zeros([self.state.ndx,self.nu])
        Lx = np.zeros([self.state.ndx])
        Lu = np.zeros([self.nu])
        Lxx = np.zeros([self.state.ndx, self.state.ndx])
        Luu = np.zeros([self.nu, self.nu])
        Lxu = np.zeros([self.state.ndx,self.nu])
        # 
        if self.isTerminal:
            Lx[0] = 2.*self.w[6]*x[0]
            Lxx[0,0] = 2.*self.w[6]
            Lx[1] = 2.*self.w[7]*x[1]
            Lxx[1,1] = 2.*self.w[7]
            Lx[2] = 2.*self.w[8]*x[2]
            Lxx[2,2] = 2.*self.w[8]
            for i in range(3,self.state.ndx):
                Lx[i] = 2.*self.w[9]*x[i]
                Lxx[i,i] =  2.*self.w[9]

        else:
            Fx, Fu = self.dynamics.derivatives(x,u)
            Lx[0] = 2.*self.w[2]*x[0]
            Lxx[0,0] = 2.*self.w[2]
            Lx[1] = 2.*self.w[3]*x[1]
            Lxx[1,1] = 2.*self.w[3]
            Lx[2] = 2.*self.w[4]*x[2]
            Lxx[2,2] = 2.*self.w[4]
            for i in range(3,self.state.ndx):
                Lx[i] = 2.*self.w[5]*x[i]
                Lxx[i,i] =  2.*self.w[5]
            Lu[0] = 2.*self.w[0]*u[0]
            Lu[1] = 2.*self.w[1]*u[1]
            Luu[0,0] = 2.*self.w[0]
            Luu[1,1] = 2.*self.w[1]
        
        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()



if __name__=="__main__": 

    x0 = np.array([10., 20., np.pi/6, 0., 0., 0.]) 
    MAX_ITER = 1000
    T = 200 
    dt = 1.e-2
    running_models = []
    for t in range(T): 
        diff_rocket = DifferentialActionModelRocket2D(t, T, False) 
        running_models += [crocoddyl.IntegratedActionModelEuler(diff_rocket, dt)] 
    diff_rocket = DifferentialActionModelRocket2D(T, T, True) 
    terminal_model = crocoddyl.IntegratedActionModelEuler(diff_rocket, dt) 

    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
    fddp = crocoddyl.SolverFDDP(problem)
    fddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])

    xs = [x0]*(T+1)
    us = [np.zeros(2)]*T

    converged = fddp.solve(xs,us, MAX_ITER)
    if not converged:
        print("Solver Did Not Converge !!")

    time_array = dt*np.arange(T+1)

    plt.figure("trajectory plot")
    plt.plot(np.array(fddp.xs)[:,0],np.array(fddp.xs)[:,1], label="Rocket Position")
    plt.gca().set_aspect('equal')

    plt.figure("rocket orientation")
    plt.plot(time_array,np.array(fddp.xs)[:,2], label="Orientation")


    plt.figure("velocities")
    plt.plot(time_array,np.array(fddp.xs)[:,3], label="vx")
    plt.plot(time_array,np.array(fddp.xs)[:,4], label="vy")
    plt.plot(time_array,np.array(fddp.xs)[:,5], label="w")

    plt.show()



