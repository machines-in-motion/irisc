""" model inspired from Sideris and Bobrow """


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 


class PenumaticHopped1D:
    def __init__(self):
        self.g = 9.81 
        self.mass = 1. 
        self.k = 500. 
        self.alpha = 1.e-2 
        self.inv_m = 1./self.mass 

    def nonlinear_dynamics(self, x, u):
        acc = np.zeros(2)
        e = x[1] - x[0]
        acc[1] = u[0] 
        if e < 0.:
            acc[0] = -self.g
        elif e >= 0 and e < self.alpha:
            scale = .5*self.k / self.alpha
            acc[0] = self.inv_m*scale*(e**2)  - self.g
        else:
            acc[0] = self.inv_m*self.k*e - .5*self.inv_m*self.k*self.alpha  - self.g
        return acc

    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u """
        dfdx = np.zeros([2,4])
        dfdu = np.zeros([2,1])
        # 
        e = x[1] - x[0]
        if e<0.:
            pass                  
        elif e >= 0. and e < self.alpha:
            dfdx[0,0] = -self.inv_m*(self.k/self.alpha)*e 
            dfdx[0,1] = self.inv_m*(self.k/self.alpha)*e 
        else:
            dfdx[0,0] = -self.inv_m*self.k 
            dfdx[0,1] = self.inv_m*self.k
        # 
        dfdu[1,0] = 1. 
        return dfdx, dfdu








class DifferentialActionModelHopper(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, t, T, isTerminal=False):
        nq = 2
        nv = 2 
        nx = nv + nq 
        ndx = nx 
        nu = 1 
        state =  crocoddyl.StateVector(nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, ndx)
        self.dynamics = PenumaticHopped1D()
        # self.g = - 9.81 
        self.isTerminal = isTerminal
        # self.mass = 1. 
        # self.k = 500. 
        # self.alpha = .01
        self.z_des = 3.
        self.scale = .5/T  
        #
        self.t = t  # will be used to scale costs 
        self.T = T  # horizon
        self.t1 = int(T/2) - 2 # phase one 
        self.t2 = self.t1 + 2
        self.w = [] 
        # running cost weights 
        self.w += [1.e+2] # piston position  w[0]
        self.w += [1.e+0] # control  w[1]
        # jump phase 
        self.w += [1.e+3] # mass height  w[2]
        self.w += [1.e+3] # mass velocity w[3] 
        self.w += [1.e+1] # piston position  w[4]
        self.w += [1.e+1] # control weight 
        # terminal 
        self.w += [1.e+2] # mass position 
        self.w += [1.e+2] # piston position 
        self.w += [1.e+2] # mass and piston velocties 
        

    def _running_cost(self, t, x, u): 
        if t<= self.t1 or t > self.t2:
            cost = self.scale*self.w[0]*x[1]**2 + self.scale*self.w[1]*u[0]**2 
        else:
            cost = .5*self.w[2]*(x[0]-self.z_des)**2 + self.w[3]*x[2]**2 
            cost +=  self.w[4]*x[1]**2 + self.w[5]*u[0]**2
        return cost 
    
    def _terminal_cost(self, x):
        cost = self.w[6]*x[0]**2 + self.w[7]*x[1]**2 + self.w[8]*x[2:].dot(x[2:])
        return cost 

    def calc(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        #
        if self.isTerminal: 
            data.cost = self._terminal_cost(x) 
            data.xout = np.zeros(2)
        else:
            data.cost = self._running_cost(self.t, x, u)
            data.xout[:] = self.dynamics.nonlinear_dynamics(x,u)


    def calcDiff(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        #
        Fx = np.zeros([2,4]) 
        Fu = np.zeros([2,1])
        Lx = np.zeros([4])
        Lu = np.zeros([1])
        Lxx = np.zeros([4,4])
        Luu = np.zeros([1,1])
        Lxu = np.zeros([4,1])
        # COST DERIVATIVES 
        if self.isTerminal:
            Lx[0] = 2.*self.w[6]*x[0]
            Lxx[0,0] = 2.*self.w[6]  
            Lx[1] = 2.*self.w[7]*x[1]
            Lxx[1,1] = 2.*self.w[7]
            Lx[2] = 2.*self.w[8]*x[2]
            Lxx[2,2] = 2.*self.w[8]
            Lx[3] = 2.*self.w[8]*x[3]
            Lxx[3,3] = 2.*self.w[8]
        else:
            t = self.t 
            if t<= self.t1 or t > self.t2:
                Lu[0] = 2*self.scale*self.w[1]*u[0]
                Luu[0,0] = 2*self.scale*self.w[1]
                Lx[1] = 2.* self.scale*self.w[0]*x[1]
                Lxx[1,1] = 2.* self.scale*self.w[0] 
            else:
                Lu[0] = 2.*self.w[5]*u[0]
                Luu[0,0] = 2.*self.w[5]
                Lx[0] = self.w[2]*(x[0]-self.z_des) 
                Lxx[0,0] = self.w[2]
                Lx[1] = 2.*self.w[4]*x[1]
                Lxx[1,1] = 2.*self.w[4]
                Lx[2] = 2.*self.w[3]*x[2] 
                Lxx[2,2] = 2.*self.w[3]
            # dynamics derivatives 
            Fx, Fu = self.dynamics.derivatives(x,u)
        # COPY TO DATA 
        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()




if __name__ =="__main__":
    print(" Testing Penumatic Hopper with DDP ".center(LINE_WIDTH, '#'))
    x0 = np.array([0., 0., 0., 0.])
    MAX_ITER = 1000
    T = 300 
    dt = 1.e-2
    running_models = []
    for t in range(T): 
        diff_hopper = DifferentialActionModelHopper(t, T, False) 
        running_models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
    diff_hopper = DifferentialActionModelHopper(T, T, True) 
    terminal_model = crocoddyl.IntegratedActionModelEuler(diff_hopper, dt) 
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

    fddp = crocoddyl.SolverFDDP(problem)
    fddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(T+1)
    us = [np.array([0.])]*T

    converged = fddp.solve(xs,us, MAX_ITER)

    if not converged:
        print("Solver Did Not Converge !!")

    time_array = dt*np.arange(T+1)

    plt.figure("trajectory plot")
    plt.plot(time_array,np.array(fddp.xs)[:,0], label="Mass Height")
    plt.plot(time_array,np.array(fddp.xs)[:,1], label="Piston Height")

    plt.show()



    
            
