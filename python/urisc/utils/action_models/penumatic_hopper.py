""" model inspired from Sideris and Bobrow """


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 


class PenumaticHopped1D:
    def __init__(self):
        self.g = 9.81 
        self.mass = 2. 
        self.k = 500. 
        self.alpha = 1.e-2
        self.inv_m = 1./self.mass 
        self.d0 = .5 # nominal position of foot relative to mass 

    def nonlinear_dynamics(self, x, u):
        acc = np.zeros(2)
        e = x[1] - x[0] + self.d0
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
        e = x[1] - x[0] + self.d0
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

    def discrete_dynamics(self, x, u, dt):
        """ computes state transitions for a given dt """
        dv = self.nonlinear_dynamics(x,u) 
        qnext = x[:2] + dt*x[2:] + .5*dv*dt**2 
        vnext = x[2:] + dt*dv 
        return qnext, vnext 

    def contact_force(self, x):
        e = x[1] - x[0] + self.d0
        if e < 0.:
            f = 0.
        elif e >= 0 and e < self.alpha:
            scale = .5*self.k / self.alpha
            f = scale*(e**2)  
        else:
            f = self.inv_m*self.k*e - .5*self.inv_m*self.k*self.alpha  
            
        return f 





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
        self.isTerminal = isTerminal
        self.z_des = 2.
        self.scale = .5/T  
        #
        self.t = t  # will be used to scale costs 
        self.T = T  # horizon
        self.t1 = int(T/2) - 2 # phase one 
        self.t2 = self.t1 + 4
        self.w = [] 
        # running cost weights 
        self.w += [1.e+1] # mass position  w[0]
        self.w += [1.e+1] # piston position  w[1]
        self.w += [1.e-2] # control w[2]
        # jump phase 
        self.w += [1.e+2] # mass height  w[3]
        self.w += [1.e-1] # piston position w[4] 
        self.w += [1.e+1] # mass velocity  w[5]
        self.w += [1.e-2] # control weight w[6]
        # terminal 
        self.w += [1.e+1] # mass position w[7]
        self.w += [1.e+1] # piston position w[8]
        self.w += [1.e+1] # mass and piston velocties w[9] 
        # extra term to phase 1 

        self.w += [1.e-5] # w[10]

        self.cost_scale = 1.e-1        
        

    def _running_cost(self, t, x, u): 
        if t<= self.t1 or t > self.t2:
            cost = self.scale*self.w[0]*(x[0]-self.dynamics.d0)**2 + self.scale*self.w[1]*(x[1]-self.dynamics.alpha)**2
            cost += self.scale*self.w[2]*u[0]**2 + self.scale*self.w[10]*x[2]**2  
        else:
            cost = .5*self.w[3]*(x[0]-self.z_des)**2 + self.w[4]*x[1]**2 
            cost +=  self.w[5]*x[2]**2 + self.w[6]*u[0]**2
        cost *= self.cost_scale
        return cost 
    
    def _terminal_cost(self, x):
        cost = self.w[7]*(x[0]-self.dynamics.d0)**2 + self.w[8]*x[1]**2 + self.w[9]*x[2:].dot(x[2:])
        cost *= self.cost_scale
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
            Lx[0] = 2.*self.w[7]*(x[0] - self.dynamics.d0)
            Lxx[0,0] = 2.*self.w[7]  
            Lx[1] = 2.*self.w[8]*x[1]
            Lxx[1,1] = 2.*self.w[8]
            Lx[2] = 2.*self.w[9]*x[2]
            Lxx[2,2] = 2.*self.w[9]
            Lx[3] = 2.*self.w[9]*x[3]
            Lxx[3,3] = 2.*self.w[9]
            Lx *= self.cost_scale 
            Lxx *= self.cost_scale
        else:
            t = self.t 
            if t<= self.t1 or t > self.t2:
                Lx[0] = 2.*self.scale * self.w[0]*(x[0]-self.dynamics.d0)
                Lxx[0,0] = 2.*self.scale * self.w[0]
                Lx[1] = 2.* self.scale*self.w[1]*(x[1]-self.dynamics.alpha)
                Lxx[1,1] = 2.* self.scale*self.w[1] 
                Lu[0] = 2*self.scale*self.w[2]*u[0]
                Luu[0,0] = 2*self.scale*self.w[2]
                Lx[2] = 2.*self.scale*self.w[10]*x[2]
                Lxx[2,2] = 2.*self.scale*self.w[10]
            else:
                Lu[0] = 2.*self.w[6]*u[0]
                Luu[0,0] = 2.*self.w[6]
                Lx[0] = self.w[3]*(x[0]-self.z_des) 
                Lxx[0,0] = self.w[3]
                Lx[1] = 2.*self.w[4]*x[1]
                Lxx[1,1] = 2.*self.w[4]
                Lx[2] = 2.*self.w[5]*x[2] 
                Lxx[2,2] = 2.*self.w[5]
            
            Lx *= self.cost_scale 
            Lxx *= self.cost_scale 
            Lu *= self.cost_scale 
            Luu *= self.cost_scale 
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
    x0 = np.array([.5, 0., 0., 0.])
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
    deltas = np.array(fddp.xs)[:,1] - np.array(fddp.xs)[:,0] + np.ones_like(time_array)
    plt.figure("trajectory plot")
    plt.plot(time_array,np.array(fddp.xs)[:,0], label="Mass Height")
    plt.plot(time_array,np.array(fddp.xs)[:,1], label="Piston Height")
    plt.plot(time_array, 0.*deltas, '--k', linewidth=2., label="ground")

    
    plt.figure("Penetration")
    plt.plot(time_array, .5*deltas)


    plt.figure("control inputs")
    plt.plot(time_array[:-1],np.array(fddp.us)[:,0], label="control inputs")


    plt.figure("feedback gains")
    for i in range(4):
        plt.plot(time_array[:-1],np.array(fddp.K)[:,i], label="$K_%s$"%i)
    plt.legend()

    plt.show()



    
            
