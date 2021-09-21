""" model inspired from Sideris and Bobrow """


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 

T = 100
dt = 1.e-2 

class DifferentialActionModelHopper(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, isTerminal=False):
        nq = 2
        nv = 2 
        nx = nv + nq 
        ndx = nx 
        nu = 1 
        state =  crocoddyl.StateVector(nx)
        crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, ndx)
        self.g = - 9.81 
        self.isTerminal = isTerminal
        self.mass = 1. 
        self.k = 500. 
        self.alpha = 0.1 
        self.y_des = 20. 
        self.scale = 1./(2.*T) # (dt*N)/(2*N) = dt/2 
    


    def _running_cost(self, x, u):
        cost = self.scale*1000.*x[1]**2 + self.scale*100.*x[3]**2 + self.scale*1.e-2*u[0]**2 
        return cost

    def _terminal_cost(self, x, u): 
        cost = 500.*(x[0]-self.y_des)**2 + x[2]**2   
        return cost 

    def calc(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        #
        if self.isTerminal: 
            data.cost = self._terminal_cost(x,u) 
            data.xout = np.zeros(2)
        else:
            data.cost = self._running_cost(x,u)
            # 
            e = x[1] - x[0]
            data.xout[1] = u[0] 
            if e < 0.:
                data.xout[0] = self.g
            elif e >= 0 and e < self.alpha:
                scale = self.k/(2.*self.alpha) 
                data.xout[0] = scale*(e**2)  + self.g
            else:
                data.xout[0] = self.k*e - .5*self.k*self.alpha  + self.g

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
        # 
        if self.isTerminal: 
            Lx[0] = 1000.*(x[0]-self.y_des)
            Lx[1] = 0.
            Lx[2] = 2.*x[2]
            Lx[3] = 0.
            Lxx[0,0] = 1000. 
            Lxx[1,1] = 0.
            Lxx[2,2] = 2.
            Lxx[3,3] = 0.
        else:
            Lx[1] = 2.*self.scale*1000.*x[1] 
            Lx[3] = 2.*self.scale*100.*x[3] 
            Lu[0] = 2.*self.scale*1.e-2*u[0]
            Lxx[1,1] = 2.*self.scale*1000. 
            Lxx[3,3] = 2.*self.scale*100. 
            Luu[0,0] = 2.*self.scale*1.e-2
            Fu[0,0] = 0. 
            Fu[1,0] = 1. 
            e = x[1] - x[0]
            if e<0.:
                pass                  
            elif e >= 0 and e < self.alpha:
                Fx[0,0] = (self.k/self.alpha)*(x[0]-x[1])
                Fx[0,1] = (self.k/self.alpha)*(x[1]-x[0])
            else:
                Fx[0,0] = -self.k*x[0] - .5*self.alpha*self.k 
                Fx[0,1] = self.k*x[1] - .5*self.alpha*self.k 


        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()



if __name__ =="__main__":
    print(" Testing Penumatic Hopper with DDP ".center(LINE_WIDTH, '#'))
    hopper_diff_running =  DifferentialActionModelHopper()
    hopper_diff_terminal = DifferentialActionModelHopper(isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, '-'))

    x0 = np.zeros(4) 
    MAX_ITER = 1000
    hopper_running = crocoddyl.IntegratedActionModelEuler(hopper_diff_running, dt) 
    hopper_terminal = crocoddyl.IntegratedActionModelEuler(hopper_diff_terminal, dt) 
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    problem = crocoddyl.ShootingProblem(x0, [hopper_running]*T, hopper_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    
    fddp = crocoddyl.SolverFDDP(problem)
    print(" Constructing FDDP solver completed ".center(LINE_WIDTH, '-'))
    fddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(T+1)
    us = [np.zeros(1)]*T
    converged = fddp.solve(xs,us, MAX_ITER)

    if not converged:
        print("Solver Did Not Converge !!")

    time_array = dt*np.arange(T+1)

    plt.figure("trajectory plot")
    plt.plot(time_array,np.array(fddp.xs)[:,0], label="Mass Height")
    plt.plot(time_array,np.array(fddp.xs)[:,1], label="Piston Height")
    plt.show()



    
            
