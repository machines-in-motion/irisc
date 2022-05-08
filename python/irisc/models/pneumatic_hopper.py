""" model inspired from Sideris and Bobrow 
states 
x0: hip position relative to groud 
x1: foot position relative to hip, positive is downward 
x2: hip velocity 
x3: foot velocity relative to hip 

"""


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 


class PneumaticHopper1D:
    def __init__(self):
        self.g = 9.81 
        self.mass = 2. 
        self.k = 500. 
        self.alpha = .01
        self.inv_m = 1./self.mass 
        self.d0 = .5 # nominal position of foot relative to mass 

    def nonlinear_dynamics(self, x, u):
        acc = np.zeros(2)
        e = x[1] - x[0] + self.d0
         
        if e < 0.:
            fc = 0. 
        elif e >= 0 and e < self.alpha:
            scale = .5*self.k / self.alpha
            fc = scale*(e**2)
        else:
            fc = self.k*e - .5*self.k*self.alpha
        if fc<0.:
            fc = 0.
        acc[0] = self.inv_m*fc  - self.g
        acc[1] =  u[0] 
        return acc

    def derivatives(self, x, u):
        """returns df/dx evaluated at x,u """
        dfdx = np.zeros([2,4])
        dfdu = np.zeros([2,1])
        # 

        e = x[1] - x[0] + self.d0
        dfdu[1,0] = 1.   
        if e < 0.:
            fc = 0. 
        elif e >= 0 and e < self.alpha:
            scale = .5*self.k / self.alpha
            fc = scale*(e**2)
            dfdx[0,0] = -self.inv_m*(self.k/self.alpha)*e 
            dfdx[0,1] = self.inv_m*(self.k/self.alpha)*e 
        else:
            fc = self.k*e - .5*self.k*self.alpha
            dfdx[0,0] = -self.inv_m*self.k 
            dfdx[0,1] = self.inv_m*self.k
        if fc<1.e-6:
            fc = 0.
            dfdx = np.zeros([2,4])

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
        self.dynamics = PneumaticHopper1D()
        self.isTerminal = isTerminal
        self.z_des = 2.
        # self.scale = .5/T  
        #
        self.t = t  # will be used to scale costs 
        self.T = T  # horizon
        self.t1 = int(T/2) - 15 # phase one 
        self.t2 = self.t1 + 10
        self.w = [] 

        self.wx_stance = np.array([[5.e+2, 0., 0., 0.],
                                 [0., 1.e+1, 0., 0.],
                                 [0., 0., 1.e-1, 0.],
                                 [0., 0., 0., 1.e-1]])
        self.wx_jump = np.array([[1.e+4, 0., 0., 0.],
                                 [0., 1.e-2, 0., 0.],
                                 [0., 0., 1.e+1, 0.],
                                 [0., 0., 0., 0.]])
        self.wx_land = np.array([[5.e+2, 0., 0., 0.],
                                 [0., 1.e+1, 0., 0.],
                                 [0., 0., 1.e+1, 0.],
                                 [0., 0., 0., 1.e-5]])
        self.wx_terminal = np.array([[5.e+2, 0., 0., 0.],
                                     [0., 1.e+2, 0., 0.],
                                     [0., 0., 1.e+1, 0.],
                                     [0., 0., 0., 1.e+1]])

        self.wu_stance =   np.array([[1.e-2]])
        self.wu_jump =     np.array([[1.e-2]])   
        self.wu_land =     np.array([[1.e-2]])         
        self.cost_scale = 1.e-1 #5.e-3 #1.e-2     
        

    def _running_cost(self, t, x, u): 
        if t<= self.t1:
            dx = x - np.array([self.dynamics.d0, self.dynamics.alpha, 0., 0.])
            cost = .5 * dx.T.dot(self.wx_stance).dot(dx) + .5* u.T.dot(self.wu_stance).dot(u)
        elif t>self.t2:
            dx = x - np.array([self.dynamics.d0, self.dynamics.alpha, 0., 0.])
            cost = .5 * dx.T.dot(self.wx_land).dot(dx) + .5* u.T.dot(self.wu_land).dot(u)
        else:
            dx = x - np.array([self.z_des, 0., 0., 0.])
            cost = .5 * dx.T.dot(self.wx_jump).dot(dx) + .5* u.T.dot(self.wu_jump).dot(u)
        cost *= self.cost_scale
        return cost 
    
    def _terminal_cost(self, x):
        dx = x - np.array([self.dynamics.d0, self.dynamics.alpha, 0., 0.])
        cost = .5 * dx.T.dot(self.wx_terminal).dot(dx)
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
            dx = x - np.array([self.dynamics.d0, self.dynamics.alpha, 0., 0.])
            Lx = self.wx_terminal.dot(dx)
            Lxx = self.wx_terminal
            Fx, _ = self.dynamics.derivatives(x,u)
        else:
            t = self.t 
            if t<= self.t1: #or t > self.t2:
                dx = x - np.array([self.dynamics.d0, self.dynamics.alpha, 0., 0.])
                Lx = self.wx_stance.dot(dx)
                Lxx = self.wx_stance
                Lu = self.wu_stance.dot(u)
                Luu = self.wu_stance
            elif t>self.t2:
                dx = x - np.array([self.dynamics.d0, self.dynamics.alpha, 0., 0.]) 
                Lx = self.wx_land.dot(dx)
                Lxx = self.wx_land
                Lu = self.wu_land.dot(u)
                Luu = self.wu_land
            else:
                dx = x - np.array([self.z_des, 0., 0., 0.])
                Lx = self.wx_jump.dot(dx)
                Lxx = self.wx_jump
                Lu = self.wu_jump.dot(u)
                Luu = self.wu_jump
            Fx, Fu = self.dynamics.derivatives(x,u)
    
        Lx *= self.cost_scale 
        Lxx *= self.cost_scale 
        Lu *= self.cost_scale 
        Luu *= self.cost_scale 

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



    
            
