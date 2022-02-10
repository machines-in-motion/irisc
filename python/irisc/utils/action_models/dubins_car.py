import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 


class DifferentialActionModelDubins(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, t, T, isTerminal=False):
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
        self.w1 = np.eye(6)
        self.w1[0,0] = 200.
        self.w1[1,1] = 200. 

        self.w1[3,3] = 50.
        self.w1[4,4] = 50. 
        self.w1[5,5] = 200. 
        self.w2 = np.eye(2)
        self.w3 = 10000.*np.eye(6)
        self.w3[2,2] = 1.e+8 

    def _running_cost(self, t, x, u): 
        return .5*x.dot(self.w1).dot(x)  +.5*u.dot(self.w2).dot(u)
    
    def _terminal_cost(self, x):
        return .5*x.dot(self.w3).dot(x) 

    def calc(self, data, x, u):
        if u is None:
            u = np.zeros(self.nu) 

        if self.isTerminal: 
            data.cost = self._terminal_cost(x) 
            data.xout = np.zeros(self.state.nv)
        else:
            data.cost = self._running_cost(self.t, x, u)
            data.xout[0] = u[0]*np.cos(x[2])
            data.xout[1] = u[0]*np.sin(x[2])
            data.xout[2] = u[1]



    def calcDiff(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu) 
        
        Fx = np.zeros([self.state.nv,self.state.ndx]) 
        Fu = np.zeros([self.state.nv,self.nu])
        Lx = np.zeros([self.state.nx])
        Lu = np.zeros([self.nu])
        Lxx = np.zeros([self.state.ndx,self.state.ndx])
        Luu = np.zeros([self.nu,self.nu])
        Lxu = np.zeros([self.state.ndx,self.nu])

        if self.isTerminal:
            Lx[:] = self.w3.dot(x) 
            Lxx[:,:] = self.w3.copy() 
        else:
            # dynamics derivatives 
            Fx[0,2] = -u[0]*np.sin(x[2])
            Fx[1,2] = u[0]*np.cos(x[2])
            Fu[0,0] = np.cos(x[2])
            Fu[1,0] = np.sin(x[2])
            Fu[2,1] = 1.  

            # cost derivatives 
            Lx[:] = self.w1.dot(x) 
            Lu[:] = self.w2.dot(u) 
            Lxx[:,:] = self.w1.copy()
            Luu[:,:] = self.w2.copy()

        # COPY TO DATA 
        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()

if __name__ == "__main__":
    print(" Testing Dubins Car with DDP ".center(LINE_WIDTH, '#'))
    x0 = np.zeros(6)
    x0[0], x0[1] = 5., 5. 
    MAX_ITER = 1000
    T = 300 
    dt = 1.e-2
    running_models = []
    for t in range(T): 
        diff_hopper = DifferentialActionModelDubins(t, T, False) 
        running_models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
    diff_hopper = DifferentialActionModelDubins(T, T, True) 
    terminal_model = crocoddyl.IntegratedActionModelEuler(diff_hopper, dt) 
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
    plt.plot(np.array(fddp.xs)[:,0],np.array(fddp.xs)[:,1], label="Mass Height")
    # plt.plot(time_array,np.array(fddp.xs)[:,1], label="Piston Height")
    # plt.plot(time_array, 0.*deltas, '--k', linewidth=2., label="ground")

    
    
    plt.figure("orientation plot")
    plt.plot(time_array,np.array(fddp.xs)[:,2], label="Mass Height")
    # plt.figure("Penetration")
    # plt.plot(time_array, .5*deltas)


    # plt.figure("control inputs")
    # plt.plot(time_array[:-1],np.array(fddp.us)[:,0], label="control inputs")


    # plt.figure("feedback gains")
    # for i in range(4):
    #     plt.plot(time_array[:-1],np.array(fddp.K)[:,i], label="$K_%s$"%i)
    # plt.legend()

    plt.show()



    
            