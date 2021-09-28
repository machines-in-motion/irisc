""" model inspired from Sideris and Bobrow """


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 
LINE_WIDTH = 100 

we = .05#001 # exponential time weight 

class DifferentialActionModelHopper(crocoddyl.DifferentialActionModelAbstract):
    def __init__(self, t, T, isTerminal=False):
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
        self.alpha = 1.e-2
        self.z_des = 1.5 
        self.scale = 1./(2.*T) # (dt*N)/(2*N) = dt/2 
        self.t = t  # will be used to scale costs 
        self.T = T  # horizon
        self.t1 = int(2*T/3) - 4 # phase one 
        self.t2 = self.t1 + 8 
        self.w = [] 
        # phase one cost weights 
        self.w += [1.e+1] # decreasing wight on same piston and mass  w[0]
        self.w += [.01] # piston velocity  w[1]
        self.w += [.1] # control weight  w[2]
        # phase two  
        self.w += [1.e+3] # increasing on desired mass height w[3] 
        self.w += [1.e+1] # increasing on piston relative  w[4]
        self.w += [1.e+3] # mass velocity  w[5]
        self.w += [.01] # piston velocity  w[6]
        # phase 2 cost weights 
        self.w += [1.e+1] # increeasing weight both mass and piston positions w[7] 
        self.w += [1.e+1] # velocity weights w[8]
        self.w += [.1] # control weight  w[9]

    # def _phase_one_cost(self, t, x, u):
    #     cost = self.w[0]*np.exp(-we*t)*(x[1]-x[0])**2 + self.w[1]*x[3]**2 + self.w[2]*u[0]**2 
    #     cost += self.w[3]*np.exp(we*t)*(x[0]-self.z_des)**2 + self.w[4]*np.exp(we*t)*(x[1]-1.)**2 
    #     cost += self.w[5]*x[2]**2 + self.w[6]*x[3]**2  
    #     return cost 

    def _running_cost(self, t, x, u): 
        if t < self.t1: 
            we = 0.
            # cost = self.w[0]*np.exp(-we*t)*(x[1]-x[0])**2 + self.w[1]*x[3]**2 + self.w[2]*u[0]**2 
            cost = self.w[0]*np.exp(-we*t)*(x[1]-1.)**2 #+ self.w[1]*x[3]**2 + self.w[2]*u[0]**2 
            cost += self.w[0]*np.exp(-we*t)*(x[0]-1.)**2 + self.w[2]*u[0]**2 
        elif t>=self.t1 and t < self.t2:
            we = 0.
            cost = self.w[3]*np.exp(we*t)*(x[0]-self.z_des)**2 + self.w[4]*np.exp(we*t)*(x[1]-1.)**2 
            # cost += self.w[5]*x[2]**2 + self.w[6]*x[3]**2  + self.w[2]*u[0]**2  
            cost += self.w[6]*x[2]**2  + self.w[2]*u[0]**2  
        else: 
            we = .0
            cost = self.w[7]*np.exp(we*t)*(x[0]-1.)**2 + self.w[7]*np.exp(we*t)*(x[1]-1.)**2
            cost += self.w[8]*x[2]**2 + self.w[8]*x[3]**2 + self.w[9]*u[0]**2  

        return cost 

    # def _phase_two_cost(self, t, x, u):
    #     cost = self.w[7]*np.exp(we*t)*(x[0]-1.)**2 + self.w[7]*np.exp(we*t)*(x[1]-1.)**2
    #     cost += self.w[8]*x[2]**2 + self.w[8]*x[3]**2 + self.w[9]*u[0]**2 
    #     return cost 


    # def _running_cost(self, x, u):
    #     cost = self.scale*1000.*x[1]**2 + self.scale*100.*x[3]**2 + self.scale*1.e-2*u[0]**2 
    #     return cost

    def _terminal_cost(self, x): 
        # cost = 500.*(x[0]-self.y_des)**2 + x[2]**2  
        we = 0.0 
        cost = self.w[7]*np.exp(we*self.T)*(x[0]-1.)**2 + self.w[7]*np.exp(we*self.T)*(x[1]-1.)**2
        cost += self.w[8]*x[2]**2 + self.w[8]*x[3]**2
        return cost 

    def calc(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        #
        if self.isTerminal: 
            # print("calling terminal calc")
            data.cost = self._terminal_cost(x) 
            data.xout = np.zeros(2)
        else:
            self.cost = self._running_cost(self.t, x, u)
            # if self.t < self.t1:
            #     # print("calling phase 1 calc")
            #     data.cost = self._phase_one_cost(self.t, x, u)
            # elif self.t >= self.t1:
            #     # print("calling phase 2 calc")
            #     data.cost = self._phase_two_cost(self.t, x, u) 
            
            e = x[1] - x[0]
            data.xout[1] = u[0] 
            if e < 0.:
                data.xout[0] = self.g
            elif e >= 0 and e < self.alpha:
                scale = .5*self.k / self.alpha
                data.xout[0] = scale*(e**2)  + self.g
            else:
                data.xout[0] = self.k*e - .5*self.k*self.alpha  + self.g


        # else:
        #     data.cost = self._running_cost(x,u)
        #     # 
        #     e = x[1] - x[0]
        #     data.xout[1] = u[0] 
        #     if e < 0.:
        #         data.xout[0] = self.g
        #     elif e >= 0 and e < self.alpha:
        #         scale = self.k/(2.*self.alpha) 
        #         data.xout[0] = scale*(e**2)  + self.g
        #     else:
        #         data.xout[0] = self.k*e - .5*self.k*self.alpha  + self.g

    def calcDiff(self, data, x, u=None): 
        if u is None:
            u = np.zeros(self.nu)
        #
        Fx = np.zeros([2,4]) 
        Fu = np.zeros(2)

        Fu[1] = 1.

        
        Lx = np.zeros(4)
        Lu = np.zeros(1)
        Lxx = np.zeros([4,4])
        Luu = np.zeros([1,1])
        Lxu = np.zeros([4,1])
        # 

        # COST DERIVATIVES 
        if self.isTerminal:
            we = .00
            Lx[0] = 2. * self.w[7]*np.exp(we*self.T)*(x[0]-1.)
            Lx[1] = 2. * self.w[7]*np.exp(we*self.T)*(x[1]-1.) 
            Lx[2] = 2.*self.w[8]*x[2]
            Lx[3] = 2.*self.w[8]*x[3]

            Lxx[0, 0] = 2. * self.w[7]*np.exp(we*self.T)
            Lxx[1, 1] = 2. * self.w[7]*np.exp(we*self.T)
            Lxx[2, 2] = 2.*self.w[8]
            Lxx[3, 3] = 2.*self.w[8] 
        else:
            t = self.t 
            if self.t < self.t1:
                we = 0. 
                Lu[0] = 2.*self.w[2]*u[0] 
                Luu[0,0] = 2.*self.w[2] 
                # 
                # Lx[0] = -2.*self.w[0]*np.exp(-we*self.t)*(x[1]-x[0])
                # Lx[1] = 2.*self.w[0]*np.exp(-we*self.t)*(x[1]-x[0])
                Lx[0] = 2.*self.w[0]*np.exp(-we*self.t)*(x[0]-1.)
                Lx[1] = 2.*self.w[0]*np.exp(-we*self.t)*(x[1]-1.)
                # Lx[2] = 2.*self.w[5]*x[2]
                # Lx[3] = 2.*self.w[1]*x[3] #  + 2*self.w[6]*x[3]
                # 
                Lxx[0,0] = 2.*self.w[0]*np.exp(-we*self.t) #+ 2.*self.w[3]*np.exp(we*self.t)
                Lxx[1,1] = 2.*self.w[0]*np.exp(-we*self.t) #+ 2.*self.w[4]*np.exp(we*self.t)
                # Lxx[2,2] = 2*self.w[5] 
                # Lxx[3,3] = 2.*self.w[1] # + 2. * self.w[6]
                Lxx[0,1] = 0. #-2.*self.w[0]*np.exp(-we*self.t)
                Lxx[1,0] = 0. # -2.*self.w[0]*np.exp(-we*self.t)
                # print("called phase 1 calcDiff")
            elif self.t >= self.t1 and self.t < self.t2: 
                we = 0. 
                Lu[0] = 2.*self.w[2]*u[0] 
                Luu[0,0] = 2.*self.w[2] 

                Lx[0] = 2.*self.w[3]*np.exp(we*self.t)*(x[0]-self.z_des) 
                Lx[1] = 2.*self.w[4]*np.exp(we*self.t)*(x[1]- 1.)
                Lx[2] = 2.*self.w[5]*x[2]
                # Lx[3] = 2*self.w[6]*x[3]
                # 
                Lxx[0,0] =  2.*self.w[3]*np.exp(we*self.t)
                Lxx[1,1] =  2.*self.w[4]*np.exp(we*self.t)
                Lxx[2,2] = 2*self.w[5] 
                # Lxx[3,3] = 2. * self.w[6]
                # Lxx[0,1] = -2.*self.w[0]*np.exp(-we*self.t)
                # Lxx[1,0] = -2.*self.w[0]*np.exp(-we*self.t)
                

            else:
                we = .0
                Lu[0] = 2.*self.w[9]*u[0] 
                Luu[0,0] =  2.*self.w[9]  
                Lx[0] = 2. * self.w[7]*np.exp(we*t)*(x[0]-1.)
                Lx[1] = 2. * self.w[7]*np.exp(we*t)*(x[1]-1.) 
                Lx[2] = 2.*self.w[8]*x[2]
                Lx[3] = 2.*self.w[8]*x[3]

                Lxx[0, 0] = 2. * self.w[7]*np.exp(we*t)
                Lxx[1, 1] = 2. * self.w[7]*np.exp(we*t)
                Lxx[2, 2] = 2.*self.w[8]
                Lxx[3, 3] = 2.*self.w[8]

            # DYNAMICS DERIVATIVES  
            e = x[1] - x[0]
            if e<0.:
                pass                  
            elif e >= 0 and e < self.alpha:
                Fx[0,0] = -(self.k/self.alpha)*e 
                Fx[0,1] = (self.k/self.alpha)*e 
            else:
                Fx[0,0] = -self.k 
                Fx[0,1] = self.k

        # else:
        #     t = self.T 
        #     if not self.isTerminal:
        #         Lu[0] = 2.*self.w[9]*u[0] 
        #         Luu[0,0] =  2.*self.w[9] 
        #         t = self.t 
            
        #     Lx[0] = 2. * self.w[7]*np.exp(we*t)*(x[0]-1.)
        #     Lx[1] = 2. * self.w[7]*np.exp(we*t)*(x[1]-1.) 
        #     Lx[2] = 2.*self.w[8]*x[2]
        #     Lx[3] = 2.*self.w[8]*x[3]

        #     Lxx[0, 0] = 2. * self.w[7]*np.exp(we*t)
        #     Lxx[1, 1] = 2. * self.w[7]*np.exp(we*t)
        #     Lxx[2, 2] = 2.*self.w[8]
        #     Lxx[3, 3] = 2.*self.w[8]

            # if self.isTerminal:
                # print("called terminal calcDiff")
            # else:
                # print("called phase 2 calcDiff")


        

        data.Fx = Fx.copy()
        data.Fu = Fu.copy()
        data.Lx = Lx.copy()
        data.Lu = Lu.copy()
        data.Lxx = Lxx.copy()
        data.Luu = Luu.copy()
        data.Lxu = Lxu.copy()

        # if self.isTerminal: 
        #     Lx[0] = 1000.*(x[0]-self.y_des)
        #     Lx[1] = 0.
        #     Lx[2] = 2.*x[2]
        #     Lx[3] = 0.
        #     Lxx[0,0] = 1000. 
        #     Lxx[1,1] = 0.
        #     Lxx[2,2] = 2.
        #     Lxx[3,3] = 0.
        # else:
        #     Lx[1] = 2.*self.scale*1000.*x[1] 
        #     Lx[3] = 2.*self.scale*100.*x[3] 
        #     Lu[0] = 2.*self.scale*1.e-2*u[0]
        #     Lxx[1,1] = 2.*self.scale*1000. 
        #     Lxx[3,3] = 2.*self.scale*100. 
        #     Luu[0,0] = 2.*self.scale*1.e-2
        #     Fu[0,0] = 0. 
        #     Fu[1,0] = 1. 
        #     e = x[1] - x[0]
        #     if e<0.:
        #         pass                  
        #     elif e >= 0 and e < self.alpha:
        #         Fx[0,0] = (self.k/self.alpha)*(x[0]-x[1])
        #         Fx[0,1] = (self.k/self.alpha)*(x[1]-x[0])
        #     else:
        #         Fx[0,0] = -self.k*x[0] - .5*self.alpha*self.k 
        #         Fx[0,1] = self.k*x[1] - .5*self.alpha*self.k 



if __name__ =="__main__":
    # print(" Testing Penumatic Hopper with DDP ".center(LINE_WIDTH, '#'))
    # hopper_diff_running =  DifferentialActionModelHopper()
    # hopper_diff_terminal = DifferentialActionModelHopper(isTerminal=True)
    # print(" Constructing differential models completed ".center(LINE_WIDTH, '-'))

    x0 = np.array([1., 1., 0., 0.])
    MAX_ITER = 1000
    T = 300 
    dt = 1.e-2
    running_models = []
    for t in range(T): 
        diff_hopper = DifferentialActionModelHopper(t, T, False) 
        running_models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
    diff_hopper = DifferentialActionModelHopper(T, T, True) 
    terminal_model = crocoddyl.IntegratedActionModelEuler(diff_hopper, dt) 
    # hopper_running = crocoddyl.IntegratedActionModelEuler(hopper_diff_running, dt) 
    # hopper_terminal = crocoddyl.IntegratedActionModelEuler(hopper_diff_terminal, dt) 
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    
    fddp = crocoddyl.SolverFDDP(problem)
    print(" Constructing FDDP solver completed ".center(LINE_WIDTH, '-'))
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

    # # t = np.arange(0, 1., 1.e-3)
    # # e = [1.e3*np.exp(ti*10.) for ti in t]
    # # plt.figure("some fig")
    # # plt.plot(t,e)
    plt.show()



    
            

# class DifferentialActionModelHopper(crocoddyl.DifferentialActionModelAbstract):
#     def __init__(self, t, T, isTerminal=False):
#         nq = 2
#         nv = 2 
#         nx = nv + nq 
#         ndx = nx 
#         nu = 1 
#         state =  crocoddyl.StateVector(nx)
#         crocoddyl.DifferentialActionModelAbstract.__init__(self, state, nu, ndx)
#         self.g = - 9.81 
#         self.isTerminal = isTerminal
#         self.mass = 1. 
#         self.k = 1.e+3
#         self.alpha = .01
#         self.z_des = 10. 
#         #
#         self.t = t  # will be used to scale costs 
#         self.T = T  # horizon
#         self.t1 = int(T/2) - 5 # phase one 
#         self.t2 = self.t1 + 5
#         self.w = [] 
#         # phase one cost weights 
#         self.w += [1.e+1] # piston position  w[0]
#         self.w += [1.e-2] # control  w[1]
#         #phase two
#         self.w += [1.e+5] # mass height  w[2]
#         self.w += [1.e+2] # mass velocity w[3] 
#         self.w += [1.e-2] # control  w[4]
#         # phase 3 
#         self.w += [1.e+1] # piston position  w[5]
#         self.w += [1.e-2] # control  w[6]
#         # terminal cost weights 
#         self.w += [1.e+1] # increeasing weight both mass and piston positions w[7] 


#     def _running_cost(self, t, x, u): 
#         if t < self.t1:
#             cost = self.w[0]*x[1]**2 + self.w[1]*u[0]**2  
#         elif t>=self.t1 and t<self.t2:
#             cost = self.w[2]*(x[0]- self.z_des)**2 + self.w[3]*x[2]**2 + self.w[4]*u[0]**2 
#         else:
#             cost = self.w[5]*x[1]**2 + self.w[6]*u[0]**2  
#         return cost 
    
#     def _terminal_cost(self, x):
#         cost =  self.w[7]*x[1]**2  
#         return cost 

#     def calc(self, data, x, u=None): 
#         if u is None:
#             u = np.zeros(self.nu)
#         #
#         if self.isTerminal: 
#             data.cost = self._terminal_cost(x) 
#             data.xout = np.zeros(2)
#         else:
#             self.cost = self._running_cost(self.t, x, u)
#             e = x[1] - x[0]
#             data.xout[1] = u[0] 
#             if e < 0.:
#                 data.xout[0] = self.g
#             elif e >= 0 and e < self.alpha:
#                 scale = .5*self.k / self.alpha
#                 data.xout[0] = scale*(e**2)  + self.g
#             else:
#                 data.xout[0] = self.k*e - .5*self.k*self.alpha  + self.g

#     def calcDiff(self, data, x, u=None): 
#         if u is None:
#             u = np.zeros(self.nu)
#         #
#         Fx = np.zeros([2,4]) 
#         Fu = np.zeros([2,1])
#         Lx = np.zeros([4])
#         Lu = np.zeros([1])
#         Lxx = np.zeros([4,4])
#         Luu = np.zeros([1,1])
#         Lxu = np.zeros([4,1])
#         # COST DERIVATIVES 
#         if self.isTerminal:
#             Lx[1] = 2.*self.w[7]*x[1]
#             Lxx[1,1] = 2.*self.w[7]  
#         else:
#             Fu[1,0] = 1.  
#             t = self.t 
#             if t < self.t1:
#                 Lu[0] = 2.*self.w[1]*u[0]
#                 Luu[0,0] = 2.*self.w[1]
#                 #
#                 Lx[1] = 2.*self.w[0]*x[1]
#                 Lxx[1,1] = 2.*self.w[0] 
                
#             elif t >= self.t1 and self.t < self.t2: 
#                 Lu[0] = 2.*self.w[4]*u[0]
#                 Luu[0,0] = 2.*self.w[4] 
#                 # 
#                 Lx[0]= 2.*self.w[2]*(x[0]- self.z_des)
#                 Lxx[0,0]= 2.*self.w[2]
#                 Lx[2]= 2.*self.w[3]*x[2]
#                 Lxx[2,2]= 2.*self.w[3]
#             else:
#                 Lu[0] = 2.*self.w[6]*u[0]
#                 Luu[0,0] = 2.*self.w[6] 
#                 # 
#                 Lx[1] = 2.*self.w[5]*x[1]
#                 Lxx[1,1] = 2.*self.w[5] 
                


#             # DYNAMICS DERIVATIVES  
#             e = x[1] - x[0]
#             if e<0.:
#                 pass                  
#             elif e >= 0. and e < self.alpha:
#                 Fx[0,0] = -(self.k/self.alpha)*e 
#                 Fx[0,1] = (self.k/self.alpha)*e 
#             else:
#                 Fx[0,0] = -self.k 
#                 Fx[0,1] = self.k
#         # COPY TO DATA 
#         data.Fx = Fx.copy()
#         data.Fu = Fu.copy()
#         data.Lx = Lx.copy()
#         data.Lu = Lu.copy()
#         data.Lxx = Lxx.copy()
#         data.Luu = Luu.copy()
#         data.Lxu = Lxu.copy()




# if __name__ =="__main__":
#     # print(" Testing Penumatic Hopper with DDP ".center(LINE_WIDTH, '#'))
#     # hopper_diff_running =  DifferentialActionModelHopper()
#     # hopper_diff_terminal = DifferentialActionModelHopper(isTerminal=True)
#     # print(" Constructing differential models completed ".center(LINE_WIDTH, '-'))

#     x0 = np.array([0.1, 0.11, 0., 0.])
#     MAX_ITER = 1000
#     T = 300 
#     dt = 1.e-2
#     running_models = []
#     for t in range(T): 
#         diff_hopper = DifferentialActionModelHopper(t, T, False) 
#         running_models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
#     diff_hopper = DifferentialActionModelHopper(T, T, True) 
#     terminal_model = crocoddyl.IntegratedActionModelEuler(diff_hopper, dt) 
#     # hopper_running = crocoddyl.IntegratedActionModelEuler(hopper_diff_running, dt) 
#     # hopper_terminal = crocoddyl.IntegratedActionModelEuler(hopper_diff_terminal, dt) 
#     print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

#     problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)
#     print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    
#     fddp = crocoddyl.SolverFDDP(problem)
#     print(" Constructing FDDP solver completed ".center(LINE_WIDTH, '-'))
#     fddp.setCallbacks([
#     crocoddyl.CallbackLogger(),
#     crocoddyl.CallbackVerbose()
#     ])
#     xs = [x0]*(T+1)
#     us = [np.array([1.e-3])]*T

#     converged = fddp.solve(xs,us, MAX_ITER)

#     if not converged:
#         print("Solver Did Not Converge !!")

#     time_array = dt*np.arange(T+1)

#     plt.figure("trajectory plot")
#     plt.plot(time_array,np.array(fddp.xs)[:,0], label="Mass Height")
#     plt.plot(time_array,np.array(fddp.xs)[:,1], label="Piston Height")

#     # # t = np.arange(0, 1., 1.e-3)
#     # # e = [1.e3*np.exp(ti*10.) for ti in t]
#     # # plt.figure("some fig")
#     # # plt.plot(t,e)
#     plt.show()



    
            
