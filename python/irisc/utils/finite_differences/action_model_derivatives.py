""" computes derivatives of the integrated action models 
for the dynamics Fx & Fu 
for the cost Lx, Lu, Lxx, Luu, Lxu 
references on finite differencing 
https://pages.mtu.edu/~msgocken/ma5630spring2003/lectures/diff/diff/node6.html
https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

"""

import numpy as np 
import crocoddyl 

DELTA = 1.e-6 # numerical differentiation step 

class CostNumDiff: 
    def __init__(self, model):
        self.model = model 
        self.state = self.model.state
        self.data = crocoddyl.IntegratedActionDataEuler(self.model)
        self.nu = self.model.nu
        self.ndx = self.model.state.ndx 

        self.Lx = np.zeros(self.ndx)
        self.Lu = np.zeros(self.nu)
        self.Lxx = np.zeros([self.ndx, self.ndx])
        self.Luu = np.zeros([self.nu, self.nu])
        self.Lxu = np.zeros([self.ndx, self.nu])

    def calcLx(self, x, u): 
        dx = np.zeros(self.ndx)
        for i in range(self.ndx):
            dx[i] = DELTA
            x_new = self.state.integrate(x, dx)
            self.model.calc(self.data, x_new, u)
            cost1 = self.data.cost 
            dx[i] = -DELTA 
            x_new = self.state.integrate(x, dx)
            self.model.calc(self.data, x_new, u)
            cost2 = self.data.cost 
            self.Lx[i] = cost1 - cost2 
            dx[i] = 0. 
        self.Lx *= 2./DELTA 

    def calcLu(self, x, u): 
        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = DELTA
            u_new = u + du 
            self.model.calc(self.data, x, u_new)
            cost1 = self.data.cost 
            du[i] = -DELTA
            u_new = u + du 
            self.model.calc(self.data, x, u_new)
            cost2 = self.data.cost 
            self.Lu[i] = cost1 - cost2 
            du[i] = 0. 
        self.Lu *= 2./DELTA

    def calcLxx(self, x, u): 
        dx1 = np.zeros(self.ndx)
        dx2 = np.zeros(self.ndx)

    def calcLuu(self, x, u): 
        du1 = np.zeros(self.nu)
        du2 = np.zeros(self.nu)

    def calcLxu(self, x, u): 
        dx = np.zeros(self.ndx) 
        du = np.zeros(self.nu)



class DynamicsNumDiff: 
    def __init__(self, model):
        self.model = model 
        self.data = crocoddyl.IntegratedActionDataEuler(self.model)
        self.state = self.model.state
        self.nu = self.model.nu
        self.ndx = self.model.state.ndx 
        self.Fx = np.zeros([self.ndx, self.ndx])
        self.Fu = np.zeros([self.ndx, self.nu])

    def calcFx(self, x, u): 
        dx = np.zeros(self.ndx)
        for i in range(self.ndx):
            dx[i] = DELTA
            x_new = self.state.integrate(x, dx)
            self.model.calc(self.data, x_new, u)
            x_next1 = self.data.xnext.copy() 
            dx[i] = -DELTA 
            x_new = self.state.integrate(x, dx)
            self.model.calc(self.data, x_new, u)
            x_next2 = self.data.xnext.copy()
            self.Fx[:,i] = self.state.diff(x_next2, x_next1)  
            dx[i] = 0. 
        self.Fx *= 2./DELTA 

    def calcFu(self, x, u): 
        du = np.zeros(self.nu)
        for i in range(self.nu):
            du[i] = DELTA
            u_new = u + du 
            self.model.calc(self.data, x, u_new)
            x_next1 = self.data.xnext.copy() 
            du[i] = -DELTA 
            u_new = u + du 
            self.model.calc(self.data, x, u_new)
            x_next2 = self.data.xnext.copy()
            self.Fu[:,i] = self.state.diff(x_next2, x_next1)  
            du[i] = 0. 
        self.Fu *= 2./DELTA 



    