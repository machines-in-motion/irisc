""" computes derivatives of the integrated action models 
for the dynamics Fx & Fu 
for the cost Lx, Lu, Lxx, Luu, Lxu 
references on finite differencing 
https://pages.mtu.edu/~msgocken/ma5630spring2003/lectures/diff/diff/node6.html
https://v8doc.sas.com/sashtml/ormp/chap5/sect28.htm

"""

import numpy as np 

DELTA = 1.e-6 # numerical differentiation step 

class CostNumDiff: 
    def __init__(self, model):
        self.model = model 
        self.state = self.model.state
        self.data = None # create new data for this particular model 
        self.nu = self.model.nu
        self.ndx = self.model.state.ndx 

        self.Lx = np.zeros(self.ndx)
        self.Lu = np.zeros(self.nu)
        self.Lxx = np.zeros([self.ndx, self.ndx])
        self.Luu = np.zeros([self.nu, self.nu])
        self.Lxu = np.zeros([self.ndx, self.nu])

    def calcLx(self, x, u): 
        dx = np.zeros(self.ndx)

    def calcLu(self, x, u): 
        du = np.zeros(self.nu)

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
        self.state = self.model.state
        self.nu = self.model.nu
        self.ndx = self.model.state.ndx 
        self.Fx = np.zeros([self.ndx, self.ndx])
        self.Fu = np.zeros([self.ndx, self.nu])

    def calcFx(self, x, u): 
        dx = np.zeros(self.ndx)

    def calcFu(self, x, u): 
        du = np.zeros(self.nu)



    