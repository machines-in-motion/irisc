import numpy as np 
import scipy.linalg as scl

from crocoddyl import SolverAbstract

LINE_WIDTH = 100 

VERBOSE = False   
def rev_enumerate(l):
    return reversed(list(enumerate(l)))


def raiseIfNan(A, error=None):
    if error is None:
        error = scl.LinAlgError("NaN in array")
    if np.any(np.isnan(A)) or np.any(np.isinf(A)) or np.any(abs(np.asarray(A)) > 1e30):
        raise error




class RiskSensitiveSolver(SolverAbstract):
    def __init__(self, shootingProblem, problemUncertainty, sensitivity, withMeasurement=False):
        SolverAbstract.__init__(self, shootingProblem)
        self.sigma = sensitivity
        self.uncertainty = problemUncertainty 
        # 
        self.wasFeasible = False # Change it to true if you know that datas[t].xnext = xs[t+1]
        self.alphas = [2**(-n) for n in range(10)]
        self.x_reg = 0
        self.u_reg = 0
        self.regFactor = 10
        self.regMax = 1e9
        self.regMin = 1e-9
        self.th_step = .5
        self.th_stop = 1.e-9 
        self.n_little_improvement = 0 
        self.gap_tolerance = 1.e-6
        # 
        self.withMeasurement = withMeasurement 
        self.withGaps = False 
        # 
        self.allocateData()

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod 

    def calc(self):
        raise NotImplementedError("Calc Method Not Implemented yet")
    
    def computeGaps(self):
        raise NotImplementedError("computeGaps Method Not Implemented yet")
    
    def filterPass(self): 
        raise NotImplementedError(" Risk Sensitive Filter Not Implemented Yet")

    def computeDirection(self, recalc=True):
        raise NotImplementedError("computeDirection Method Not Implemented yet")

    def tryStep(self, stepLength):
        raise NotImplementedError("tryStep Method Not Implemented yet")

    def expectedImprovement(self):
        raise NotImplementedError("expectedImprovement Method Not Implemented yet")
        

    def stoppingCriteria(self):
        raise NotImplementedError("stoppingCriteria Method Not Implemented yet")
        
    def solve(self, maxiter=100, init_xs=None, init_us=None, isFeasible=False, regInit=None):
        raise NotImplementedError("solve Method Not Implemented yet")

    def forwardPass(self, stepLength, warning='error'):
        raise NotImplementedError("forwardPass Method Not Implemented yet")

    def recouplingControls(self): 
        raise NotImplementedError("controls recoupling not implemented yet")

    def backwardPass(self):
        raise NotImplementedError("Backward Pass for iRiSC not implemented yet")

    def increaseRegularization(self):
        self.x_reg *= self.regFactor
        if self.x_reg > self.regMax:
            self.x_reg = self.regMax
        self.u_reg = self.x_reg

    def decreaseRegularization(self):
        self.x_reg /= self.regFactor
        if self.x_reg < self.regMin:
            self.x_reg = self.regMin
        self.u_reg = self.x_reg
   
    def allocateData(self):
        """  Allocate memory for all variables needed, control, state, value function and estimator.
        """
        # state and control 
        self.xs_try = [self.problem.x0] + [np.nan] * self.problem.T
        self.us_try = [np.nan] * self.problem.T

        # dynamics and measurement approximations 
        self.A = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.problem.runningModels] 
        self.B = [np.zeros([p.state.ndx, p.nu]) for p in self.problem.runningModels]
        self.Omega = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.problem.runningModels] 
        self.H = [np.zeros([m.ny , p.state.ndx]) for p,m in zip(self.problem.runningModels, self.uncertainty.runningModels)] 
        self.Gamma = [np.zeros([m.ny, m.ny]) for m in self.uncertainty.runningModels] 
        
        # cost approximations  
        self.Q = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.models()]  
        self.q = [np.zeros(p.state.ndx) for p in self.models()]  
        self.S = [np.zeros([p.nu, p.state.ndx]) for p in self.problem.runningModels]  
        self.R = [np.zeros([p.nu, p.nu]) for p in self.problem.runningModels]  
        self.r = [np.zeros(p.nu) for p in self.problem.runningModels]  

        # backward pass variables 
        self.M = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.models()]   
        self.kff = [np.zeros([p.nu, p.state.ndx]) for p in self.problem.runningModels] 
        self.Kfb = [np.zeros([p.nu]) for p in self.problem.runningModels]
        self.V = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.models()]   
        self.v = [np.zeros(p.state.ndx) for p in self.models()]   
        
        # forward estimation 
        self.xhat = [self.problem.x0] + [np.nan] * self.problem.T 
        self.G = [np.zeros([p.state.ndx, m.ny]) for p,m in zip(self.problem.runningModels, self.uncertainty.runningModels)]  
        self.P = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.models()]   

        # recoupling 
        self.K = [np.zeros([p.nu, p.state.ndx]) for p in self.problem.runningModels]
        self.k = [np.zeros([p.nu]) for p in self.problem.runningModels]

        # gaps 
        self.fs = [np.zeros(self.problem.runningModels[0].state.ndx)
                     ] + [np.zeros(p.state.ndx) for p in self.problem.runningModels]




