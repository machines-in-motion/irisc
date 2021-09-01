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
        # compute cost and derivatives at deterministic nonlinear trajectory 
        self.problem.calc(self.xs, self.us)
        self.cost = self.problem.calcDiff(self.xs, self.us)

        self.uncertainty.calc(self.xs, self.us) # compute H, Omega, Gamma 
        self.filterPass() # compute  \hat{x} , P, G 
    
    def computeGaps(self):
        raise NotImplementedError("computeGaps Method Not Implemented yet")
    
    def filterPass(self): 
        """ involved computing \hat{x} with zero innovation """
        for t, (pdata, mdata) in enumerate(zip(self.problem.runningDatas, 
                                            self.uncertainty.runningDatas)):
            # inv_Pt + H^T invGamma H + \sigma Q 
            inv_skewed_covariance = np.linalg.inv(self.P[t]) + mdata.H.T.dot(mdata.invGamma).dot(mdata.H) + self.sigma*pdata.Lxx
            Lb = scl.cho_factor(inv_skewed_covariance , lower=True)
            # compute filter gain 
            rightG = mdata.H.T.dot(mdata.invGamma)
            gain = scl.cho_solve(Lb, rightG)
            self.G[t][:,:] = pdata.Fx.dot(gain)
            # compute covariance update 
            right_cov = pdata.Fx.T
            next_skewed_cov = scl.cho_solve(Lb, right_cov)
            self.P[t+1][:,:] = mdata.Omega + pdata.Fx.dot(next_skewed_cov)
            # update estimate 
            right_skewed_estimate = pdata.Lxx.dot(self.xhat[t]) - pdata.Lxu.dot(self.us[t]) - pdata.Lx 
            skewed_estimate = scl.cho_solve(Lb, right_skewed_estimate)
            self.xhat[t+1][:] = pdata.Fx.dot(self.xhat[t]) + pdata.Fu.dot(self.us[t]) + self.fs[t+1] -self.sigma* pdata.Fx.dot(skewed_estimate)

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from cmpute direction")
            self.calc()
        
        self.backwardPass() 
        

    def tryStep(self, stepLength):
        raise NotImplementedError("tryStep Method Not Implemented yet")

    def expectedImprovement(self):
        raise NotImplementedError("expectedImprovement Method Not Implemented yet")
        

    def stoppingCriteria(self):
        """ it will be feedforward norm along the trajectory for now """
        knormSquared = [ki.dot(ki) for ki in self.k]
        knorm = np.sqrt(np.array(knormSquared))
        return knorm
        
    def solve(self, maxiter=100, init_xs=None, init_us=None, isFeasible=True, regInit=None):
        
        self.setCandidate(init_xs, init_us, isFeasible) 
        if VERBOSE: print("solve setCandidate works just fine ")
        self.n_little_improvement = 0

        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin
        for i in range(maxiter):
            recalc = True   # this will recalculated derivatives in Compute Direction 
            while True:     # backward pass with regularization 
                try: 
                    self.computeDirection(recalc=recalc)
                except:
                    recalc = False 
                    self.increaseRegularization()
                    if self.x_reg == self.regMax: # if max reg reached, faild to converge, end solve attempt  
                        return self.xs, self.us, False
                    else:  # continue to next while  
                        continue
                break # compute direction succeeded, exit and proceed to line search 
            
            self.recouplingControls() # recoupling pass 

            for a in self.alphas:
                try: 
                    self.dV = self.tryStep(a)
                except:
                    # repeat starting from a smaller alpha 
                    print("Try Step Faild for alpha = %s"%a) 
                    continue 

                if self.dV > 0.: # Cost has decreased, accept step 
                    self.setCandidate(self.xs_try, self.us_try, True)
                    self.cost = self.cost_try 
                    break # stop line search and proceed to next iteration 
            

            if a > self.th_step: # decrease regularization if alpha > .5 
                self.decreaseRegularization()
            if a == self.alphas[-1] :  
                self.increaseRegularization()
                if self.x_reg == self.regMax:
                    return self.xs, self.us, False
            
            self.stepLength = a
            self.iter = i
            self.stop = sum(self.stoppingCriteria())
            
            if self.callback is not None: # this way callback prints appear before solver convergence message 
                [c(self) for c in self.callback]

            if  self.stop < self.th_stop:
                self.n_little_improvement += 1

            if self.n_little_improvement == 10:
                print('Solver converged with little improvement in the last 10 iterations')
                return self.xs, self.us, True

            # Warning: no convergence in max iterations
            print('max iterations with no convergance')
            return self.xs, self.us, self.isFeasible 


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
        self.xhat = [self.uncertainty.x0] + [np.zeros(p.state.nx) for p in self.problem.runningModels]
        self.G = [np.zeros([p.state.ndx, m.ny]) for p,m in zip(self.problem.runningModels, self.uncertainty.runningModels)]  
        self.P = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.models()]   

        # recoupling 
        self.xcheck = [self.uncertainty.x0] + [np.zeros(p.state.nx) for p in self.problem.runningModels]
        self.K = [np.zeros([p.nu, p.state.ndx]) for p in self.problem.runningModels]
        self.k = [np.zeros([p.nu]) for p in self.problem.runningModels]

        # gaps 
        self.fs = [np.zeros(self.problem.runningModels[0].state.ndx)
                     ] + [np.zeros(p.state.ndx) for p in self.problem.runningModels]

        # initializations 
        self.P[0][:,:] = self.uncertainty.P0
        self.xhat[0] = self.uncertainty.x0




