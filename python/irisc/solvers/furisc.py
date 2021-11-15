import numpy as np
from numpy import linalg
from numpy.core.fromnumeric import shape, transpose 
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




class FeasibilityRiskSensitiveSolver(SolverAbstract):
    def __init__(self, shootingProblem, problemUncertainty, sensitivity):
        SolverAbstract.__init__(self, shootingProblem)
        self.sigma = sensitivity
        self.inv_sigma = 1./self.sigma
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
        self.th_stop =  1.e-9 
        self.n_little_improvement = 0 
        self.n_min_alpha = 0
        self.gap_tolerance = 1.e-9
        self.cost_try = 0.
        # 
        self.rv_dim = 0 
        self.a = 1.e-2 # alpha for the unscented transform 
        # 
        self.allocateData()
        # 
        self.initialize_unscented_transform()

    def models(self):
        mod = [m for m in self.problem.runningModels]
        mod += [self.problem.terminalModel]
        return mod 

    def calc(self):
        # compute cost and derivatives at deterministic nonlinear trajectory 
        self.problem.calc(self.xs, self.us)
        self.problem.calcDiff(self.xs, self.us)
        self.uncertainty.calc(self.xs, self.us) # compute H, Omega, Gamma 
        # print("went into calc")

        # if not self.isFeasible:
        # Gap store the state defect from the guess to feasible (rollout) trajectory, i.e.
        #   gap = x_rollout [-] x_guess = DIFF(x_guess, x_rollout)
        # print("not feasible")
        self.fs[0] = self.problem.runningModels[0].state.diff(self.xs[0], self.problem.x0)
        # print("initial gap")
        ng = np.linalg.norm(self.fs[0])
        # print("initial gap norm")
        for t, (m, d, x) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas, self.xs[1:])):
            self.fs[t + 1] = m.state.diff(x, d.xnext)
            ng += np.linalg.norm(self.fs[t+1])
        
        if ng<self.gap_tolerance:
            self.isFeasible = True
        else:
            self.isFeasible = False 

        # print("calc passed")

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass() 
        # print("Backward Pass is Done")
        
    def tryStep(self, stepLength):
        self.unscentedForwardPass(stepLength)
        return self.cost - self.cost_try 

    def expectedImprovement(self):
        # dL = -self.inv_sigma*np.log(np.exp(-self.sigma*self.dv[0]))
        dL = self.dv[0]
        # print("Expected improvement is %s"%dL)
        return np.array([0.]), np.array([0.])
        
    def stoppingCriteria(self):
        """ it will be feedforward norm along the trajectory for now """
        knormSquared = [ki.dot(ki) for ki in self.k]
        knorm = np.sqrt(np.array(knormSquared))
        return knorm
        
    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=False, regInit=None):
        self.setCandidate(init_xs, init_us, isFeasible)
        self.cost = self.unscentedCost()
        print("initial cost is %s"%self.cost)
        print("initial trajectory feasibility %s"%self.isFeasible)      
        self.n_little_improvement = 0
        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin

        for i in range(maxiter):
            # print("running iteration no. %s".center(LINE_WIDTH,'#')%i)
            recalc = True   # this will recalculated derivatives in Compute Direction 
            while True:     # backward pass with regularization 
                try:
                    # print("iteration number %s"%i) 
                    self.computeDirection(recalc=recalc)
                    # _,_ = self.expectedImprovement()
                except:
                    print("compute direcrtion failed") 
                    recalc = True 
                    self.increaseRegularization()
                    print("increasing regularization at iterations %s"%i)
                    if self.x_reg == self.regMax: # if max reg reached, faild to converge, end solve attempt  
                        print("Backward Pass Maximum Regularization Reached at iteration %s"%i) 
                        return False #self.xs, self.us, False
                    else:  # continue to next while  
                        continue
                break # compute direction succeeded, exit and proceed to line search 
            
            for a in self.alphas:
                try: 
                    self.dV = self.tryStep(a)
                except:
                    # repeat starting from a smaller alpha 
                    print("Try Step Faild for alpha = %s"%a) 
                    continue 
            
                if self.dV > 0.: # Cost has decreased, accept step 
                    # print("step accepted at iteration %s for alpha %s"%(i,a))
                    self.setCandidate(self.xs_try, self.us_try, self.isFeasible)
                    self.cost = self.cost_try 
                    if self.dV < 1.e-12:
                        self.n_little_improvement += 1
                    break # stop line search and proceed to next iteration 
        
            if a > self.th_step: # decrease regularization if alpha > .5 
                self.decreaseRegularization()
                self.n_min_alpha = 0
            if a == self.alphas[-1] :  
                self.n_min_alpha += 1
                self.increaseRegularization()
                if self.x_reg == self.regMax:
                    print(" Maximum Regularixation Reached with no Convergence ".center(LINE_WIDTH,'='))
                    return False # self.xs, self.us, False
            
            self.stepLength = a
            self.iter = i
            self.stop = sum(self.stoppingCriteria())
            
            if self.getCallbacks() is not None: # this way callback prints appear before solver convergence message 
                [c(self) for c in self.getCallbacks()]

            if  self.stop < self.th_stop:
                self.n_little_improvement += 1

            if self.n_little_improvement == 10:
                print('Solver converged with little improvement in the last 5 iterations')
                return True 

            if self.n_min_alpha == 10:
                print("Line search is not making any improvements")
                return False 

        # print("Now we are completely out of the for loop")

        # Warning: no convergence in max iterations
        print('max iterations with no convergance')
        return False 

    def backwardPass(self):
        self.V[-1][:,:] = self.problem.terminalData.Lxx
        self.v[-1][:] = self.problem.terminalData.Lx

        for t, (model, data, umodel, udata) in rev_enumerate(zip(self.problem.runningModels,
                                                        self.problem.runningDatas,
                                                        self.uncertainty.runningModels,  
                                                        self.uncertainty.runningDatas)):
            inv_V = np.linalg.inv(self.V[t+1]) + self.x_reg*np.eye(model.state.ndx)
            if VERBOSE: print("inv V[%s]"%t)
            self.M[t] = np.linalg.inv(self.sigma * udata.Omega + inv_V)
            # self.M[t] = .5*(self.M[t]  + self.M[t].T) # a bit more stability 
            if VERBOSE: print("M[%s]"%t)
            N_t = self.v[t+1] - self.sigma * self.M[t].dot(udata.Omega).dot(self.v[t+1])
            if VERBOSE: print("N[%s]"%t)
            #
            Qx = data.Lx + data.Fx.T.dot(self.M[t]).dot(self.fs[t+1]) + data.Fx.T.dot(N_t) 
            Qu = data.Lu + data.Fu.T.dot(self.M[t]).dot(self.fs[t+1]) + data.Fu.T.dot(N_t) 
            Quu = data.Luu + data.Fu.T.dot(self.M[t]).dot(data.Fu) + self.u_reg*np.eye(model.nu)
            # Quu = .5*(Quu + Quu.T) # a bit more stability 
            Qux = data.Lxu.T + data.Fu.T.dot(self.M[t]).dot(data.Fx) 
            if len(Qux.shape) == 1:
                Qux = np.resize(Qux,(1,Qux.shape[0]))
            Qxx = data.Lxx + data.Fx.T.dot(self.M[t]).dot(data.Fx) 
            # compute the optimal control 
            Lb = scl.cho_factor(Quu, lower=True)
            self.k[t][:] = scl.cho_solve(Lb, Qu)
            if VERBOSE: print("kff[%s]"%t)
            if VERBOSE:
                print("Qu is given by \n", Qu)
                print("Quu is given by \n", Quu)
                print("Qux is given by \n", Qux)
            self.K[t][:, :] = scl.cho_solve(Lb, Qux)
            # hessian
            self.V[t][:,:] = Qxx + self.K[t].T.dot(Quu).dot(self.K[t]) - self.K[t].T.dot(Qux) - Qux.T.dot(self.K[t])
            self.V[t][:,:] = .5 *(self.V[t] + self.V[t].T) # ensure symmetry 
            if VERBOSE: print("V[%s]"%t)
            # gradient 
            self.v[t][:] = Qx + self.K[t].T.dot(Quu).dot(self.k[t]) - self.K[t].T.dot(Qu) - Qux.T.dot(self.k[t])
            if VERBOSE: print("v[%s]"%t)
            # improvement 
            # invWt = np.linalg.inv(self.inv_sigma*udata.invOmega + self.V[t+1])
            # gt = -.5*self.v[t+1].T.dot(invWt).dot(self.v[t+1]) + self.dv[t+1]
            # qt = gt + self.fs[t+1].T.dot(self.M[t].dot(self.fs[t+1]) + N_t) #- .5*self.inv_sigma*np.log(linalg.det(2*np.pi*invWt))
            # self.dv[t] = qt - self.k[t].T.dot(Qu) + .5*self.k[t].T.dot(Quu).dot(self.k[t])
              
    def unscentedForwardPass(self, stepLength, warning='error'):
        self.cost_try = 0
        self.xs_try[0] = self.xs[0].copy()
        m0 = self.problem.runningModels[0]
        for i in range(self.sample_size): 
            
            xs_try = [m0.state.integrate(self.problem.x0, (stepLength-1)*self.fs[0])] + [np.nan] * self.problem.T
            us_try =  [np.nan] * self.problem.T
        
            for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # update control 
                if t == 0:
                    ti = 0 
                    tf = self.ndxs[0]
                else:
                    ti = sum(self.ndxs[:t])
                    tf = sum(self.ndxs[:t+1])
                if i != 0 and self.samples[ti:tf, i].dot(self.samples[ti:tf, i])>1.e-10:
                    xs_try[t] = m.state.integrate(xs_try[t], self.samples[ti:tf, i])
                us_try[t] = self.us[t] - stepLength*self.k[t] - \
                    self.K[t].dot(m.state.diff(self.xs[t], xs_try[t]))

                with np.warnings.catch_warnings():
                    np.warnings.simplefilter(warning)
                    m.calc(d, xs_try[t], us_try[t])

                # update state 
                xs_try[t + 1] = m.state.integrate(d.xnext.copy(), (stepLength-1)*self.fs[t+1])  
                self.sample_costs[t, i] = d.cost
                raiseIfNan([self.sample_costs[t, i], d.cost], BaseException('forward error'))
                raiseIfNan(xs_try[t + 1], BaseException('forward error'))
                # store undisturbed trajectory 
                if i == 0:
                    self.xs_try[t+1] = xs_try[t+1].copy()
                    self.us_try[t] = us_try[t].copy()

            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                fm = self.problem.terminalModel
                xs_try[-1] = fm.state.integrate(xs_try[-1], self.samples[-fm.state.ndx:,i])
                self.problem.terminalModel.calc(self.problem.terminalData, xs_try[-1])
                self.sample_costs[-1, i] = self.problem.terminalData.cost
            raiseIfNan(self.sample_costs[-1, i], BaseException('forward error'))
        
        self.cost_try = self.expected_cost()
        # self.isFeasible = True 
        return self.xs_try, self.us_try, self.cost_try

    def forwardPass(self, stepLength, warning='error'):
        ctry = 0
        self.xs_try[0] = self.xs[0].copy()
        self.fs[0] = np.zeros(self.problem.runningModels[0].state.ndx)

        for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
            # update control 
            self.us_try[t] = self.us[t] - stepLength*self.kff[t] - \
                self.Kfb[t].dot(m.state.diff(self.xs[t], self.xs_try[t]))
            
            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                m.calc(d, self.xs_try[t], self.us_try[t])
            # update state 
            self.xs_try[t + 1] = d.xnext.copy()  # not sure copy helpful here.
            ctry += d.cost
            self.fs[t+1] = np.zeros(m.state.ndx)
            raiseIfNan([ctry, d.cost], BaseException('forward error'))
            raiseIfNan(self.xs_try[t + 1], BaseException('forward error'))
        with np.warnings.catch_warnings():
            np.warnings.simplefilter(warning)
            self.problem.terminalModel.calc(
                self.problem.terminalData, self.xs_try[-1])
            ctry += self.problem.terminalData.cost
        raiseIfNan(ctry, BaseException('forward error'))
        self.cost_try = ctry
        self.isFeasible = True 
        return self.xs_try, self.us_try, ctry


    def expected_cost(self): 
        costs = []
        for i in range(self.sample_size):
            if i == 0:
                costs += [self.w0 * np.exp(-self.sigma *np.sum(self.sample_costs[:,i])) ]
            else:
                costs += [self.wi * np.exp(-self.sigma *np.sum(self.sample_costs[:,i])) ]
        ctry =  - self.inv_sigma*np.log(sum(costs))
        return ctry

    
    def unscentedCost(self, warning='error'):
        for i in range(self.sample_size): 
            
            xs_try = [xi.copy() for xi in self.xs] 
            us_try = [ui.copy() for ui in self.us]
        
            for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # update control 
                ti = sum(self.ndxs[:t])
                tf = sum(self.ndxs[:t+1])
                if i != 0 and self.samples[ti:tf, i].dot(self.samples[ti:tf, i])>1.e-10:
                    xs_try[t] = m.state.integrate(xs_try[t], self.samples[ti:tf, i])
       
                with np.warnings.catch_warnings():
                    np.warnings.simplefilter(warning)
                    m.calc(d, xs_try[t], us_try[t])
                # update state 

                self.sample_costs[t, i] = d.cost
                raiseIfNan([self.sample_costs[t, i], d.cost], BaseException('forward error'))
                raiseIfNan(xs_try[t], BaseException('forward error'))

            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                fm = self.problem.terminalModel
                xs_try[-1] = fm.state.integrate(xs_try[-1], self.samples[-fm.state.ndx:,i])
                self.problem.terminalModel.calc(self.problem.terminalData, xs_try[-1])
                self.sample_costs[-1, i] = self.problem.terminalData.cost
            raiseIfNan(self.sample_costs[-1, i], BaseException('forward error'))
        
        cost = self.expected_cost()
        # self.isFeasible = True 
        return cost
 

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
        # backward pass variables 
        self.M = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.models()]   
        self.Kfb = [np.zeros([p.nu, p.state.ndx]) for p in self.problem.runningModels] 
        self.kff = [np.zeros(p.nu) for p in self.problem.runningModels]
        self.V = [np.zeros([p.state.ndx, p.state.ndx]) for p in self.models()]   
        self.v = [np.zeros(p.state.ndx) for p in self.models()]   
        self.dv = [0. for _ in self.models()]
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
        self.xhat[0][:] = self.uncertainty.x0


        self.delta_xhat = [np.zeros(p.state.ndx) for p in self.models()]   
        self.delta_xcheck = [np.zeros(p.state.ndx) for p in self.models()]   

        self.kff_init = None 
        self.Kfb_init = None # first iteration 



    def initialize_unscented_transform(self):
        # for now this is demo specific, this would have to be initialized in a cleaner way 
        self.uncertainty.calc(self.xs, self.us) # compute H, Omega, Gamma 
        self.ndxs = [p.state.ndx for p in self.models()] 
        self.rv_dim = sum(self.ndxs) 
        self.sample_size = 2*self.rv_dim + 1
        self.samples = np.zeros([self.rv_dim, self.sample_size]) 
        self.sample_costs = np.zeros([len(self.ndxs), self.sample_size])
        # this is actually on the disturbance sample, not the actual trajectory 
        self.lam = (self.a**2 - 1) * self.rv_dim
        self.w0 = self.lam/(self.rv_dim + self.lam) 
        self.wi = .5/(self.rv_dim + self.lam)
        self.P = np.zeros([self.rv_dim, self.rv_dim])
        # fill the P matrix, scale it then compute its square root 
        j = 0 
        for i, m in enumerate(self.models()):
            if i == 0:
                self.P[j:j+m.state.ndx, j:j+m.state.ndx] = self.uncertainty.P0
            else:
                self.P[j:j+m.state.ndx, j:j+m.state.ndx] = self.uncertainty.runningDatas[i-1].Omega 
            j += m.state.ndx
        
        self.P *= (self.rv_dim + self.lam)
        self.rootP = scl.sqrtm(self.P)
        # loop and compute the samples 

        for i in range(self.rv_dim): 
            self.samples[:, i+1] = self.rootP[:,i] 
            self.samples[:, i+1+self.rv_dim] = - self.rootP[:, i]

   