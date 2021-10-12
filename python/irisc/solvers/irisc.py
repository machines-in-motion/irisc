import numpy as np
from numpy.core.fromnumeric import transpose 
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
        self.th_stop = 2.e-3 # 1.e-9 
        self.n_little_improvement = 0 
        self.gap_tolerance = 1.e-6
        self.cost_try = 0.
        # 
        self.withMeasurement = withMeasurement 
        self.withGaps = False 

        self.rv_dim = 0 
        self.a = 1.#e-3 # alpha for the unscented transform 
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
    
    def computeGaps(self):
        raise NotImplementedError("computeGaps Method Not Implemented yet")

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass() 
        
    def tryStep(self, stepLength):
        # self.forwardPass(stepLength)
        self.unscentedForwardPass(stepLength)
        return self.cost - self.cost_try 

    def expectedImprovement(self):
        return np.array([0.]), np.array([0.])
        
    def stoppingCriteria(self):
        """ it will be feedforward norm along the trajectory for now """
        knormSquared = [ki.dot(ki) for ki in self.kff]
        knorm = np.sqrt(np.array(knormSquared))
        return knorm
        
    def solve(self, init_xs=None, init_us=None, maxiter=100, isFeasible=True, regInit=None):
        if isFeasible:
            self.setCandidate(init_xs, init_us, True) 
            if VERBOSE: print("solve setCandidate works just fine ")
        else: # for now if not feasible, force rollout 
            xs = self.problem.rollout(init_us)
            self.setCandidate(xs, init_us, True)
            self.unscentedForwardPass(1.)
            self.cost = self.cost_try
            print("initial cost is %s"%self.cost)

        self.n_little_improvement = 0
        self.x_reg = regInit if regInit is not None else self.regMin
        self.u_reg = regInit if regInit is not None else self.regMin
        for i in range(maxiter):
            # print("running iteration no. %s".center(LINE_WIDTH,'#')%i)
            recalc = True   # this will recalculated derivatives in Compute Direction 
            while True:     # backward pass with regularization 
                try:
                    # print("Accessing try") 
                    self.computeDirection(recalc=recalc)
                    if i == 0:
                        print("logging initial gains")
                        self.kff_init = [ki for ki in self.kff]
                        self.Kfb_init = [ki for ki in self.Kfb]
                        # self.cost = 1.e+45
                except:
                    print("compute direcrtion failed")
                    pass 
                    # recalc = True 
                    # self.increaseRegularization()
                    # print("increasing regularization at iterations %s"%i)
                    # if self.x_reg == self.regMax: # if max reg reached, faild to converge, end solve attempt  
                    #     print("Backward Pass Maximum Regularization Reached for alpha = %s"%a) 
                    #     return False #self.xs, self.us, False
                    # else:  # continue to next while  
                    #     continue
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
                    self.setCandidate(self.xs_try, self.us_try, True)
                    self.cost = self.cost_try 
                    break # stop line search and proceed to next iteration 
        
            if a > self.th_step: # decrease regularization if alpha > .5 
                self.decreaseRegularization()
            if a == self.alphas[-1] :  
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

            if self.n_little_improvement == 5:
                print('Solver converged with little improvement in the last 10 iterations')
                return True # self.xs, self.us, True

        print("Now we are completely out of the for loop")

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
            inv_Wt = udata.invOmega - self.sigma * self.V[t+1]
            Wt = np.linalg.inv(inv_Wt)
            Mt = np.eye(model.state.ndx) + self.sigma*self.V[t+1].dot(Wt)
            Qx = data.Lx + data.Fx.T.dot(Mt).dot(self.v[t+1]) 
            Qx += data.Fx.T.dot(Mt).dot(self.V[t+1]).dot(self.fs[t+1])
            Qu = data.Lu + data.Fu.T.dot(Mt).dot(self.v[t+1]) 
            Qu +=  data.Fu.T.dot(Mt).dot(self.V[t+1]).dot(self.fs[t+1])
            Qux = data.Lxu.T + data.Fu.T.dot(Mt).dot(self.V[t+1]).dot(data.Fx)
            Qxx = data.Lxx + data.Fx.T.dot(Mt).dot(self.V[t+1]).dot(data.Fx)
            Quu = data.Luu + data.Fu.T.dot(Mt).dot(self.V[t+1]).dot(data.Fu)
            # 
            Lb = scl.cho_factor(Quu, lower=True) 
            self.kff[t][:] = -scl.cho_solve(Lb, Qu)
            self.Kfb[t][:, :] = -scl.cho_solve(Lb, Qux)
            #
            self.v[t][:] = Qx + self.Kfb[t].T.dot(Quu).dot(self.kff[t]) + self.Kfb[t].T.dot(Qu) + Qux.T.dot(self.kff[t]) 
            self.V[t][:,:] = Qxx + self.Kfb[t].T.dot(Quu).dot(self.Kfb[t]) + self.Kfb[t].T.dot(Qux) + Qux.T.dot(self.Kfb[t]) 
            self.V[t][:,:] = .5*(self.V[t].T + self.V[t])



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

    def pastStressEstimate(self, t, xcheck, u, y, x_hat, P): 
        p_model = self.problem.runningModels[t]
        m_model = self.uncertainty.runningModels[t]
        p_data = self.problem.runningDatas[t]
        m_data = self.uncertainty.runningDatas[t]
        p_model.calc(p_data, x_hat, u)
        p_model.calcDiff(p_data, x_hat, u)
        # compute deviations 
        del_xhat = p_model.state.diff(self.xs[t], x_hat)
        del_y = m_model.measurement_deviation(y, self.xs[t]) 
        du = u - self.us[t]
        # compuate estimation gain 
        inv_skewed_covariance = np.linalg.inv(P) + m_data.H.T.dot(m_data.invGamma).dot(m_data.H) + self.sigma*p_data.Lxx
        Lb = scl.cho_factor(inv_skewed_covariance , lower=True)
        # compute filter gain 
        rightG = m_data.H.T.dot(m_data.invGamma)
        gain = scl.cho_solve(Lb, rightG)
        self.G[t][:,:] = p_data.Fx.dot(gain)
        # update deviation estimate 
        right_dxhat = p_data.Lxx.dot(del_xhat) - p_data.Lxu.dot(du) - p_data.Lx 
        dx_hat =  scl.cho_solve(Lb, right_dxhat)
        estimate_dx =- self.sigma*p_data.Fx.dot(dx_hat)
        estimate_dx += p_data.Fx.dot(del_xhat) + p_data.Fu.dot(du)  
        estimate_dx += self.G[t].dot(del_y - m_data.H.dot(del_xhat)) 
        # update deviation covariance 
        right_P = p_data.Fx.T 
        Pt = scl.cho_solve(Lb, right_P)
        P_next = m_data.Omega + p_data.Fx.dot(Pt)
        # update estimates 
        x_next = p_model.state.integrate(self.xs[t+1], estimate_dx)
        return x_next, P_next

    def controllerStep(self, t, x_hat, P, interpolation=0.): 
        """ runs a single control update step which includes 
        1. compute past stress estimate \hat{x}_t , P_t, G_t 
        2. compute minimal stress estimate \check{x}_t 
        3. compute deviation of minimal stress estimate from nominal trajectory x^n_t 
        4. update feedforward and feedback accordingly 
        Args: 
            x: disturbed x coming back from simulator 
            t: time index along planned horizon
        """
        p_model = self.problem.runningModels[t-1]
        m_model = self.uncertainty.runningModels[t-1]
        p_data = self.problem.runningDatas[t-1]
        m_data = self.uncertainty.runningDatas[t-1]
        # compute deviations 
        del_xhat = p_model.state.diff(self.xs[t], x_hat)

        # compute x_check 
        left_dx_ch =  np.eye(p_model.state.ndx) + self.sigma*P.dot(self.V[t])  
        Lb_dx_ch = scl.cho_factor(left_dx_ch , lower=True) 
        right_dx_ch = del_xhat - self.sigma * P.dot(self.v[t])
        del_xcheck = scl.cho_solve(Lb_dx_ch, right_dx_ch)
        #
        xcheck = p_model.state.integrate(self.xs[t], del_xcheck)
        #
        right_kff = P.dot(self.v[t])
        kff = scl.cho_solve(Lb_dx_ch, right_kff)
        self.k[t-1] = self.us[t-1] + self.sigma*self.Kfb[t-1].dot(kff)
        #
        right_Kfb = self.Kfb[t-1].T 
        tran_left_dx_ch = left_dx_ch.T 
        Lb_tran_dx_ch = scl.cho_factor(tran_left_dx_ch, lower=True)
        Kfb = scl.cho_solve(Lb_tran_dx_ch, right_Kfb) 
        #
        self.K[t-1] = Kfb.T  
        ui = self.k[t-1] - self.K[t-1].dot(del_xhat)
        return xcheck, ui 

    def perfectObservationControl(self, t, x): 
        err = self.problem.runningModels[t].state.diff(self.xs[t], x)
        control = self.us[t] - self.Kfb[t].dot(err)
        return control 

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

    def expected_cost(self): 
        costs = []
        for i in range(self.sample_size):
            if i == 0:
                costs += [self.w0 * np.exp(.5*self.sigma *np.sum(self.sample_costs[:,i])) ]
            else:
                costs += [self.wi * np.exp(.5*self.sigma *np.sum(self.sample_costs[:,i])) ]
         
        ctry =  (2./self.sigma)*np.log(sum(costs))
        return ctry

    def unscentedForwardPass(self, stepLength, warning='error'):
        self.cost_try = 0
        self.xs_try[0] = self.xs[0].copy()
        self.fs[0] = np.zeros(self.problem.runningModels[0].state.ndx)

        for i in range(self.sample_size): 
            
            xs_try = [self.problem.x0] + [np.nan] * self.problem.T
            us_try =  [np.nan] * self.problem.T
        
            for t, (m, d) in enumerate(zip(self.problem.runningModels, self.problem.runningDatas)):
                # update control 
                ti = sum(self.ndxs[:t])
                tf = sum(self.ndxs[:t+1])
                if i != 0:
                    xs_try[t] = m.state.integrate(xs_try[t], self.samples[ti:tf, i])
                us_try[t] = self.us[t] + stepLength*self.kff[t] + \
                    self.Kfb[t].dot(m.state.diff(self.xs[t], xs_try[t]))
                
                with np.warnings.catch_warnings():
                    np.warnings.simplefilter(warning)
                    m.calc(d, xs_try[t], us_try[t])
                # update state 
                xs_try[t + 1] = d.xnext.copy()  # not sure copy helpful here.
                self.sample_costs[t, i] = d.cost
                
                raiseIfNan([self.sample_costs[t, i], d.cost], BaseException('forward error'))
                raiseIfNan(xs_try[t + 1], BaseException('forward error'))
                # store undisturbed trajectory 
                if i == 0:
                    self.xs_try[t+1] = xs_try[t+1].copy()
                    self.us_try[t] = us_try[t].copy()
                    self.fs[t+1] = np.zeros(m.state.ndx)

            with np.warnings.catch_warnings():
                np.warnings.simplefilter(warning)
                fm = self.problem.terminalModel
                xs_try[-1] = fm.state.integrate(xs_try[-1], self.samples[-fm.state.ndx:,i])
                self.problem.terminalModel.calc(self.problem.terminalData, xs_try[-1])
                self.sample_costs[-1, i] = self.problem.terminalData.cost
            raiseIfNan(self.sample_costs[-1, i], BaseException('forward error'))
        
        self.cost_try = self.expected_cost()
        self.isFeasible = True 
        return self.xs_try, self.us_try, self.cost_try