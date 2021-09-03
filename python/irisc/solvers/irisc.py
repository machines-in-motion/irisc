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
        self.cost_try = 0.
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
    
    def computeGaps(self):
        raise NotImplementedError("computeGaps Method Not Implemented yet")
    
    

    def computeDirection(self, recalc=True):
        if recalc:
            if VERBOSE: print("Going into Calc from compute direction")
            self.calc()
        
        if VERBOSE: print("Going into Backward Pass from compute direction")
        self.backwardPass() 
        

    def tryStep(self, stepLength):
        self.forwardPass(stepLength)
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
                        print("Backward Pass Maximum Regularization Reached for alpha = %s"%a) 
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
                    self.setCandidate(self.xs_try, self.us_try, True)
                    self.cost = self.cost_try 
                    break # stop line search and proceed to next iteration 
        

            if a > self.th_step: # decrease regularization if alpha > .5 
                self.decreaseRegularization()
            if a == self.alphas[-1] :  
                self.increaseRegularization()
                if self.x_reg == self.regMax:
                    return False # self.xs, self.us, False
            
            self.stepLength = a
            self.iter = i
            self.stop = sum(self.stoppingCriteria())
            
            if self.getCallbacks() is not None: # this way callback prints appear before solver convergence message 
                [c(self) for c in self.getCallbacks()]

            if  self.stop < self.th_stop:
                self.n_little_improvement += 1

            if self.n_little_improvement == 10:
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
            

            inv_V = np.linalg.inv(self.V[t+1])
            if VERBOSE: print("inv V[%s]"%t)
            self.M[t] = np.linalg.inv(self.sigma * udata.Omega + inv_V)
            if VERBOSE: print("M[%s]"%t)
            N_t = self.v[t+1] - self.sigma * self.M[t].dot(udata.Omega).dot(self.v[t+1])
            if VERBOSE: print("N[%s]"%t)
            # 
            uleft = data.Luu + data.Fu.T.dot(self.M[t]).dot(data.Fu)
            # print("Quu \n", uleft)
            Lb = scl.cho_factor(uleft + self.u_reg*np.eye(model.nu), lower=True)
            uff_right = data.Lu + data.Fu.T.dot(self.M[t].dot(self.fs[t+1]) + N_t)
            ufb_right = data.Lxu.T + data.Fu.T.dot(self.M[t]).dot(data.Fx) 
            # print("Qxu \n", ufb_right)

            print("K fb \n", scl.cho_solve(Lb, ufb_right))
            # 
            self.kff[t][:] = scl.cho_solve(Lb, uff_right)
            if VERBOSE: print("kff[%s]"%t)
            self.Kfb[t][:, :] = scl.cho_solve(Lb, ufb_right)
            if VERBOSE: print("Kfb[%s]"%t)
            # aux term 
            A_BK = data.Fx - data.Fu.dot(self.Kfb[t])
            # hessian 
            self.V[t][:,:] = data.Lxx + self.Kfb[t].T.dot(data.Luu).dot(self.Kfb[t]) - data.Lxu.dot(self.Kfb[t]) - self.Kfb[t].T.dot(data.Lxu.T)
            self.V[t][:,:] += A_BK.T.dot(self.M[t]).dot(A_BK)
            self.V[t][:,:] = .5 *(self.V[t] + self.V[t].T) # ensure symmetry 
            if VERBOSE: print("V[%s]"%t)
            # gradient 
            self.v[t][:] = data.Lx + self.Kfb[t].T.dot(data.Luu).dot(self.kff[t]) - self.Kfb[t].T.dot(data.Lu) - data.Lxu.dot(self.kff[t])
            self.v[t][:] += A_BK.T.dot(self.M[t]).dot(self.fs[t+1]- data.Fu.dot(self.kff[t])) 
            self.v[t][:] += A_BK.T.dot(N_t)
            if VERBOSE: print("v[%s]"%t)

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


    def controllerStep(self, t, y, u=None, interpolation=0.): 
        """ runs a single control update step which includes 
        1. compute past stress estimate \hat{x}_t , P_t, G_t 
        2. compute minimal stress estimate \check{x}_t 
        3. compute deviation of minimal stress estimate from nominal trajectory x^n_t 
        4. update feedforward and feedback accordingly 
        Args: 
            x: disturbed x coming back from simulator 
            t: time index along planned horizon
        """
        
        # if t == 0: 
        #     left_dx_ch = np.eye(self.problem.runningModels[t].state.ndx) + self.sigma*self.P[t].dot(self.V[t]) 
        #     Lb_dx_ch = scl.cho_factor(left_dx_ch , lower=True)
        #     right_dx_ch = - self.sigma * self.P[t].dot(self.v[t])
        #     self.delta_xcheck[0][:]  = scl.cho_solve(Lb_dx_ch, right_dx_ch)
        #     self.xcheck[0][:] = self.problem.runningModels[t].state.integrate(self.xhat[0], self.delta_xcheck[0]) 
        # else:
        #     pdata = self.problem.runningDatas[t-1]
        #     pmodel = self.problem.runningModels[t]
        #     mdata = self.uncertainty.runningDatas[t]
        #     mmodel = self.uncertainty.runningModels[t]


        #     delta_u = u - self.us[t] 
        #     delta_y = mmodel.measurement_deviation(y, self.xs[t-1]) 

        #     inv_skewed_covariance = np.linalg.inv(self.P[t-1]) + mdata.H.T.dot(mdata.invGamma).dot(mdata.H) + self.sigma*pdata.Lxx
        #     Lb = scl.cho_factor(inv_skewed_covariance , lower=True)
        #     # compute filter gain 
        #     rightG = mdata.H.T.dot(mdata.invGamma)
        #     gain = scl.cho_solve(Lb, rightG)
        #     self.G[t][:,:] = pdata.Fx.dot(gain)
        #     # cmpute xhat             
        #     right_dxhat = pdata.Lxx.dot(self.delta_xhat[t-1]) - pdata.Lxu.dot(delta_u) - pdata.Lx 
        #     dx_hat =  scl.cho_solve(Lb, right_dxhat)
        #     self.delta_xhat[t][:] = - self.sigma*pdata.A.dot(dx_hat)
        #     self.delta_xhat[t][:] += pdata.Fx.dot(self.delta_xhat[t-1]) + pdata.Fu.dot(delta_u)  
        #     self.delta_xhat[t][:] += self.G[t].dot(delta_y - mdata.H.dot(self.delta_xhat[t-1])) 
        #     # compute P 
        #     right_P = pdata.Fx.T 
        #     Pt = scl.cho_solve(Lb, right_P)
        #     self.P[t][:,:] = mdata.Omega + pdata.Fx.dot(Pt)

        #     # compute x_check 
        #     left_dx_ch =  np.eye(pmodel.state.ndx) + self.sigma*self.P[t].dot(self.V[t])  
        #     Lb_dx_ch = scl.cho_factor(left_dx_ch , lower=True) 
        #     right_dx_ch = self.delta_xhat[t] - self.sigma * self.P[t].dot(self.v[t])
        #     self.delta_xcheck[t][:] = scl.cho_solve(Lb_dx_ch, right_dx_ch)
        #     self.xcheck[t][:] = pmodel.state.integrate(self.xs[t], self.delta_xcheck[t])
        return self.us[t]



    def perfectObservationControl(self, t, x): 
        err = self.problem.runningModels[t].state.diff(self.xs[t], x)
        print("error norm\n", np.linalg.norm(err))
        # print("feedback norm norm\n",np.linalg.norm(self.Kfb[t]))
        print("max feedback gain\n",np.amax(np.abs(self.Kfb[t])))
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
        self.kff = [np.zeros([p.nu]) for p in self.problem.runningModels]
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


        # print(len(self.V))
        # print(len(self.problem.runningModels))
        # print(len(self.problem.runningDatas))
        # print(len(self.uncertainty.runningModels))
        # print(len(self.uncertainty.runningDatas))



