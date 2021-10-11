import numpy as np 
import scipy.linalg as scl
import os, sys
src_path = os.path.abspath('../../') # append library directory without packaging 
sys.path.append(src_path)
from utils.uncertainty import measurement_models, process_models, problem_uncertainty

class AbstractEstimator:
    def __init__(self, processModels, uncertaintyModels):
        self.uncertaintyModels = uncertaintyModels
        self.processModels = processModels
        self.processDatas = []
        self.uncertaintyDatas = []
        for iModels in self.uncertaintyModels:
            self.uncertaintyDatas += [[model.createData() for model in iModels]]

        for iModels in self.processModels:
            self.processDatas += [[model.createData() for model in iModels]]


class ExtendedKalmanFilter(AbstractEstimator):
    def __init__(self, x0, chi0, processModels, uncertaintyModels, n_steps):
        super().__init__(processModels, uncertaintyModels)
        
        self.chi = [chi0.copy()]
        self.xhat = [x0.copy()]
        self.nsteps = n_steps


    def predict(self,t,i,x,u):
        """ returns predicted state and covariance 
        Args:
            x: filtered state at previous time step 
            u: control 
            t: time index """
        model = self.uncertaintyModels[t][i] # uncertainty model
        data = self.uncertaintyDatas[t][i] # uncertainty data 
        model.calc(data, x, u)  # calculate covariance for process & measurement 
        pmodel = self.processModels[t][i] # process dynamics 
        pdata = self.processDatas[t][i] 
        pmodel.calc(pdata, x, u)  # calculate nonlinear next state 
        pmodel.calcDiff(pdata, x, u) #  calculate dynamics approximation 
        xnext = pdata.xnext.copy() # obtain next state 
        # predicted covariance 
        # print(i)
        # print(t)
        # print(self.nsteps)
        # print(i+t*self.nsteps )
        # print(len(self.chi))
        # print("walaaa shiii")
        pred_cov = pdata.Fx.dot(self.chi[i+ t*self.nsteps]).dot(pdata.Fx.T)
        pred_cov += data.Omega
        return xnext, pred_cov


    def update(self, t, i, y, xprev, u): 
        model = self.uncertaintyModels[t][i] # uncertainty model
        data = self.uncertaintyDatas[t][i] # uncertainty data 
        # recall all approximations have been calculated by now 
        xp, Cp = self.predict(t,i,xprev, u)
        innovation = y - data.y 
        mat = data.H.dot(Cp).dot(data.H.T) + data.Gamma
        Lb = scl.cho_factor(mat, lower=True) 
        s = data.H.dot(Cp.T)
        transposeG = scl.cho_solve(Lb, s)
        gain = transposeG.T
        self.xhat +=[xp + gain.dot(innovation)]
        self.chi += [(np.eye(model.ny) - gain.dot(data.H)).dot(Cp)]


class RiskSensitiveFilter(AbstractEstimator):
    def __init__(self, shootingProblem, problemUncertainty, xs, us, sensitivity):
        super().__init__(problemUncertainty)

        self.chi = [self.uncertainty.P0.copy()]
        self.xhat = [self.uncertainty.x0.copy()]
        self.problem = shootingProblem
        self.sigma = sensitivity
        self.xs = xs 
        self.us = us 
        # evaluate the cost along the nominal trajectory 
        # keep in mind all approximations are computed along nominal trajectory 
        self.problem.calc(self.xs, self.us)
        self.problem.calcDiff(self.xs, self.us)
        self.uncertainty.calc(self.xs, self.us) # compute H, Omega, Gamma 

    def update(self, t, y, xprev, u):
        """ some stuff
        Args:
            xprev: previous filtered state and not minimum stress 
        
        """
        pdata = self.problem.runningDatas[t]
        udata = self.uncertainty.runningDatas

        left = scl.linalg.inv(self.chi[t]) + udata.H.T.dot(udata.invGamma).dot(udata.H)
        left += self.sigma*pdata.Lxx

        Lb = scl.cho_factor(left, lower=True)

        # filter gain 
        rightG = udata.H.T.dot(udata.invGamma)
        preG = scl.cho_solve(Lb, rightG)
        Gain = pdata.Fx.dot(preG)
        # curvature 
        rightChi = pdata.Fx.T
        preChi = scl.cho_solve(Lb, rightChi)
        self.chi += [udata.Omega + pdata.Fx.dot(preChi)]
        # estimate 
        innovation = y-udata.H.dot(xprev)
        correction = Gain.dot(innovation)
        prediction = pdata.Fx.dot(xprev) + pdata.Fu.dot(u) + correction
        # 
        right_xhat = pdata.Lxx.dot(xprev) + pdata.Lxu.dot(u) + pdata.Lx
        pre_xhat = scl.cho_solve(Lb, right_xhat)
        self.xhat += [ prediction - self.sigma*pdata.Fx.dot(pre_xhat)]





