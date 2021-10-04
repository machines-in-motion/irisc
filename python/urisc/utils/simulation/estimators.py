import numpy as np 
import scipy.linalg as scl

class AbstractEstimator:
    def __init__(self, problemUncertainty):
        self.uncertainty = problemUncertainty


class ExtendedKalmanFilter(AbstractEstimator):
    def __init__(self, problemUncertainty):
        super().__init__(problemUncertainty)
        self.chi = [self.uncertainty.P0.copy()]
        self.xhat = [self.uncertainty.x0.copy()]


    def predict(self,t,x,u):
        """ returns predicted state and covariance 
        Args:
            x: filtered state at previous time step 
            u: control 
            t: time index """
        model = self.uncertainty.runningModels[t] # uncertainty model
        data = self.uncertainty.runningDatas[t] # uncertainty data 
        model.calc(data, x, u)  # calculate covariance for process & measurement 
        pmodel = self.uncertainty.runningModels[t].pModel # process dynamics 
        pmodel.model.calc(pmodel.data, x, u)  # calculate nonlinear next state 
        pmodel.model.calcDiff(pmodel.data, x, u) #  calculate dynamics approximation 
        xnext = pmodel.data.xnext.copy() # obtain next state 
        # predicted covariance 
        pred_cov = pmodel.data.Fx.dot(self.chi[t]).dot(pmodel.data.Fx.T)
        pred_cov += data.Omega
        return xnext, pred_cov


    def update(self, t, y, xprev, u): 
        model = self.uncertainty.runningModels[t] # uncertainty model
        data = self.uncertainty.runningDatas[t] # uncertainty data 
        # recall all approximations have been calculated by now 
        xp, Cp = self.predict(t,xprev, u)
        innovation = y - data.y 
        mat = data.H.dot(Cp).dot(data.H.T) + data.Gamma
        Lb = scl.cho_factor(mat, lower=True) 
        s = data.H.dot(Cp.T)
        transposeG = scl.cho_solve(Lb, s)
        gain = transposeG.T
        self.xhat +=[xp + gain.dot(innovation)]
        self.chi += [(np.eye(model.ny) - gain.dot(data.H)).dot(Cp)]


class RiskSensitiveFilter(AbstractEstimator):
    def __init__(self, problemUncertainty):
        super().__init__(problemUncertainty)

