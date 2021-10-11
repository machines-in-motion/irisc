""" A class that holds process and measurement uncertainty models for the shooting problem """
import numpy as np 



class UncertaintyModel: 
    def __init__(self, process_model, measurement_model): 
        """ A class the combines model uncertainty for both process and measurements """

        self.pModel = process_model
        self.ndx = self.pModel.state.ndx 
        self.mModel = measurement_model 
        self.ny = self.mModel.ny
        self.state = self.pModel.state

    def createData(self): 
        return UncertaintyData(self)
        

    def calc(self, data, x, u): 
        self.pModel.calc(x,u)
        self.mModel.calc(x,u)   
        # in case we have some weird state dependent filter, it can be applied here 
        data.Omega[:,:] = self.pModel.filter.dot(self.pModel.Omega).dot(self.pModel.filter.T) 
        data.invOmega[:,:] = np.linalg.inv(data.Omega) 
        data.Gamma[:,:] = self.mModel.filter.dot(self.mModel.Gamma).dot(self.mModel.filter.T)  
        data.invGamma[:,:] = np.linalg.inv(data.Gamma)  
        data.H[:,:] = self.mModel.H.copy()
        data.y[:] = self.mModel.calc(x,u)

    def measurement_deviation(self, y, xn): 
        return self.mModel.deviation(y, xn)

    def measurement_increment(self, y, dy):
        return self.mModel.integrate(y, dy)

    def sample_process(self, data, x, u):
        self.calc(data, x, u)
        dist = np.random.multivariate_normal(np.zeros(self.ndx), data.Omega)
        return self.state.integrate(x, dist)

    def sample_measurement(self, data, x, u):
        self.calc(data, x, u)
        dist = np.random.multivariate_normal(np.zeros(self.ny), data.Gamma)
        return self.measurement_increment(data.y, dist)




class UncertaintyData: 
    def __init__(self, model):
        self.Omega = np.zeros([model.ndx,model.ndx])
        self.invOmega = np.zeros([model.ndx,model.ndx])
        self.Gamma = np.zeros([model.ny,model.ny])
        self.invGamma = np.zeros([model.ny,model.ny])
        self.H = np.zeros([model.ny, model.ndx])
        self.y = np.zeros(model.ny)





class ProblemUncertainty: 
    def __init__(self, x0, initialUncertainty, modelUncertainties):
        """ A class that models process and measurement uncertainties for 
        the entire shooting problem as added Gaussian noise 
        Args:
            x0: initial state mean 
            winitialUncertainty: initial state covariance 
            initialUncertainty: list contaning 
            """
        self.x0 = x0 
        self.P0 = initialUncertainty 
        self.runningModels = modelUncertainties 
        self.runningDatas = []

        for uModel in self.runningModels: 
            self.runningDatas += [uModel.createData()]

    def calc(self, xs, us): 
        for i, uModel in enumerate(self.runningModels): 
            uModel.calc(self.runningDatas[i], xs[i], us[i])
            


    def sample_process(self, t, x, u):
        """ returns a random disturbed state """
        np.random.seed()
        self.runningModels[t].calc(self.runningDatas[t], x, u)
        dist = np.random.multivariate_normal(np.zeros(self.runningModels[t].ndx),
                                            self.runningDatas[t].Omega)
        return self.runningModels[t].state.integrate(x, dist)


    def sample_measurement(self, t, x, u): 
        np.random.seed()
        self.runningModels[t].calc(self.runningDatas[t], x, u)  
        dist = np.random.multivariate_normal(np.zeros(self.runningModels[t].ny),
                                            self.runningDatas[t].Gamma)
        
        return self.runningModels[t].measurement_increment(self.runningDatas[t].y, dist)
             


