import numpy as np 
import os, sys
src_path = os.path.abspath('../') 
sys.path.append(src_path)

from test_factory import create_point_cliff_models

from utils.uncertainty import measurement_models, process_models, problem_uncertainty

def test_point_cliff_full_state(): 
    dt = 0.01 
    horizon = 3
    models = create_point_cliff_models(dt, horizon) 

    x0 = np.zeros(4)
    initial_covariance = 1.e-4 * np.eye(4)
    uncertainty_models = []

    for i, m in enumerate(models[:-1]):
        # loop only over running models 
        process_noise = 1.e-5*np.eye(m.state.ndx)
        p_model = process_models.FullStateProcess(m, process_noise) 
        measurement_noise = 1.e-3*np.eye(m.state.ndx)
        m_model = measurement_models.FullStateMeasurement(m, measurement_noise)
        uncertainty_models += [problem_uncertainty.UncertaintyModel(p_model, m_model)]

 
    pUncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, uncertainty_models)





if __name__=="__main__":
    test_point_cliff_full_state() 

