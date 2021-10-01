""" here we setup a point cliff problem with uncertainty model """
import numpy as np  
import os, sys
src_path = os.path.abspath('../../') # append library directory without packaging 
sys.path.append(src_path)
from utils.uncertainty import measurement_models, process_models, problem_uncertainty
from utils.action_models import point_cliff 
import crocoddyl 



def full_state_uniform_cliff_problem(dt, horizon, process_noise, measurement_noise):
    """ creates the cliff problem for the defined dt and horizon 
    also creates the uncertainty models along this horizon with uniform noise and 
    full state measurement model """
    """ creates model of cliff point for testing purposes """
    cliff_diff_running =  point_cliff.DifferentialActionModelCliff()
    cliff_diff_terminal = point_cliff.DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, dt) 
    models = [cliff_running]*(horizon-1) + [cliff_terminal]
    uncertainty_models = []
    for i, m in enumerate(models[:-1]):
        p_model = process_models.FullStateProcess(m, process_noise) 
        m_model = measurement_models.FullStateMeasurement(m, measurement_noise)
        uncertainty_models += [problem_uncertainty.UncertaintyModel(p_model, m_model)]


    return models, uncertainty_models  


