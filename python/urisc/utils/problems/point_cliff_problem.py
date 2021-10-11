""" here we setup a point cliff problem with uncertainty model """
import numpy as np  
import os, sys
src_path = os.path.abspath('../../') # append library directory without packaging 
sys.path.append(src_path)
from utils.uncertainty import measurement_models, process_models, problem_uncertainty
from utils.action_models import point_cliff 
import crocoddyl 



def full_state_uniform_cliff_problem(plan_dt, horizon, process_noise, measurement_noise, dt_control=None):
    """ creates the cliff problem for the defined dt and horizon 
    also creates the uncertainty models along this horizon with uniform noise and 
    full state measurement model """
    """ creates model of cliff point for testing purposes """
    cliff_diff_running =  point_cliff.DifferentialActionModelCliff()
    cliff_diff_terminal = point_cliff.DifferentialActionModelCliff(isTerminal=True)
    cliff_running = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, plan_dt) 
    cliff_terminal = crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, plan_dt) 
    models = [cliff_running]*(horizon) + [cliff_terminal]
    uncertainty_models = []
    for i, m in enumerate(models[:-1]):
        p_model = process_models.FullStateProcess(m, np.sqrt(plan_dt)*process_noise) 
        m_model = measurement_models.FullStateMeasurement(m, np.sqrt(plan_dt)*measurement_noise)
        uncertainty_models += [problem_uncertainty.UncertaintyModel(p_model, m_model)]

    estimation_models = []
    estimation_uncertainty = []
    if dt_control is not None:
        n_steps = int(plan_dt/dt_control)
        cliff_running_est = crocoddyl.IntegratedActionModelEuler(cliff_diff_running, dt_control) 
        
        for t in range(horizon):
            p_model = process_models.FullStateProcess(models[t], np.sqrt(plan_dt)*process_noise) 
            m_model = measurement_models.FullStateMeasurement(models[t], np.sqrt(plan_dt)*measurement_noise)
            umodel = problem_uncertainty.UncertaintyModel(p_model, m_model)
            estimation_models += [[cliff_running_est]*n_steps]
            estimation_uncertainty += [[umodel]*n_steps]
        estimation_models += [[crocoddyl.IntegratedActionModelEuler(cliff_diff_terminal, dt_control) ]]


    return models, uncertainty_models, estimation_models, estimation_uncertainty  


