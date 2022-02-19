""" here we setup a point cliff problem with uncertainty model """
import numpy as np  
import os, sys
src_path = os.path.abspath('../../') # append library directory without packaging 
sys.path.append(src_path)
from utils import measurement_models, process_models, problem_uncertainty
from models import pneumatic_hopper 
import crocoddyl 


def full_state_uniform_hopper(plan_dt, horizon, process_noise, measurement_noise, dt_control): 
    models = []
    for t in range(horizon): 
        diff_hopper = pneumatic_hopper.DifferentialActionModelHopper(t, horizon, False) 
        models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, plan_dt)] 
    diff_hopper = pneumatic_hopper.DifferentialActionModelHopper(horizon, horizon, True) 
    models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, plan_dt)]
    #
    uncertainty_models = []
    for i, m in enumerate(models[:-1]):
        p_model = process_models.FullStateProcess(m, np.sqrt(plan_dt)*process_noise) 
        m_model = measurement_models.FullStateMeasurement(m, np.sqrt(plan_dt)*measurement_noise)
        uncertainty_models += [problem_uncertainty.UncertaintyModel(p_model, m_model)]
    # 
    estimation_models = []
    estimation_uncertainty = []
    if dt_control is not None:
        n_steps = int(plan_dt/dt_control)
        for t in range(horizon):
            diff_hopper = pneumatic_hopper.DifferentialActionModelHopper(t, horizon, False) 
            hopper_running_est_model = crocoddyl.IntegratedActionModelEuler(diff_hopper, dt_control)
            p_model = process_models.FullStateProcess(hopper_running_est_model, np.sqrt(plan_dt)*process_noise) 
            m_model = measurement_models.FullStateMeasurement(hopper_running_est_model, np.sqrt(plan_dt)*measurement_noise)
            estimation_models += [[hopper_running_est_model]*n_steps]
            umodel = problem_uncertainty.UncertaintyModel(p_model, m_model)
            estimation_uncertainty += [[umodel]*n_steps]


        diff_hopper = pneumatic_hopper.DifferentialActionModelHopper(horizon, horizon, True) 
        estimation_models += [[crocoddyl.IntegratedActionModelEuler(diff_hopper, dt_control) ]]
    return models, uncertainty_models, estimation_models, estimation_uncertainty  




