""" here we setup a point cliff problem with uncertainty model """
import numpy as np  
import os, sys
src_path = os.path.abspath('../../') # append library directory without packaging 
sys.path.append(src_path)
from utils.uncertainty import measurement_models, process_models, problem_uncertainty
from utils.action_models import penumatic_hopper 
import crocoddyl 


def full_state_uniform_hopper(dt, horizon, process_noise, measurement_noise): 
    models = []
    for t in range(horizon): 
        diff_hopper = penumatic_hopper.DifferentialActionModelHopper(t, horizon, False) 
        models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
    diff_hopper = penumatic_hopper.DifferentialActionModelHopper(horizon, horizon, True) 
    models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)]
    #
    uncertainty_models = []
    for i, m in enumerate(models[:-1]):
        p_model = process_models.FullStateProcess(m, process_noise) 
        m_model = measurement_models.FullStateMeasurement(m, measurement_noise)
        uncertainty_models += [problem_uncertainty.UncertaintyModel(p_model, m_model)]
    # 
    return models, uncertainty_models  




