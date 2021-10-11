import numpy as np 
import os, sys

from scipy.linalg.basic import solve

src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from utils.action_models import point_cliff
from utils.problems import point_cliff_problem
from utils.uncertainty import problem_uncertainty
from solvers import firisc
import crocoddyl 
from config_point_cliff import *


if __name__ == "__main__":
    p_models, u_models, p_estimate, u_estimate = point_cliff_problem.full_state_uniform_cliff_problem(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)



    ddp_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, u_models)
    solver = firisc.FeasibilityRiskSensitiveSolver(ddp_problem,irisc_uncertainty, sensitivity)
    solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(horizon+1)
    us = [np.zeros(2)]*horizon

    cost_try = solver.expected_cost(xs, us)
    
    print("approximate cost= %s"%cost_try)
    print("total cost = %s"%solver.nl_cost)