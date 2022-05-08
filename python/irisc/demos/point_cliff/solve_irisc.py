""" sets up and solve a ddp problem for the point cliff example """

import numpy as np 
import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from models import point_cliff
from demos.point_cliff import point_cliff_problem
from utils import problem_uncertainty
from solvers import irisc
import crocoddyl 

import matplotlib.pyplot as plt 


plan_dt = 1.e-2 
control_dt = 1.e-3 
sim_dt = 1.e-4

horizon = 100


horizon = 100 
x0 = np.zeros(4)
initial_covariance = 1.e-4 * np.eye(4)
process_noise = 1.e-4*np.eye(4)
# process_noise[1,1] = 1.e-2 
measurement_noise = 1.e-4*np.eye(4)
sensitivity = 2.


MAX_ITER = 100 
PLOT_SOLN = True
SAVE_SOLN = True 
LINE_WIDTH = 100
MAX_ITER = 1000 
LINE_WIDTH = 100

SAVE_SOLN = True 
PLOT_SOLN = True 


if __name__ == "__main__":
    p_models, u_models, p_estimate, u_estimate = point_cliff_problem.full_state_uniform_cliff_problem(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)


    ddp_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, u_models)
    solver = irisc.RiskSensitiveSolver(ddp_problem,irisc_uncertainty, sensitivity)
    solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(horizon+1)
    us = [np.zeros(2)]*horizon

    irisc_converged = solver.solve(xs, us, MAX_ITER, False)

 
