""" Test the unscented transform of the cost function """

import numpy as np 
import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from utils.action_models import point_cliff
from utils.problems import point_cliff_problem
from utils.uncertainty import problem_uncertainty
from solvers import irisc
import crocoddyl 

import matplotlib.pyplot as plt 

MAX_ITER = 1000 
LINE_WIDTH = 100



if __name__ == "__main__":
    
    dt = 0.01 
    horizon = 300 
    x0 = np.zeros(4)
    initial_covariance = 1.e-3 * np.eye(4)
    process_noise = 1.e-5*np.eye(4)
    # process_noise[1,1] = 1.e-2 
    measurement_noise = 1.e-3*np.eye(4)
    sensitivity = -.3
    p_models, u_models = point_cliff_problem.full_state_uniform_cliff_problem(dt, horizon, process_noise, measurement_noise)

    ddp_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    ddp_solver = crocoddyl.SolverDDP(ddp_problem)
    ddp_solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    ddp_xs = [x0]*horizon
    ddp_us = [np.zeros(2)]*(horizon-1)
    ddp_converged = ddp_solver.solve(ddp_xs,ddp_us, MAX_ITER)

    if ddp_converged:
        print("DDP Converged".center(LINE_WIDTH, '#'))
        print("Starting iRiSC".center(LINE_WIDTH, '#'))

    irisc_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, u_models)
    irisc_solver = irisc.RiskSensitiveSolver(irisc_problem, irisc_uncertainty, sensitivity)
    print(" iRiSC Samples ".center(LINE_WIDTH, '#'))
    print(irisc_solver.samples)
    print(" Square Root of P ".center(LINE_WIDTH, '#'))
    print(irisc_solver.rootP)