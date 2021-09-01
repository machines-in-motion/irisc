""" what does the filter bias without innovations look like ?
1. import a problem setup
2. run ddp to get some nominal state and control trajectories 
3. setup the risk solver itself 
4. in risk solver  setCandidate(ddp.xs, ddp.us)
5. in risk solver  calc()
6. how is risk.xhat different from risk.xs ? 
 """

import numpy as np 
import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path)

from utils.problems import point_cliff_problem
from utils.uncertainty import problem_uncertainty
from solvers import irisc
import crocoddyl 

import matplotlib.pyplot as plt 

MAX_ITER = 100 
LINE_WIDTH = 100

if __name__ == "__main__":
    
    dt = 0.01 
    horizon = 300 
    x0 = np.zeros(4)
    initial_covariance = 1.e-3 * np.eye(4)
    process_noise = 1.e-5*np.eye(4)
    measurement_noise = 1.e-4*np.eye(4)
    sensitivity = 100.

    p_models, u_models = point_cliff_problem.full_state_uniform_cliff_problem(dt, horizon, process_noise, measurement_noise)

    ddp_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    ddp_solver = crocoddyl.SolverDDP(ddp_problem)
    ddp_solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    ddp_xs = [x0]*horizon
    ddp_us = [np.zeros(2)]*(horizon-1)
    converged = ddp_solver.solve(ddp_xs,ddp_us, MAX_ITER)

    if converged:
        print("DDP Converged".center(LINE_WIDTH, '#'))
        print("Starting iRiSC".center(LINE_WIDTH, '#'))

    irisc_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, u_models)
    irisc_solver = irisc.RiskSensitiveSolver(irisc_problem, irisc_uncertainty, sensitivity)

    irisc_solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])


    irisc_solver.setCandidate(ddp_solver.xs, ddp_solver.us, True)
    print(" iRiSC setCandidates works ".center(LINE_WIDTH, '-'))
    irisc_solver.calc()
    print(" iRiSC calc including filterPass works ".center(LINE_WIDTH, '-'))

    plt.figure("trajectory plot")
    plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="ddp")
    plt.plot(np.array(irisc_solver.xhat)[:,0],np.array(irisc_solver.xhat)[:,1], label="irisc")
    plt.legend()

    plt.show()