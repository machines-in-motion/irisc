""" in this document i test the feasibility risk sensitive solver """

import numpy as np 
import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from utils.action_models import point_cliff
from utils.problems import point_cliff_problem
from utils.uncertainty import problem_uncertainty
from solvers import irisc, furisc
import crocoddyl 

import matplotlib.pyplot as plt 

MAX_ITER = 1000 
LINE_WIDTH = 100



if __name__ == "__main__":
    
    dt = 0.01 
    horizon = 100 
    x0 = np.zeros(4)
    initial_covariance = 1.e-3 * np.eye(4)
    process_noise = 1.e-3*np.eye(4)
    # process_noise[1,1] = 1.e-2 
    measurement_noise = 1.e-3*np.eye(4)
    sensitivity =  .2
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
    irisc_solver = furisc.FeasibilityRiskSensitiveSolver(irisc_problem, irisc_uncertainty, sensitivity)

    irisc_solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    irisc_xs = [x0]*horizon
    irisc_us = [np.zeros(2)]*(horizon-1)

    irisc_converged = irisc_solver.solve(irisc_xs, irisc_us, MAX_ITER, False)

    plt.figure("trajectory plot")
    plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="ddp")
    plt.plot(np.array(irisc_solver.xs)[:,0],np.array(irisc_solver.xs)[:,1], label="irisc")
    plt.legend()
    plt.show()

    # irisc_solver.setCandidate(irisc_xs, irisc_us, False)
    # irisc_solver.cost = irisc_solver.unscentedCost()
    # print("initial cost is %s"%irisc_solver.cost)
    # print("initial trajectory feasibility %s"%irisc_solver.isFeasible) 

    # # some reg shit
    # irisc_solver.n_little_improvement = 0
    # irisc_solver.x_reg = irisc_solver.regMin
    # irisc_solver.u_reg = irisc_solver.regMin

    # try:
    #     irisc_solver.calc()
    #     print("calc completed")
    #     irisc_solver.backwardPass()
    #     print("backward pass  succeeded ")
    # except:
    #     print("compute direcrtion failed")
    
     