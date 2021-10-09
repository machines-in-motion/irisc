""" solves risk sensitive control for hopper and stores the solution """

import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 


import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 

from utils.action_models import penumatic_hopper 
from solvers import furisc
from utils.uncertainty import measurement_models, process_models, problem_uncertainty
from utils.problems import penumatic_hopper_problem 
from hopper_config import *

if __name__ == "__main__":
    print(" Running uRiSC for Penumatic Hopper ".center(LINE_WIDTH, '#'))
    # horizon = 100
    # uncertainty models 
    initial_covariance = 1.e-3 * np.eye(4)
    process_noise = 1.e-4*np.eye(4)
    measurement_noise = 1.e-5*np.eye(4)
    sensitivity = -.1
    # MAX_ITER = 1

    pmodels, umodels = penumatic_hopper_problem.full_state_uniform_hopper(dt, horizon, process_noise, measurement_noise)

    problem = crocoddyl.ShootingProblem(x0, pmodels[:-1], pmodels[-1])

    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, umodels)
    solver = furisc.FeasibilityRiskSensitiveSolver(problem, irisc_uncertainty, sensitivity)

    solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])

    # xs_ddp = np.load("solutions/ddp_xs.npy")
    # us_ddp = np.load("solutions/ddp_us.npy")
    # xs = [xi.copy() for xi in xs_ddp]
    # xs = [ui.copy() for ui in us_ddp]
    xs = [x0]*(horizon+1)
    us = [np.array([0.])]*horizon    
    # 
    converged = solver.solve(xs,us, MAX_ITER, True)
    if not converged:
        print(" uRiSC Solver Did Not Converge ".center(LINE_WIDTH, '!'))
    else:
        print(" uRiSC Solver Converged ".center(LINE_WIDTH, '='))

    #
    if SAVE_SOLN:
        print(" Saving uRiSC Solution ".center(LINE_WIDTH, '-'))
        np.save("solutions/uRiSC_xs", np.array(solver.xs))
        np.save("solutions/uRiSC_us", np.array(solver.us))
        np.save("solutions/uRiSC_K", np.array(solver.K))  
        logger = solver.getCallbacks()[0] 
        np.save("solutions/uRiSC_costs", np.array(logger.costs))
        np.save("solutions/uRiSC_stepLengths", np.array(logger.steps))
        np.save("solutions/uRiSC_gaps", np.array(logger.fs))
        np.save("solutions/uRiSC_grads", np.array(logger.grads))
        np.save("solutions/uRiSC_stops", np.array(logger.stops))
        np.save("solutions/uRiSC_uRegs", np.array(logger.u_regs))
        np.save("solutions/uRiSC_xRegs", np.array(logger.x_regs))
    #
    if PLOT_FIGS:
        print(" Plotting uRiSC Solution ".center(LINE_WIDTH, '-'))
        time_array = dt*np.arange(horizon+1)
        #
        plt.figure("trajectory plot")
        plt.plot(time_array,np.array(solver.xs)[:,0], label="Mass Height")
        plt.plot(time_array,np.array(solver.xs)[:,1], label="Piston Height")
        #
        plt.figure("control inputs")
        plt.plot(time_array[:-1],np.array(solver.us)[:], label="control inputs")
        # 
        # plt.figure("feedback gains")
        # for i in range(4):
        #     plt.plot(time_array[:-1],np.array(solver.K)[:,i], label="$K_%s$"%i)
        # plt.legend()
        #
        plt.show()



    
            
 