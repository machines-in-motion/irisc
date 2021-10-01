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

from hopper_config import *

if __name__ == "__main__":
    print(" Running uRiSC for Penumatic Hopper ".center(LINE_WIDTH, '#'))
    # 
    running_models = []
    for t in range(horizon): 
        diff_hopper = penumatic_hopper.DifferentialActionModelHopper(t, horizon, False) 
        running_models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
    diff_hopper = penumatic_hopper.DifferentialActionModelHopper(horizon, horizon, True) 
    terminal_model = crocoddyl.IntegratedActionModelEuler(diff_hopper, dt) 
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

    # uncertainty models 
    initial_covariance = 1.e-3 * np.eye(4)
    process_noise = 1.e-7*np.eye(4)
    measurement_noise = 1.e-5*np.eye(4)
    sensitivity = -.1
    uncertainty_models = []
    for i, m in enumerate(running_models+[terminal_model]):
        # loop only over running models 
        p_model = process_models.FullStateProcess(m, process_noise) 
        m_model = measurement_models.FullStateMeasurement(m, measurement_noise)
        uncertainty_models += [problem_uncertainty.UncertaintyModel(p_model, m_model)]


    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, uncertainty_models)
    solver = furisc.RiskSensitiveSolver(problem, irisc_uncertainty, sensitivity)

    solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])

    xs_ddp = np.load("solutions/ddp_xs.npy")
    us_ddp = np.load("solutions/ddp_us.npy")
    xs = [xs_ddp[i].copy() for i in range(horizon+1)]
    us = [us_ddp[i].copy() for i in range(horizon)]    
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
        plt.plot(time_array[:-1],np.array(solver.us)[:,0], label="control inputs")
        # 
        plt.figure("feedback gains")
        for i in range(4):
            plt.plot(time_array[:-1],np.array(solver.K)[:,i], label="$K_%s$"%i)
        plt.legend()
        #
        plt.show()



    
            
 