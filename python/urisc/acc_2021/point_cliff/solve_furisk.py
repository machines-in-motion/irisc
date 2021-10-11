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

SAVE_SOLN = True 
PLOT_SOLN = True 
if __name__ == "__main__":
    
    dt = 0.01 
    horizon = 100 
    x0 = np.zeros(4)
    initial_covariance = 1.e-3 * np.eye(4)
    process_noise = 1.e-3*np.eye(4)
    # process_noise[1,1] = 1.e-2 
    measurement_noise = 1.e-3*np.eye(4)
    sensitivity =  -.2
    p_models, u_models = point_cliff_problem.full_state_uniform_cliff_problem(dt, horizon, process_noise, measurement_noise)

    ddp_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    ddp_solver = crocoddyl.SolverDDP(ddp_problem)
    ddp_solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    ddp_xs = [x0]*(horizon+1)     
    ddp_us = [np.zeros(2)]*horizon 
    ddp_converged = ddp_solver.solve(ddp_xs,ddp_us, MAX_ITER)

    if ddp_converged:
        print("DDP Converged".center(LINE_WIDTH, '#'))
        print("Starting iRiSC".center(LINE_WIDTH, '#'))

    irisc_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, u_models)
    solver = furisc.FeasibilityRiskSensitiveSolver(irisc_problem, irisc_uncertainty, sensitivity)

    solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    irisc_xs = [x0]*(horizon+1)     
    irisc_us = [np.zeros(2)]*horizon

    irisc_converged = solver.solve(irisc_xs, irisc_us, MAX_ITER, False)


    if SAVE_SOLN:
        print(" Saving FDDP Solution ".center(LINE_WIDTH, '-'))
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
    if PLOT_SOLN:
        print(" Plotting FDDP Solution ".center(LINE_WIDTH, '-'))
        time_array = dt*np.arange(horizon+1)
        #
        plt.figure("trajectory plot")
        plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="ddp")
        plt.plot(np.array(solver.xs)[:,0],np.array(solver.xs)[:,1], label="irisc")
        plt.legend()
        #
        plt.figure("control inputs")
        for i in range(2):
            plt.plot(time_array[:-1],np.array(solver.us)[:,i], label="control inputs")
        #
        plt.show()