import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 


import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 

from solvers import firisc
from utils import problem_uncertainty
import pneumatic_hopper_problem 


LINE_WIDTH = 100 
PLOT_FIGS = True 
SAVE_SOLN = True 

x0 = np.array([0.5, 0., 0., 0.])
MAX_ITER = 1000
horizon = 300


plan_dt = 1.e-2
control_dt = 1.e-3
sim_dt = 1.e-5

initial_covariance = 1.e-4 * np.eye(4)
process_noise = 1.e-4*np.eye(4)
measurement_noise = 1.e-4*np.eye(4)
sensitivity = -.5

if __name__ == "__main__":
    # load ddp solution 

    p_models, u_models, p_estimate, u_estimate = pneumatic_hopper_problem.full_state_uniform_hopper(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)

    problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])

    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, u_models)
    solver = firisc.FeasibilityRiskSensitiveSolver(problem, irisc_uncertainty, sensitivity)

    solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])

    xs = [x0]*(horizon+1)
    us = [np.array([0.])]*horizon    
    # 
    converged = solver.solve(xs,us, MAX_ITER, False)
    if not converged:
        print(" iRiSC Solver Did Not Converge ".center(LINE_WIDTH, '!'))
    else:
        print(" iRiSC Solver Converged ".center(LINE_WIDTH, '='))

    #
    logger = solver.getCallbacks()[0] 
    # 
    if sensitivity < 0.:
        save_path = "solutions/risk_averse"
    else:
        save_path = "solutions/risk_seeking"
        
    if SAVE_SOLN:
        print(" Saving iRiSC Solution ".center(LINE_WIDTH, '-'))
        np.save(save_path+"/iRiSC_xs", np.array(solver.xs))
        np.save(save_path+"/iRiSC_us", np.array(solver.us))
        np.save(save_path+"/iRiSC_K", np.array(solver.K))  
        np.save(save_path+"/iRiSC_V", np.array(solver.V))
        np.save(save_path+"/iRiSC_v", np.array(solver.v))    
        np.save(save_path+"/iRiSC_costs", np.array(logger.costs))
        np.save(save_path+"/iRiSC_stepLengths", np.array(logger.steps))
        np.save(save_path+"/iRiSC_gaps", np.array(logger.fs))
        np.save(save_path+"/iRiSC_grads", np.array(logger.grads))
        np.save(save_path+"/iRiSC_stops", np.array(logger.stops))
        np.save(save_path+"/iRiSC_uRegs", np.array(logger.u_regs))
        np.save(save_path+"/iRiSC_xRegs", np.array(logger.x_regs))
    #
    if PLOT_FIGS:
        print(" Plotting iRiSC Solution ".center(LINE_WIDTH, '-'))
        time_array = plan_dt*np.arange(horizon+1)
        #
        foot_planned = np.array(solver.xs)[:,0] - np.array(solver.xs)[:,1] - .5*np.ones_like(time_array)
        plt.figure("trajectory plot")
        plt.plot(time_array,np.array(solver.xs)[:,0], label="Mass Height")
        plt.plot(time_array,foot_planned, label="Foot height")
        # plt.plot(time_array,np.array(solver.xs)[:,1], label="Piston Height")
        #
        plt.figure("control inputs")
        plt.plot(time_array[:-1],np.array(solver.us)[:], label="control inputs")
        # 
        plt.figure("feedback gains")
        for i in range(4):
            plt.plot(time_array[:-1],np.array(solver.K)[:,0,i], label="$K_%s$"%i)
        plt.legend()
        #
        plt.show()





