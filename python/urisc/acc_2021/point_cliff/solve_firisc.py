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
import matplotlib.pyplot as plt 


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

    irisc_converged = solver.solve(xs, us, MAX_ITER, False)

    #
    if sensitivity < 0.:
        save_path = "solutions/risk_averse"
    else:
        save_path = "solutions/risk_seeking"

    logger = solver.getCallbacks()[0]  
    if SAVE_SOLN:
        print(" Saving FDDP Solution ".center(LINE_WIDTH, '-'))
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
    if PLOT_SOLN:
        print(" Plotting FDDP Solution ".center(LINE_WIDTH, '-'))
        time_array = plan_dt*np.arange(horizon+1)
        #
        plt.figure("trajectory plot")
        plt.plot(np.array(solver.xs)[:,0],np.array(solver.xs)[:,1], label="irisc")
        plt.legend()
        #
        plt.figure("control inputs")
        for i in range(2):
            plt.plot(time_array[:-1],np.array(solver.us)[:,i], label="control inputs")
        #
        plt.show()