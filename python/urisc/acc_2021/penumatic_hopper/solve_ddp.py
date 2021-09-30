""" runs ddp solver for hopper jumping and stores the solution """


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 

import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 

from utils.action_models import penumatic_hopper 
from hopper_config import *

if __name__ == "__main__":
    print(" Running FDDP for Penumatic Hopper ".center(LINE_WIDTH, '#'))
    # 
    running_models = []
    for t in range(horizon): 
        diff_hopper = penumatic_hopper.DifferentialActionModelHopper(t, horizon, False) 
        running_models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
    diff_hopper = penumatic_hopper.DifferentialActionModelHopper(horizon, horizon, True) 
    terminal_model = crocoddyl.IntegratedActionModelEuler(diff_hopper, dt) 
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

    solver = crocoddyl.SolverFDDP(problem)
    solver.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(horizon+1)
    us = [np.array([0.])]*horizon
    # 
    converged = solver.solve(xs,us, MAX_ITER)
    if not converged:
        print(" FDDP Solver Did Not Converge ".center(LINE_WIDTH, '!'))
    else:
        print(" FDDP Solver Converged ".center(LINE_WIDTH, '='))

    #
    if SAVE_SOLN:
        print(" Saving FDDP Solution ".center(LINE_WIDTH, '-'))
        np.save("solutions/ddp_xs", np.array(solver.xs))
        np.save("solutions/ddp_us", np.array(solver.us))
        np.save("solutions/ddp_K", np.array(solver.K))  
        logger = solver.getCallbacks()[0] 
        np.save("solutions/ddp_costs", np.array(logger.costs))
        np.save("solutions/ddp_stepLengths", np.array(logger.steps))
        np.save("solutions/ddp_gaps", np.array(logger.fs))
        np.save("solutions/ddp_grads", np.array(logger.grads))
        np.save("solutions/ddp_stops", np.array(logger.stops))
        np.save("solutions/ddp_uRegs", np.array(logger.u_regs))
        np.save("solutions/ddp_xRegs", np.array(logger.x_regs))
    #
    if PLOT_FIGS:
        print(" Plotting FDDP Solution ".center(LINE_WIDTH, '-'))
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



    
            
 
