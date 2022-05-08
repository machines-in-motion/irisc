""" runs ddp solver for hopper jumping and stores the solution """


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 


import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 

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


    solver = crocoddyl.SolverDDP(problem)
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
        np.save("solutions/ddp/ddp_xs", np.array(solver.xs))
        np.save("solutions/ddp/ddp_us", np.array(solver.us))
        np.save("solutions/ddp/ddp_K", np.array(solver.K))  
        logger = solver.getCallbacks()[0] 
        np.save("solutions/ddp/ddp_costs", np.array(logger.costs))
        np.save("solutions/ddp/ddp_stepLengths", np.array(logger.steps))
        np.save("solutions/ddp/ddp_gaps", np.array(logger.fs))
        np.save("solutions/ddp/ddp_grads", np.array(logger.grads))
        np.save("solutions/ddp/ddp_stops", np.array(logger.stops))
        np.save("solutions/ddp/ddp_uRegs", np.array(logger.u_regs))
        np.save("solutions/ddp/ddp_xRegs", np.array(logger.x_regs))
    #
    if PLOT_FIGS:
        print(" Plotting FDDP Solution ".center(LINE_WIDTH, '-'))
        time_array = plan_dt*np.arange(horizon+1)
        #
        foot_planned = np.array(solver.xs)[:,0] - np.array(solver.xs)[:,1] - .5*np.ones_like(time_array)
        plt.figure("trajectory plot")
        plt.plot(time_array,np.array(solver.xs)[:,0], label="Mass Height")
        plt.plot(time_array,foot_planned, label="Foot Height")
        # plt.plot(time_array,np.array(solver.xs)[:,1], label="Piston Height")
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



    
            
 
