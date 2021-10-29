import numpy as np
import crocoddyl 
from robot_properties_solo.config import Solo12Config 
import os, sys, time 
src_path = os.path.abspath('../../../')
sys.path.append(src_path)
from demos.quadruped.utils import control_problem_solo12 
from demos.quadruped.config_solo12 import BalanceConfig
# some plotting stuff 
import matplotlib.pyplot as plt 

LINE_WIDTH = 100 
SAVE_SOLN = True 

if __name__ == "__main__": 
    # first load solo12 pinocchio wrapper 
    # contact point names too  
    leg = ["FL", "FR", "HL", "HR"]
    contact_names = []
    for li in leg:
        contact_names +=[li+"_ANKLE"]
    solo12_config = Solo12Config()
    solo12 = solo12_config.pin_robot 

    problemConfig = BalanceConfig()

    gaits = control_problem_solo12.Solo12Gaits(solo12, contact_names)

    optModels, _ = gaits.createBalanceProblem(problemConfig)

    ddp_problem = crocoddyl.ShootingProblem(problemConfig.x0, optModels[:-1], optModels[-1])

    solver = crocoddyl.SolverFDDP(ddp_problem)
    solver.setCallbacks(
            [crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose()])
    xs_init = [problemConfig.x0]*(problemConfig.horizon+1)
    us_init = [np.zeros(gaits.actuation.nu)]*problemConfig.horizon

    solver.solve(xs_init, us_init, 1000, False)
    print("Solving FDDP Completed".center(LINE_WIDTH,'-'))

    xs = np.array(solver.xs)
    us = np.array(solver.us)



    x_axis_time = problemConfig.timeStep*np.arange(problemConfig.horizon+1)



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




    plt.figure("base height")
    plt.plot(x_axis_time, xs[:,2])

    plt.figure("input controls")
    for i, name in enumerate(solo12_config.joint_names):
        plt.plot(x_axis_time[:-1], us[:,i], label=name)
    plt.legend()


    plt.show()