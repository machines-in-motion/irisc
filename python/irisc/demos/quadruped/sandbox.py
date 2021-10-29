import numpy as np
import crocoddyl 
from robot_properties_solo.config import Solo12Config 
import os, sys, time 
src_path = os.path.abspath('../../')
sys.path.append(src_path)
from demos.quadruped.utils import control_problem_solo12 
from demos.quadruped.config_solo12 import BalanceConfig
# some plotting stuff 
import matplotlib.pyplot as plt 

LINE_WIDTH = 100 


if __name__ == "__main__": 
    # first load solo12 pinocchio wrapper 
    # contact point names too  
    leg = ["FL", "FR", "HL", "HR"]
    contact_names = []
    for li in leg:
        contact_names +=[li+"_ANKLE"]
    solo12_config = Solo12Config()
    solo12 = solo12_config.pin_robot 
    q0 = solo12_config.initial_configuration 
    v0 = np.zeros(solo12.model.nv)
    x0 = np.hstack([q0, v0])
    gaits = control_problem_solo12.Solo12Gaits(solo12, contact_names)

    optModels, _ = gaits.createBalanceProblem(x0, BalanceConfig.timeStep, BalanceConfig.horizon)

    ddp_problem = crocoddyl.ShootingProblem(x0, optModels[:-1], optModels[-1])

    ddp_solver = crocoddyl.SolverFDDP(ddp_problem)
    ddp_solver.setCallbacks(
            [crocoddyl.CallbackLogger(),
            crocoddyl.CallbackVerbose()])
    xs_init = [x0]*(BalanceConfig.horizon+1)
    us_init = [np.zeros(gaits.actuation.nu)]*BalanceConfig.horizon

    ddp_solver.solve(xs_init, us_init, 1000, False, 0.1)
    print("Solving FDDP Completed".center(LINE_WIDTH,'-'))

    xs = np.array(ddp_solver.xs)
    us = np.array(ddp_solver.us)

    x_axis_time = BalanceConfig.timeStep*np.arange(BalanceConfig.horizon+1)
    plt.figure("base height")
    plt.plot(x_axis_time, xs[:,2])

    plt.figure("input controls")
    for i, name in enumerate(solo12_config.joint_names):
        plt.plot(x_axis_time[:-1], us[:,i], label=name)
    plt.legend()


    plt.show()