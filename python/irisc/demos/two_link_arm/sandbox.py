import os, sys, time 
src_path = os.path.abspath('../../')
sys.path.append(src_path)

import numpy as np 
import crocoddyl 
from models import two_link_manipulator as arm 
import matplotlib.pyplot as plt 

LINE_WIDTH = 100 

if __name__ == "__main__":
    xdes = np.array([1., 1., 0., 0.])
    print(" Testing Point Mass Cliff with DDP ".center(LINE_WIDTH, '#'))
    arm_diff_running =  arm.DifferentialActionTwoLinkManipulator()
    arm_diff_terminal = arm.DifferentialActionTwoLinkManipulator(xdes, isTerminal=True)
    print(" Constructing differential models completed ".center(LINE_WIDTH, '-'))
    dt = 0.01 
    T = 100 
    x0 = np.zeros(4) 
    MAX_ITER = 1000
    arm_running = crocoddyl.IntegratedActionModelEuler(arm_diff_running, dt) 
    arm_terminal = crocoddyl.IntegratedActionModelEuler(arm_diff_terminal, dt) 
    print(" Constructing integrated models completed ".center(LINE_WIDTH, '-'))

    problem = crocoddyl.ShootingProblem(x0, [arm_running]*T, arm_terminal)
    print(" Constructing shooting problem completed ".center(LINE_WIDTH, '-'))
    
    ddp = crocoddyl.SolverFDDP(problem)
    print(" Constructing DDP solver completed ".center(LINE_WIDTH, '-'))
    ddp.setCallbacks([
    crocoddyl.CallbackLogger(),
    crocoddyl.CallbackVerbose()
    ])
    xs = [x0]*(T+1)
    us = [np.zeros(2)]*T
    converged = ddp.solve(xs,us, MAX_ITER)

    time_array = dt*np.arange(T+1)

    eepos = np.zeros([T+1, 4])
    for i in range(T+1):
        p1, eepos[i,:2] = arm_diff_running.dynamics.position_kinematics(ddp.xs[i])

    # if converged:
    print(" DDP solver has CONVERGED ".center(LINE_WIDTH, '-'))
    plt.figure("trajectory plot")
    plt.plot(eepos[:,0],eepos[:,1], label="ddp")

    plt.figure("control plot")
    plt.plot(time_array[:-1],np.array(ddp.us)[:,0], label="ddp")
    plt.plot(time_array[:-1],np.array(ddp.us)[:,1], label="ddp")

    plt.show()