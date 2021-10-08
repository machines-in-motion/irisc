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
    ddp_xs = [x0]*horizon
    ddp_us = [np.zeros(2)]*(horizon-1)
    ddp_converged = ddp_solver.solve(ddp_xs,ddp_us, MAX_ITER)

    if ddp_converged:
        print("DDP Converged".center(LINE_WIDTH, '#'))
        print("Starting iRiSC".center(LINE_WIDTH, '#'))

    irisc_problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])
    irisc_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, u_models)
    irisc_solver = furisc.FeasibilityRiskSensitiveSolver(irisc_problem, irisc_uncertainty, sensitivity)

    irisc_solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    irisc_xs = [x0]*horizon
    irisc_us = [np.zeros(2)]*(horizon-1)

    # MAX_ITER = 5
    irisc_converged = irisc_solver.solve(irisc_xs, irisc_us, MAX_ITER, False)

    # if irisc_converged:
    #     print("iRiSC Converged".center(LINE_WIDTH, '#'))
    #     print("Plotting Results".center(LINE_WIDTH, '#'))



    # ## ddp gains 
    # # 
    # for t in range(horizon-1):
    #     print("DDP max feedback gain\n",np.amax(np.abs(ddp_solver.K[t])))    
    #     print("iRiSC max feedback gain\n",np.amax(np.abs(irisc_solver.Kfb[t])))    
    plt.figure("trajectory plot")
    plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="ddp")
    plt.plot(np.array(irisc_solver.xs)[:,0],np.array(irisc_solver.xs)[:,1], label="irisc")
    plt.legend()

    # x_sim = [x0]
    # xhat_sim = [x0]
    # cov_sim = [initial_covariance]
    # y_sim = [x0]
    # u_sim = []
    # # acceleration model 
    # dv_model = point_cliff.PointMassDynamics()
    # q_next = np.zeros(2)
    # v_next = np.zeros(2)

    # irisc_kff_norm = []
    # irisc_kfb_norm = []
    # irisk_open_loop = []
    
    # for t in range(horizon-1): 
    #     if t == 0:
    #         u_prev = None 
    #     else:
    #         u_prev = u_sim[-1]
    #     # u_sim += [irisc_solver.controllerStep(t, y_sim[t], u_prev)]
    #     u_sim += [irisc_solver.perfectObservationControl(t, x_sim[t])]
    #     dv = dt*dv_model(x_sim[-1], u_sim[-1]) # get acceleration 

    #     irisc_kff_norm += [np.linalg.norm(irisc_solver.kff[t])]
    #     irisc_kfb_norm += [np.linalg.norm(irisc_solver.Kfb[t])]
    #     irisk_open_loop += [np.linalg.norm(irisc_solver.us[t])]

    #     v_next[:] = x_sim[-1][2:] + dv 
    #     q_next[:] = x_sim[-1][:2] + dt* x_sim[-1][2:] + .5 * dt*dv 
    #     x_new = np.hstack([q_next, v_next])
    #     x_sim += [irisc_uncertainty.sample_process(t, x_new, u_sim[-1])]
    #     y_sim += [irisc_uncertainty.sample_measurement(t, x_sim[-2], u_sim[-1])]


    # plt.figure("iRiSC trajectory plot")
    # plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="ddp")
    # plt.plot(np.array(irisc_solver.xs)[:,0],np.array(irisc_solver.xs)[:,1], label="irisc")
    # # plt.plot(np.array(x_sim)[:,0],np.array(x_sim)[:,1], label="actual")
    # # plt.plot(np.array(y_sim)[:,0],np.array(y_sim)[:,1], label="measured")
    # plt.legend()



    # x_sim = [x0]
    # xhat_sim = [x0]
    # cov_sim = [initial_covariance]
    # y_sim = [x0]
    # u_sim = []
    # # acceleration model 
    # dv_model = point_cliff.PointMassDynamics()
    # q_next = np.zeros(2)
    # v_next = np.zeros(2)

    # ddp_kff_norm = []
    # ddp_kfb_norm = []
    # ddp_open_loop = []
    
    # for t in range(horizon-1): 
    #     if t == 0:
    #         u_prev = None 
    #     else:
    #         u_prev = u_sim[-1]
    #     # u_sim += [irisc_solver.controllerStep(t, y_sim[t], u_prev)]
    #     err = ddp_solver.problem.runningModels[t].state.diff(ddp_solver.xs[t], x_sim[t])
    #     u_sim += [ddp_solver.us[t]- ddp_solver.K[t].dot(err)]
    #     dv = dt*dv_model(x_sim[-1], u_sim[-1]) # get acceleration 

    #     ddp_kff_norm += [np.linalg.norm(ddp_solver.k[t])]
    #     ddp_kfb_norm += [np.linalg.norm(ddp_solver.K[t])]
    #     ddp_open_loop += [np.linalg.norm(ddp_solver.us[t])]


    #     v_next[:] = x_sim[-1][2:] + dv 
    #     q_next[:] = x_sim[-1][:2] + dt* x_sim[-1][2:] + .5 * dt*dv 
    #     x_new = np.hstack([q_next, v_next])
    #     x_sim += [irisc_uncertainty.sample_process(t, x_new, u_sim[-1])]
    #     y_sim += [irisc_uncertainty.sample_measurement(t, x_sim[-2], u_sim[-1])]


    # # plt.figure("DDP trajectory plot")
    # # plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="ddp")
    # # plt.plot(np.array(x_sim)[:,0],np.array(x_sim)[:,1], label="actual")
    # # plt.plot(np.array(y_sim)[:,0],np.array(y_sim)[:,1], label="measured")
    # # plt.legend()


    # t_array = dt*np.arange(horizon-1)
    # plt.figure("feedforward norms")
    # plt.plot(t_array, ddp_kff_norm, label="ddp kff")
    # plt.plot(t_array, irisc_kff_norm, label="irisc kff")
    # plt.legend()


    # plt.figure("feedback norms")
    # plt.plot(t_array, ddp_kfb_norm, label="ddp Kfb")
    # plt.plot(t_array, irisc_kfb_norm, label="irisc Kfb")
    # plt.legend()

    # plt.figure("open loop norms")
    # plt.plot(t_array, ddp_open_loop, label="ddp us")
    # plt.plot(t_array, irisk_open_loop, label="irisc us")
    # plt.legend()

    plt.show()