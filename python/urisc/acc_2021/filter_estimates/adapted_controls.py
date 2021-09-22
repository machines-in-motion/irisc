""" what does the filter bias without innovations look like ?
1. import a problem setup
2. run ddp to get some nominal state and control trajectories 
3. setup the risk solver itself 
4. in risk solver  setCandidate(ddp.xs, ddp.us)
5. in risk solver  calc()
6. how is risk.xhat different from risk.xs ? 
 """

import numpy as np 
import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from utils.action_models import point_cliff
from utils.problems import point_cliff_problem
from utils.uncertainty import problem_uncertainty
from solvers import irisc
import crocoddyl 

import matplotlib.pyplot as plt 

MAX_ITER = 1000 
LINE_WIDTH = 100



if __name__ == "__main__":
    
    dt = 0.01 
    horizon = 300 
    x0 = np.zeros(4)
    initial_covariance = 1.e-3 * np.eye(4)
    process_noise = 1.e-3*np.eye(4)
    # process_noise[1,1] = 1.e-2 
    measurement_noise = 1.e-5*np.eye(4)
    # process_noise[1,1] = 1.e-3
    sensitivity = 1. 
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
    irisc_solver = irisc.RiskSensitiveSolver(irisc_problem, irisc_uncertainty, sensitivity)

    irisc_solver.setCallbacks([crocoddyl.CallbackLogger(), crocoddyl.CallbackVerbose()])

    irisc_xs = [x0]*horizon
    irisc_us = [np.array([0., 9.81]) for _ in ddp_solver.us]
    # irisc_us = [ui for ui in ddp_solver.us] 
    

    # MAX_ITER = 1 
    irisc_converged = irisc_solver.solve(irisc_xs, irisc_us, MAX_ITER, False)

    # if irisc_converged:
    #     print("iRiSC Converged".center(LINE_WIDTH, '#'))
    #     print("Plotting Results".center(LINE_WIDTH, '#'))



    # ## ddp gains 
    # # 
    # for t in range(horizon-1):
    #     print("DDP max feedback gain\n",np.amax(np.abs(ddp_solver.K[t])))    
    #     print("iRiSC max feedback gain\n",np.amax(np.abs(irisc_solver.Kfb[t])))    
    # plt.figure("trajectory plot")
    # plt.plot(np.array(ddp_solver.xs)[:,0],np.array(ddp_solver.xs)[:,1], label="ddp")
    # plt.plot(np.array(irisc_solver.xs)[:,0],np.array(irisc_solver.xs)[:,1], label="irisc")
    # plt.legend()

    x_sim = [x0]
    xhat_sim = [x0]
    xcheck_sim = [x0]
    cov_sim = [initial_covariance]
    y_sim = [x0]
    u_sim = [irisc_solver.us[0]]
    # acceleration model 
    dv_model = point_cliff.PointMassDynamics()
    q_next = np.zeros(2)
    v_next = np.zeros(2)

    irisc_kff_norm = []
    irisc_kfb_norm = []

    

    for t in range(horizon-1): 


        # noiseless simulation step 
        dv = dt*dv_model(x_sim[t], u_sim[t]) # get acceleration 
        v_next[:] = x_sim[t][2:] + dv 
        q_next[:] = x_sim[t][:2] + dt* x_sim[t][2:] + .5 * dt*dv 
        x_new = np.hstack([q_next, v_next])
        # add disturbance to measurement 
        y_sim += [irisc_uncertainty.sample_measurement(t, x_sim[t], u_sim[t])]
        # add disturbance to simulation 
        x_sim += [irisc_uncertainty.sample_process(t, x_new, u_sim[t])]
        # compute the estimate 
        newX, newP = irisc_solver.pastStressEstimate(t, xcheck_sim[t], u_sim[t], y_sim[t+1], xhat_sim[t], cov_sim[t])

        xhat_sim += [newX]
        cov_sim += [newP]

        xcheck, ucheck = irisc_solver.controllerStep(t+1, xhat_sim[t+1], cov_sim[t+1])
        xcheck_sim += [xcheck]
        # compute control for next step 
        u_sim += [ucheck]

        irisc_kff_norm += [np.linalg.norm(irisc_solver.k[t])]
        irisc_kfb_norm += [np.linalg.norm(irisc_solver.K[t])]


    plt.figure("iRiSC trajectory plot")
    plt.plot(np.array(irisc_solver.xs)[:,0],np.array(irisc_solver.xs)[:,1], label="irisc")
    plt.plot(np.array(x_sim)[:,0],np.array(x_sim)[:,1], label="actual")
    plt.plot(np.array(y_sim)[:,0],np.array(y_sim)[:,1], label="measured")
    plt.plot(np.array(xhat_sim)[:,0],np.array(xhat_sim)[:,1], label="estimated")
    plt.plot(np.array(xcheck_sim)[:,0],np.array(xcheck_sim)[:,1], label="check")
    plt.legend()


    x_sim = [x0]
    xhat_sim = [x0]
    
    y_sim = [x0]
    u_sim = []
    # acceleration model 
    dv_model = point_cliff.PointMassDynamics()
    q_next = np.zeros(2)
    v_next = np.zeros(2)

    ddp_kff_norm = []
    ddp_kfb_norm = []
    ddp_open_loop = []
    
    for t in range(horizon-1): 
        if t == 0:
            u_prev = None 
        else:
            u_prev = u_sim[-1]
        # u_sim += [irisc_solver.controllerStep(t, y_sim[t], u_prev)]
        err = ddp_solver.problem.runningModels[t].state.diff(ddp_solver.xs[t], x_sim[t])
        u_sim += [ddp_solver.us[t]- ddp_solver.K[t].dot(err)]
        dv = dt*dv_model(x_sim[-1], u_sim[-1]) # get acceleration 

        ddp_kff_norm += [np.linalg.norm(ddp_solver.us[t]-ddp_solver.k[t])]
        ddp_kfb_norm += [np.linalg.norm(ddp_solver.K[t])]
        # ddp_open_loop += [np.linalg.norm(ddp_solver.us[t]-ddp_solver.k[t])]


        v_next[:] = x_sim[-1][2:] + dv 
        q_next[:] = x_sim[-1][:2] + dt* x_sim[-1][2:] + .5 * dt*dv 
        x_new = np.hstack([q_next, v_next])
        x_sim += [irisc_uncertainty.sample_process(t, x_new, u_sim[-1])]
        y_sim += [irisc_uncertainty.sample_measurement(t, x_sim[-2], u_sim[-1])]

    t_array = dt*np.arange(horizon-1)
    # plt.figure("feedforward norms")
    # plt.plot(t_array, ddp_kff_norm, label="ddp kff")
    # plt.plot(t_array, irisc_kff_norm, label="irisc kff")
    # plt.legend()


    # plt.figure("feedback norms")
    # plt.plot(t_array, ddp_kfb_norm, label="ddp kff")
    # plt.plot(t_array, irisc_kfb_norm, label="irisc kff")
    # plt.legend()

    solvers = [ddp_solver, irisc_solver]
    solver_names=["ddp", "irisc"]
    plt.rc('legend', fontsize=30)    # legend fontsize
    fig, ax = plt.subplots(2, 4,figsize=(30,15))

    for i in range(2):
        for j in range(4): 
            for k, solver in enumerate(solvers): 
                if "ddp" in solver_names[k]:
                    ax[i,j].plot(t_array, np.array(solver.K)[:,i,j], 'k', linewidth=5., label=solver_names[k])
                else:
                    ax[i,j].plot(t_array, np.array(solver.K)[:,i,j],'b', linewidth=4., label=solver_names[k])
                    ax[i,j].plot(t_array, np.array(solver.Kfb)[:,i,j],'g', linewidth=3., label=solver_names[k]+" perfect_obs")
    ax[0,3].legend(loc="center left", bbox_to_anchor=(1., -.1), ncol= 1) 

    # maybe plot the covariance norm ? 
    horizon_array = dt*np.arange(horizon)
    plt.figure("covariance norm")
    plt.plot(horizon_array,[np.linalg.norm(pi) for pi in cov_sim] , label="estimate covariance")
    plt.legend()
    plt.show()