""" load ddp results, cacl cost from problem """

""" runs ddp solver for hopper jumping and stores the solution """


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 


import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 
from utils.problems import pneumatic_hopper_problem 
from config_pneumatic_hopper import *


N_SIMULATIONS = 1000

if __name__ == "__main__":
    # load ddp solution 

    p_models, u_models, p_estimate, u_estimate = pneumatic_hopper_problem.full_state_uniform_hopper(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)


    problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])

    # DDP Cost Statistics
    ddp_xsim = np.load("results2/ddp/ddp_xsim.npy")
    ddp_xhsim= np.load("results2/ddp/ddp_xhsim.npy") 
    ddp_usim= np.load("results2/ddp/ddp_usim.npy")
    ddp_Ksim = np.load("results2/ddp/ddp_Ksim.npy") 
    ddp_fsim = np.load("results2/ddp/ddp_fsim.npy") 
    costs = []
    for i in range(N_SIMULATIONS):
        cost_t = []
        for t in range(horizon):
            problem.runningModels[t].calc(problem.runningDatas[t], ddp_xsim[i,t,:], ddp_usim[i,t,:])
            if t == 0:
                cost_t += [problem.runningDatas[t].cost]
            else:
                cost_t += [cost_t[-1]+problem.runningDatas[t].cost]
        
        problem.terminalModel.calc(problem.terminalData, ddp_xsim[i,horizon,:])
        cost_t += [cost_t[-1]+problem.terminalData.cost]

        costs += [cost_t]

    np.save("results2/ddp_cost_stats", np.array(costs))

    cost_stats = np.load("results2/ddp_cost_stats.npy") 
    print(cost_stats.shape)

    # Risk Averse Cost Statistics 

    averse_xsim = np.load("results2/risk_averse/averse_xsim.npy")
    averse_xhsim= np.load("results2/risk_averse/averse_xhsim.npy") 
    averse_usim= np.load("results2/risk_averse/averse_usim.npy")
    averse_Ksim = np.load("results2/risk_averse/averse_Ksim.npy") 
    averse_fsim = np.load("results2/risk_averse/averse_fsim.npy") 
    costs = []
    for i in range(N_SIMULATIONS):
        cost_t = []
        for t in range(horizon):
            problem.runningModels[t].calc(problem.runningDatas[t], averse_xsim[i,t,:], averse_usim[i,t,:])
            if t == 0:
                cost_t += [problem.runningDatas[t].cost]
            else:
                cost_t += [cost_t[-1]+problem.runningDatas[t].cost]
        
        problem.terminalModel.calc(problem.terminalData, averse_xsim[i,horizon,:])
        cost_t += [cost_t[-1]+problem.terminalData.cost]

        costs += [cost_t]

    np.save("results2/averse_cost_stats", np.array(costs))

    cost_stats = np.load("results2/averse_cost_stats.npy") 
    print(cost_stats.shape)


    # Risk Seeking Cost Statistics 

    seeking_xsim = np.load("results2/risk_seeking/seeking_xsim.npy")
    seeking_xhsim= np.load("results2/risk_seeking/seeking_xhsim.npy") 
    seeking_usim= np.load("results2/risk_seeking/seeking_usim.npy")
    seeking_Ksim = np.load("results2/risk_seeking/seeking_Ksim.npy") 
    seeking_fsim = np.load("results2/risk_seeking/seeking_fsim.npy") 
    costs = []
    for i in range(N_SIMULATIONS):
        cost_t = []
        for t in range(horizon):
            problem.runningModels[t].calc(problem.runningDatas[t], seeking_xsim[i,t,:], seeking_usim[i,t,:])
            if t == 0:
                cost_t += [problem.runningDatas[t].cost]
            else:
                cost_t += [cost_t[-1]+problem.runningDatas[t].cost]
        
        problem.terminalModel.calc(problem.terminalData, seeking_xsim[i,horizon,:])
        cost_t += [cost_t[-1]+problem.terminalData.cost]

        costs += [cost_t]

    np.save("results2/seeking_cost_stats", np.array(costs))

    cost_stats = np.load("results2/seeking_cost_stats.npy") 
    print(cost_stats.shape)