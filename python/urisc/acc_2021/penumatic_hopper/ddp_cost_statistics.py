""" load ddp results, cacl cost from problem """

""" runs ddp solver for hopper jumping and stores the solution """


import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 


import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 

from utils.action_models import penumatic_hopper 
from solvers import firisc
from utils.uncertainty import measurement_models, process_models, problem_uncertainty
from utils.problems import penumatic_hopper_problem 
from config_penumatic_hopper import *


N_SIMULATIONS = 1000

if __name__ == "__main__":
    # load ddp solution 

    p_models, u_models, p_estimate, u_estimate = penumatic_hopper_problem.full_state_uniform_hopper(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)


    problem = crocoddyl.ShootingProblem(x0, p_models[:-1], p_models[-1])

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