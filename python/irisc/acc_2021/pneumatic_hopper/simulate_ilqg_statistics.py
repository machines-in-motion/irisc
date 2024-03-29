""" simulates iLQR solution with EKF solutions N times while sampling random disturbances """

import numpy as np 
import os, sys

from numpy.core.arrayprint import set_string_function

src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from utils.action_models import pneumatic_hopper
from utils.problems import pneumatic_hopper_problem
from utils.uncertainty import estimators
from utils.simulation import controllers, simulator

import matplotlib.pyplot as plt 
from config_pneumatic_hopper import *

MAX_ITER = 1000 
LINE_WIDTH = 100

SAVE_SOLN = True 
PLOT_SOLN = True 

N_SIMULATIONS = 1000
if __name__ == "__main__":
    sensitivity = -.5

    if sensitivity < 0.:
        solution_path = "solutions/ddp" 
    else:
        solution_path = "solutions/ddp"

    xs = np.load(solution_path+'_xs.npy')
    us = np.load(solution_path+'_us.npy')
    feedback = np.load(solution_path+'_K.npy')
    # V = np.load(solution_path+'_V.npy')
    # v = np.load(solution_path+'_v.npy')

    p_models, u_models, p_estimate, u_estimate = pneumatic_hopper_problem.full_state_uniform_hopper(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)

    n_steps = int(plan_dt/control_dt)
    controller = controllers.DDPController(p_models, xs, us, feedback, control_dt)
    estimator = estimators.ExtendedKalmanFilter(x0, initial_covariance, p_estimate, u_estimate, n_steps)


    hopper_dynamics = pneumatic_hopper.PneumaticHopper1D()
    sim = simulator.HopperSimulator(hopper_dynamics, controller, estimator, x0, horizon, plan_dt, control_dt, sim_dt)

    trajectory_actual = []
    trajectory_estimated = []
    feedforward = []
    feedback = []
    forces = []

    for i in range(N_SIMULATIONS):
        sim.reset()
        sim.simulate()
        trajectory_actual += [sim.xsim]
        trajectory_estimated += [sim.xhsim]
        feedforward += [sim.usim]
        feedback += [sim.controller.K]
        forces += [sim.fsim]

    save_path = "results/ddp"

    np.save(save_path+"/ddp_xsim", np.array(trajectory_actual))
    np.save(save_path+"/ddp_xhsim", np.array(trajectory_estimated))
    np.save(save_path+"/ddp_usim", np.array(feedforward))
    np.save(save_path+"/ddp_Ksim", np.array(feedback))
    np.save(save_path+"/ddp_fsim", np.array(forces))