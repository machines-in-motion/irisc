""" First Simulation of entire risk sensitive loop """

""" runs a point mass simulation with ekf estimation """

import numpy as np 
import os, sys

from numpy.core.arrayprint import set_string_function

src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from utils.action_models import penumatic_hopper
from utils.problems import penumatic_hopper_problem
from utils.uncertainty import problem_uncertainty


from utils.uncertainty import measurement_models, process_models, problem_uncertainty, estimators
from utils.simulation import controllers, simulator

import matplotlib.pyplot as plt 
from config_penumatic_hopper import *

MAX_ITER = 1000 
LINE_WIDTH = 100

SAVE_SOLN = True 
PLOT_SOLN = True 
if __name__ == "__main__":
    if sensitivity < 0.:
        solution_path = "solutions/risk_averse/iRiSC" 
    else:
        solution_path = "solutions/risk_seeking/iRiSC"

    xs = np.load(solution_path+'_xs.npy')
    us = np.load(solution_path+'_us.npy')
    feedback = np.load(solution_path+'_K.npy')
    V = np.load(solution_path+'_V.npy')
    v = np.load(solution_path+'_v.npy')

    p_models, u_models, p_estimate, u_estimate = penumatic_hopper_problem.full_state_uniform_hopper(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)

    n_steps = int(plan_dt/control_dt)
    controller = controllers.RiskSensitiveController(p_models, xs, us, feedback, V, v, sensitivity, control_dt)
    estimator = estimators.RiskSensitiveFilter(x0, initial_covariance, p_estimate, u_estimate, n_steps, xs, us, sensitivity)


    point_cliff_dynamics = penumatic_hopper.PenumaticHopped1D()
    sim = simulator.HopperSimulator(point_cliff_dynamics, controller, estimator, x0, horizon, plan_dt, control_dt, sim_dt)

    sim.simulate()
    
    trajectory = np.array(sim.xsim)
    hat_traj =  np.array(sim.xhsim)


    time_array = plan_dt*np.arange(horizon+1)

    contact_pt_height = trajectory[:,0] - trajectory[:,1] - .5*np.ones_like(time_array)
    estimated_contact_pt = hat_traj[:,0] - hat_traj[:,1] - .5*np.ones_like(time_array)
    foot_planned = xs[:,0] - xs[:,1] - .5*np.ones_like(time_array)
    ground_hieght = sim.env*np.ones_like(time_array)
    plt.figure("trajectory plot")
    plt.plot(time_array,trajectory[:,0], label="Mass Actual")
    plt.plot(time_array,xs[:,0], '--', label="Mass Planned")
    plt.plot(time_array,hat_traj[:,0], label="Mass Estimate")
    plt.plot(time_array,contact_pt_height, label="Foot Actual")
    plt.plot(time_array,estimated_contact_pt, label="Foot Estimate")
    plt.plot(time_array, foot_planned, '--', label="Foot Planned")
    plt.plot(time_array,ground_hieght, '--k',linewidth=2.)
    plt.legend()

    norms = []
    for covi in sim.chi: 
        norms += [np.linalg.norm(covi)]
    plt.figure("State Covariance Norm")
    plt.plot(time_array,norms, label="$\chi$")

    force_time_array = 1.e-3*np.arange(10*horizon)
    plt.figure("contact forces")
    plt.plot(force_time_array, sim.fsim, label="$F^n$")
    plt.show()
