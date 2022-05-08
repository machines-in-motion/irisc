""" First Simulation of entire risk sensitive loop """

""" runs a point mass simulation with ekf estimation """

import numpy as np 
import os, sys

from numpy.core.arrayprint import set_string_function

src_path = os.path.abspath('../../') 
sys.path.append(src_path)
print(src_path)
from models import pneumatic_hopper
import pneumatic_hopper_problem

from utils import estimators
from utils import controllers, simulator

import matplotlib.pyplot as plt 
# from config_pneumatic_hopper import *

MAX_ITER = 1000 
LINE_WIDTH = 100

PLOT_FIGS = True 
SAVE_SOLN = True 

x0 = np.array([0.5, 0., 0., 0.])
MAX_ITER = 1000
horizon = 300

x0[0] += 5.e-3

plan_dt = 1.e-2
control_dt = 1.e-3
sim_dt = 1.e-5

initial_covariance = 1.e-4 * np.eye(4)
process_noise = 1.e-4 * np.eye(4)
measurement_noise = 1.e-4 * np.eye(4)
sensitivity = -.5

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

    p_models, u_models, p_estimate, u_estimate = pneumatic_hopper_problem.full_state_uniform_hopper(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)

    n_steps = int(plan_dt/control_dt)
    controller = controllers.RiskSensitiveController(p_models, xs, us, feedback, V, v, sensitivity, control_dt)
    estimator = estimators.RiskSensitiveFilter(x0, initial_covariance, p_estimate, u_estimate, n_steps, xs, us, sensitivity)


    hopper_dynamics = pneumatic_hopper.PneumaticHopper1D()
    sim = simulator.HopperSimulator(hopper_dynamics, controller, estimator, x0, horizon, plan_dt, control_dt, sim_dt)
    sim.estimator = None 
    sim.simulate()
    
    trajectory = np.array(sim.xsim)
    hat_traj =  np.array(sim.xhsim)


    time_array = plan_dt*np.arange(horizon+1)
    fine_time_array = control_dt*np.arange(trajectory.shape[0])

    contact_pt_height = trajectory[:,0] - trajectory[:,1] - .5*np.ones_like(fine_time_array)
    estimated_contact_pt = hat_traj[:,0] - hat_traj[:,1] - .5*np.ones_like(fine_time_array)
    foot_planned = xs[:,0] - xs[:,1] - .5*np.ones_like(time_array)
    ground_hieght = sim.env*np.ones_like(time_array)
    plt.figure("trajectory plot")
    plt.plot(fine_time_array,trajectory[:,0], linewidth=3., label="Mass Actual")
    plt.plot(time_array,xs[:,0], '--',linewidth=3.,  label="Mass Planned")
    plt.plot(fine_time_array,hat_traj[:,0],linewidth=3.,  label="Mass Estimate")
    plt.plot(fine_time_array,contact_pt_height, label="Foot Actual")
    plt.plot(fine_time_array,estimated_contact_pt, label="Foot Estimate")
    plt.plot(time_array, foot_planned, '--', label="Foot Planned")
    plt.plot(time_array,ground_hieght, '--k',linewidth=2.)
    plt.legend()


    usim = np.array(sim.usim)
    time_us = control_dt*np.arange(usim.shape[0])
    plt.figure("control plot")
    plt.plot(time_us, usim[:,0])

    # norms = []
    # for covi in sim.chi: 
    #     norms += [np.linalg.norm(covi)]
    # if sim.estimator is not None:
    #     plt.figure("State Covariance Norm")
    #     plt.plot(time_array,norms, label="$\chi$")

    force_time_array = control_dt*np.arange(len(sim.fsim))
    plt.figure("contact forces")
    plt.plot(force_time_array, sim.fsim, label="$F^n$")
    plt.show(block=False)
    plt.show()
