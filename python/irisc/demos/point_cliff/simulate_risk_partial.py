""" First Simulation of entire risk sensitive loop """

""" runs a point mass simulation with ekf estimation """

import numpy as np 
import os, sys

from numpy.core.arrayprint import set_string_function

src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from utils.action_models import point_cliff
from utils.problems import point_cliff_problem
from utils.uncertainty import problem_uncertainty


from utils.uncertainty import measurement_models, process_models, problem_uncertainty, estimators
from utils.simulation import controllers, simulator

import matplotlib.pyplot as plt 
from config_point_cliff import *

MAX_ITER = 1000 
LINE_WIDTH = 100

SAVE_SOLN = True 
PLOT_SOLN = True 
if __name__ == "__main__":
    solution_path = "solutions/risk_seeking/iRiSC"

    xs = np.load(solution_path+'_xs.npy')
    us = np.load(solution_path+'_us.npy')
    feedback = np.load(solution_path+'_K.npy')
    V = np.load(solution_path+'_V.npy')
    v = np.load(solution_path+'_v.npy')

    p_models, u_models, p_estimate, u_estimate = point_cliff_problem.full_state_uniform_cliff_problem(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)

    n_steps = int(plan_dt/control_dt)
    controller = controllers.RiskSensitiveController(p_models, xs, us, feedback, V, v, sensitivity, control_dt)
    estimator = estimators.RiskSensitiveFilter(x0, initial_covariance, p_estimate, u_estimate, n_steps, xs, us, sensitivity)


    point_cliff_dynamics = point_cliff.PointMassDynamics()
    sim = simulator.PointMassSimulator(point_cliff_dynamics, controller, estimator, x0, horizon, plan_dt, control_dt, sim_dt)

    sim.simulate()
    
    trajectory = np.array(sim.xsim)
    hat_traj =  np.array(sim.xhsim)


    plt.figure("Trajectory")
    plt.plot(trajectory[:,0], trajectory[:,1], label="Mass Actual")
    plt.plot(hat_traj[:,0], hat_traj[:,1], label="Mass Estimated")
    plt.legend()
    plt.show()