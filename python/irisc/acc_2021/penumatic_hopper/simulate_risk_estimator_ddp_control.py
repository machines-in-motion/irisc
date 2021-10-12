
import numpy as np 
import os,sys 

src_path = os.path.abspath('../../') 
sys.path.append(src_path)
from utils.action_models import penumatic_hopper
from utils.problems import penumatic_hopper_problem
from utils.uncertainty import problem_uncertainty


from utils.uncertainty import measurement_models, process_models, problem_uncertainty, estimators
from utils.simulation import controllers, simulator

import matplotlib.pyplot as plt 
from config_penumatic_hopper import *




if __name__ == "__main__":
    # load ddp solution 
    solution_path = "solutions/ddp"
    xs = np.load(solution_path+'_xs.npy')
    us = np.load(solution_path+'_us.npy')
    feedback = np.load(solution_path+'_K.npy')


    p_models, u_models, p_estimate, u_estimate = penumatic_hopper_problem.full_state_uniform_hopper(plan_dt, horizon, 
    process_noise, measurement_noise, control_dt)


    controller = controllers.DDPController(p_models, xs ,us, feedback, control_dt)
    n_steps = int(plan_dt/1.e-3)
    hopper_dynamics = penumatic_hopper.PenumaticHopped1D()
    risk_estimator = estimators.RiskSensitiveFilter(x0, initial_covariance, p_estimate, u_estimate, n_steps, xs, us, sensitivity)

    risk_sim = simulator.HopperSimulator(hopper_dynamics, controller, risk_estimator, x0, horizon, plan_dt, control_dt, sim_dt)


    risk_sim.simulate()


    risk_trajectory = np.array(risk_sim.xsim)
    risk_hat_traj =  np.array(risk_sim.xhsim)

    time_array = plan_dt*np.arange(horizon+1)
    actual_foot = risk_trajectory[:,0] - risk_trajectory[:,1] - .5*np.ones_like(time_array)
    estimated_foot = risk_hat_traj[:,0] - risk_hat_traj[:,1] - .5*np.ones_like(time_array)

    foot_planned = xs[:,0] - xs[:,1] - .5*np.ones_like(time_array)
    ground_hieght = risk_sim.env*np.ones_like(time_array)

    plt.figure("estimated trajectories")
    plt.plot(time_array,risk_trajectory[:,0], label="actual mass")
    plt.plot(time_array,risk_hat_traj[:,0], label="estimated mass")
    plt.plot(time_array,actual_foot, label="actual foot")
    plt.plot(time_array,estimated_foot, label="estimated foot")
    plt.plot(time_array,xs[:,0], '--', label="Mass Planned")
    plt.plot(time_array, foot_planned, '--', label="Foot Planned")
    plt.plot(time_array,ground_hieght, '--k',linewidth=2.)
    plt.legend()
    plt.show()


    
 
    