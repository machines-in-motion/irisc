""" here we run ekf on the actual hopper simulation """

import numpy as np 
import crocoddyl 
import matplotlib.pyplot as plt 


import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 

from utils.action_models import penumatic_hopper 
from utils.uncertainty import measurement_models, process_models, problem_uncertainty, estimators
from utils.simulation import controllers, simulator

from hopper_config import *

if __name__ == "__main__":
    print(" Running EKF for Penumatic Hopper ".center(LINE_WIDTH, '#'))
    # 
    ddp_soln = solutions_path  + '/ddp'

    xs = np.load(ddp_soln+'_xs.npy')
    us = np.load(ddp_soln+'_us.npy')
    feedback = np.load(ddp_soln+'_K.npy')
    
    # setup crocoddyl models 
    running_models = []
    for t in range(horizon): 
        diff_hopper = penumatic_hopper.DifferentialActionModelHopper(t, horizon, False) 
        running_models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
    diff_hopper = penumatic_hopper.DifferentialActionModelHopper(horizon, horizon, True) 
    terminal_model = crocoddyl.IntegratedActionModelEuler(diff_hopper, dt) 
    problem = crocoddyl.ShootingProblem(x0, running_models, terminal_model)

    # setup the  dynamics & controller 
    hopper_dynamics = penumatic_hopper.PenumaticHopped1D()
    controller = controllers.DDPController(running_models+[terminal_model], xs ,us, feedback)

    # uncertainty models 
    initial_covariance = 1.e-5 * np.eye(4)
    process_noise = 1.e-5*np.eye(4)
    measurement_noise = 1.e-5*np.eye(4)
    uncertainty_models = []
    for i, m in enumerate(running_models+[terminal_model]):
        # loop only over running models 
        p_model = process_models.FullStateProcess(m, process_noise) 
        m_model = measurement_models.FullStateMeasurement(m, measurement_noise)
        uncertainty_models += [problem_uncertainty.UncertaintyModel(p_model, m_model)]

    trajectory_uncertainty = problem_uncertainty.ProblemUncertainty(x0, initial_covariance, uncertainty_models)
    # setup estimator 
    ekf = estimators.ExtendedKalmanFilter(trajectory_uncertainty)
    # setup estimation 
    sim = simulator.HopperSimulator(hopper_dynamics, controller, ekf, 
                                    trajectory_uncertainty, x0, horizon)




    sim.simulate()
    
    trajectory = np.array(sim.xsim)
    hat_traj =  np.array(sim.xhsim)

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

    # estimation covariance norm 

    norms = []
    for covi in sim.chi: 
        norms += [np.linalg.norm(covi)]
    plt.figure("State Covariance Norm")
    plt.plot(time_array,norms, label="$\chi$")
    # fine_time_array = np.arange(0., dt*horizon, 1.e-5)
    # plt.figure("Contact Forces")
    # plt.plot(fine_time_array, sim.fsim)



    plt.show()