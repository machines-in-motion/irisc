import os, sys, time 
from os import path
import numpy as np
import pybullet
import pinocchio
from copy import deepcopy
from pathlib import Path
src_path = os.path.abspath('../../../')
sys.path.append(src_path)

from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from utils.simulation import controllers 
from demos.quadruped.utils import control_problem_solo12 
import config as problemConfig 
import matplotlib.pyplot as plt 

from mim_estimation_cpp import RobotStateEstimator, RobotStateEstimatorSettings
from mim_data_utils import DataLogger, DataReader


EXPERIMENT_NAME = "ddp_with_ekf_bullet_sim"
abs_path = os.path.abspath("../log_files")


# create contact names for solo 
leg = ["FL", "FR", "HL", "HR"]
contact_names = []
for li in leg:
    contact_names +=[li+"_ANKLE"]

 



if __name__ == "__main__":

    # create bullet environment 
    np.set_printoptions(precision=2, suppress=True)
    env = BulletEnvWithGround(pybullet.GUI)
    # add the robot 
    robot = Solo12Robot()
    env.add_robot(robot)
    robot_config = Solo12Config() 

    # load ddp controller 
    solution_path = "solutions/ddp"
    xs = np.load(solution_path+'_xs.npy')
    us = np.load(solution_path+'_us.npy')
    feedback = np.load(solution_path+'_K.npy')
    pin_robot = robot_config.pin_robot
    gaits = control_problem_solo12.Solo12Gaits(pin_robot , contact_names)
    contact_ids = gaits.contact_ids # will need this later 
    optModels, _ = gaits.createBalanceProblem(problemConfig)
    cntrl = controllers.DDPController(optModels,  xs ,us, feedback)

    # rest robot state 
    q0 = xs[0][:pin_robot.nq]
    v0 = xs[0][pin_robot.nq:]
    robot.reset_state(q0, v0)

    # create arrays to store 
    x_hat = np.zeros([problemConfig.horizon + 1, pin_robot.nq + pin_robot.nv])
    u_act = np.zeros([problemConfig.horizon,pin_robot.nv-6])
    f_act = np.zeros([problemConfig.horizon,4,3]) # 4 feet and 3 directions 
    x_hat[0] = xs[0].copy()
    tau = np.zeros(pin_robot.nv)

    # ---------------- Configure Estimator ----------------# 

    estimator_settings = RobotStateEstimatorSettings()
    estimator_settings.is_imu_frame = False
    estimator_settings.pinocchio_model = pin_robot.model
    estimator_settings.imu_in_base = pinocchio.SE3(
        robot.rot_base_to_imu.T, robot.r_base_to_imu
    )
    estimator_settings.end_effector_frame_names = (
        robot_config.end_effector_names
    )
    estimator_settings.urdf_path = robot_config.urdf_path
    robot_weight_per_ee = robot_config.mass * 9.81 / 4
    estimator_settings.force_threshold_up = 0.8 * robot_weight_per_ee
    estimator_settings.force_threshold_down = 0.2 * robot_weight_per_ee
    print("force_threshold_up = ", estimator_settings.force_threshold_up)
    print("force_threshold_down = ", estimator_settings.force_threshold_down)

    # ---------------- Initialize Estimator ----------------# 
    estimator = RobotStateEstimator()
    estimator.initialize(estimator_settings)
    estimator.set_initial_state(q0, v0)
    # ---------------- Create the Log Files ----------------# 
    logger_file_name = str(abs_path+"/"+EXPERIMENT_NAME+"_"
    +deepcopy(time.strftime("%Y_%m_%d_%H_%M_%S")) + ".mds")

    logger = DataLogger(logger_file_name)
    # Input the data fields.
    id_time = logger.add_field("sim_time", 1)
    id_sim_imu_linacc = logger.add_field("sim_imu_linacc", 3)
    id_sim_imu_angvel = logger.add_field("sim_imu_angvel", 3)
    id_sim_base_pos = logger.add_field("sim_base_pos", 3)
    id_sim_base_vel = logger.add_field("sim_base_vel", 3)
    id_sim_base_rpy = logger.add_field("sim_base_rpy", 3)
    id_est_base_pos = logger.add_field("est_base_pos", 3)
    id_est_base_vel = logger.add_field("est_base_vel", 3)
    id_est_base_rpy = logger.add_field("est_base_rpy", 3)
    id_est_force = {}
    id_est_contact = {}
    id_est_force_norm = {}
    for ee in estimator_settings.end_effector_frame_names:
        id_est_force[ee] = logger.add_field("est_" + ee + "_force", 3)
        id_est_contact[ee] = logger.add_field("est_" + ee + "_contact", 1)
        id_est_force_norm[ee] = logger.add_field(
            "est_" + ee + "_force_norm", 1
        )
    logger.init_file()

    # initialize, reset sim, log initial state, control and forces 
    


    for t in range(problemConfig.horizon):
        for i in range(problemConfig.control_steps): 
            time_sec = t * 0.01 + i * 0.001

            # send control command 

            # get feedback 


            # compute control update 

            #  log shit 


            # get feedback 
            q, dq = robot.get_state()
            xi_hat = np.hstack([q,dq]) 
            # estimate state x_har 
            # for now assumed to be perfect observation 
            # compute control 
            ui = cntrl(t, .1*i, xi_hat)
            robot.send_joint_command(ui)
            
            env.step(sleep=False) 
            active_ids, active_forces = robot.get_force()
            if i == 0:
                # log stuff 
                x_hat[t] = xi_hat.copy()
                u_act[t] = ui.copy()
                for k, id in enumerate(active_ids):
                    if id in contact_ids:
                        f_act[t, contact_ids.index(id), :] = active_forces[k][:3]



    # # now lets plot stuff 



    # plt.figure("contact forces")
    # for i, cnt in enumerate(contact_names):
    #     plt.plot(1.e-2*np.arange(problemConfig.horizon), f_act[:, i, 2], label=cnt)
    # plt.legend()
    # plt.title("Contact Forces")
    # plt.show()