""" sets up control loop in simulation with dg head stuff """

import numpy as np
import scipy.linalg as scl 
import matplotlib.pylab as plt
import os, sys, time 
from copy import deepcopy
from pathlib import Path
# src_path = os.path.abspath('../../../')
# sys.path.append(src_path)


from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController
import pinocchio as pin
from utils.simulation import controllers 
from demos.quadruped.utils import control_problem_solo12 
import config as problemConfig 

from mim_estimation_cpp import RobotStateEstimator, RobotStateEstimatorSettings
from mim_data_utils import DataLogger, DataReader





class SliderPDController:
    def __init__(self, head, vicon_name, robot, Kp, Kd, q0, v0):
        #________ PD Parameters ________#
        self.head = head
        self.scale = np.pi
        self.Kp = Kp
        self.Kd = Kd
        self.q0 = q0 
        self.v0 = v0

        #________ Robot Parameters ________#
        self.robot = robot
        self.robot_config = Solo12Config() 
        self.pin_robot = self.robot_config.buildRobotWrapper()
        self.vicon_name = vicon_name
        self.contact_names = []
        leg = ["FL", "FR", "HL", "HR"]
        for li in leg:
            self.contact_names +=[li+"_ANKLE"]

        #________ Data Logs ________#
        self.tau = np.zeros(self.pin_robot.nv)
        self.q_sim = np.zeros(self.pin_robot.nq)
        self.v_sim = np.zeros(self.pin_robot.nv)
        self.q_est = np.zeros(self.pin_robot.nq)
        self.v_est = np.zeros(self.pin_robot.nv)
        self.x_sim = np.zeros(self.pin_robot.nq+ self.pin_robot.nv)
        self.x_est = np.zeros(self.pin_robot.nq+ self.pin_robot.nv)
        self.f_sim = np.zeros([4,3])
        self.f_est = np.zeros([4,3])
        self.c_sim = np.zeros(4)
        self.c_est = np.zeros(4)    
        # 
        self.sim_imu_linacc = np.zeros(3)
        self.sim_imu_angvel = np.zeros(3) 
    
        #________ initialze estimator ________#

        estimator_settings = RobotStateEstimatorSettings()
        estimator_settings.is_imu_frame = False
        estimator_settings.pinocchio_model = self.pin_robot.model
        estimator_settings.imu_in_base = pin.SE3(self.robot.rot_base_to_imu.T, self.robot.r_base_to_imu)
        estimator_settings.end_effector_frame_names = (self.robot_config.end_effector_names)
        estimator_settings.urdf_path = self.robot_config.urdf_path
        robot_weight_per_ee = self.robot_config.mass * 9.81 / 4
        estimator_settings.force_threshold_up = 0.8 * robot_weight_per_ee
        estimator_settings.force_threshold_down = 0.2 * robot_weight_per_ee
        self.estimator = RobotStateEstimator()
        self.estimator.initialize(estimator_settings)
        self.estimator.set_initial_state(self.q0, self.v0)

        #________ map to sensors ________#

        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')
        self.slider_positions = head.get_sensor('slider_positions')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')
        self.imu_accelerometer = head.get_sensor('imu_accelerometer')


    def map_sliders(self, sliders):
        sliders_out = np.zeros(12)
        slider_A = sliders[0]
        slider_B = sliders[1]
        for i in range(4):
            sliders_out[3 * i + 0] = slider_A
            sliders_out[3 * i + 1] = slider_B
            sliders_out[3 * i + 2] = 2. * (1. - slider_B)
            if i >= 2:
                sliders_out[3 * i + 1] *= -1
                sliders_out[3 * i + 2] *= -1
        # Swap the hip direction.
        sliders_out[3] *= -1
        sliders_out[9] *= -1
        return sliders_out

    def warmup(self, thread):
        self.zero_pos = self.map_sliders(self.slider_positions)
        thread.vicon.bias_position(self.vicon_name)

    def get_base(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        base_vel[3:] = self.imu_gyroscope
        return base_pos, base_vel

    def run(self, thread):
        #________ read vicon, encoders and imu ________#
        self.q_sim[:7], self.v_sim[:6] = self.get_base(thread)
        self.q_sim[7:] = self.joint_positions.copy()
        self.v_sim[6:] = self.joint_velocities.copy()
        self.x_sim[:self.pin_robot.nq] = self.q_sim
        self.x_sim[self.pin_robot.nq:] = self.v_sim
        self.sim_imu_linacc[:] = self.imu_accelerometer.copy()
        self.sim_imu_angvel[:] = self.imu_gyroscope.copy()  

        #________ Compute updated estimate from EKF ________#
        self.estimator.run(self.c_est, self.sim_imu_linacc, self.sim_imu_angvel, 
                           self.q_sim[7:], self.v_sim[6:])
        self.estimator.get_states(self.q_est, self.v_est)
        for i,n in enumerate(self.contact_names):
            self.f_est[i,:] = self.estimator.get_force(n)
        self.c_est = self.estimator.get_detected_contact()
        
        self.x_est[:self.pin_robot.nq] = self.q_est
        self.x_est[self.pin_robot.nq:] = self.v_est

        #________ Run Actual Controller ________#
        self.des_position = self.scale * (
            self.map_sliders(self.slider_positions) - self.zero_pos)

        self.tau = self.Kp * (self.des_position - self.joint_positions) - self.Kd * self.joint_velocities
        thread.head.set_control('ctrl_joint_torques', self.tau)



class IterativeLinearQuadraticController:
    def __init__(self, head, vicon_name, path, experiment_name="ilqg"):
        """ 
        args:
            path: path to files where iLQR Solution is stored  
        """
        self.robot = Solo12Robot()
        self.robot_config = Solo12Config() 
        self.pin_robot = self.robot_config.buildRobotWrapper()
        self.vicon_name = vicon_name
        self.contact_names = []
        leg = ["FL", "FR", "HL", "HR"]
        for li in leg:
            self.contact_names +=[li+"_ANKLE"]
        
        self.t = 0 
        self.d = 0 
        #________ Data Logs ________#
        self.tau = np.zeros(self.pin_robot.nv)
        self.q_sim = np.zeros(self.pin_robot.nq)
        self.v_sim = np.zeros(self.pin_robot.nv)
        self.q_est = np.zeros(self.pin_robot.nq)
        self.v_est = np.zeros(self.pin_robot.nv)
        self.x_sim = np.zeros(self.pin_robot.nq+ self.pin_robot.nv)
        self.x_est = np.zeros(self.pin_robot.nq+ self.pin_robot.nv)
        self.f_sim = np.zeros([4,3])
        self.f_est = np.zeros([4,3])
        self.c_sim = np.zeros(4)
        self.c_est = np.zeros(4)    
        # 
        self.sim_imu_linacc = np.zeros(3)
        self.sim_imu_angvel = np.zeros(3) 
    
        #________ parse the plan ________#  

        self.path = path 
        self.xs = np.load(self.path+'_xs.npy')
        self.us = np.load(self.path+'_us.npy')
        self.feedback = np.load(self.path+'_K.npy')
        self.q0 = self.xs[0][:self.pin_robot.nq]
        self.v0 = self.xs[0][self.pin_robot.nq:]
    
        
        
        #________ initialze controller ________#

        self.gaits = control_problem_solo12.Solo12Gaits(self.pin_robot , self.contact_names)
        self.contact_ids = self.gaits.contact_ids # will need this later 
        self.optModels, _ = self.gaits.createBalanceProblem(problemConfig)
        self.cntrl = controllers.DDPController(self.optModels,  self.xs ,self.us, self.feedback)

        #________ initialze estimator ________#

        estimator_settings = RobotStateEstimatorSettings()
        estimator_settings.is_imu_frame = False
        estimator_settings.pinocchio_model = self.pin_robot.model
        estimator_settings.imu_in_base = pin.SE3(self.robot.rot_base_to_imu.T, self.robot.r_base_to_imu)
        estimator_settings.end_effector_frame_names = (self.robot_config.end_effector_names)
        estimator_settings.urdf_path = self.robot_config.urdf_path
        robot_weight_per_ee = self.robot_config.mass * 9.81 / 4
        estimator_settings.force_threshold_up = 0.8 * robot_weight_per_ee
        estimator_settings.force_threshold_down = 0.2 * robot_weight_per_ee
        self.estimator = RobotStateEstimator()
        self.estimator.initialize(estimator_settings)
        self.estimator.set_initial_state(self.q0, self.v0)

        #________ initialze data logger ________#

        # self.abs_path = abs_path = os.path.abspath("../log_files")
        # self.exp_name = experiment_name
        # self.logger_file_name = str(abs_path+"/"+self.exp_name+"_"+deepcopy(time.strftime("%Y_%m_%d_%H_%M_%S")) + ".mds")
        # self.logger = DataLogger(self.logger_file_name)
        # # Input the data fields.
        # self.id_time = self.logger.add_field("sim_time", 1)
        # self.sim_q = self.logger.add_field("sim_q", self.pin_robot.nq)
        # self.sim_v = self.logger.add_field("sim_v", self.pin_robot.nv)
        # self.est_q = self.logger.add_field("est_q", self.pin_robot.nq)
        # self.est_v = self.logger.add_field("est_v", self.pin_robot.nv)
        # self.sim_imu_linacc = self.logger.add_field("sim_imu_linacc", 3)
        # self.sim_imu_angvel = self.logger.add_field("sim_imu_angvel", 3)
        # self.est_imu_linacc = self.logger.add_field("est_imu_linacc", 3)
        # self.est_imu_angvel = self.logger.add_field("est_imu_angvel", 3)
        # self.sim_forces = {}
        # self.est_forces = {}
        # self.sim_contacts = {}
        # self.est_contacts = {}
        # for ee in estimator_settings.end_effector_frame_names:
        #     self.sim_forces[ee] = self.logger.add_field("sim_" + ee + "_force", 3)
        #     self.est_forces[ee] = self.logger.add_field("est_" + ee + "_force", 3)
        #     self.sim_contacts[ee] = self.logger.add_field("sim_" + ee + "_contact", 3)
        #     self.est_contacts[ee] = self.logger.add_field("est_" + ee + "_contact", 3)
        # self.logger.init_file()

        #________ map to sensors ________#

        self.joint_positions = head.get_sensor('joint_positions')
        self.joint_velocities = head.get_sensor('joint_velocities')
        self.slider_positions = head.get_sensor('slider_positions')
        self.imu_gyroscope = head.get_sensor('imu_gyroscope')
        self.imu_accelerometer = head.get_sensor('imu_accelerometer')

    def warmup(self, thread):
        # thread.vicon.bias_position(self.vicon_name)
        thread.vicon.bias_position()

    def get_base(self, thread):
        base_pos, base_vel = thread.vicon.get_state(self.vicon_name)
        base_vel[3:] = self.imu_gyroscope
        return base_pos, base_vel
    
    def run(self, thread):
        #________ read vicon, encoders and imu ________#
        self.q_sim[:7], self.v_sim[:6] = self.get_base(thread)
        self.q_sim[7:] = self.joint_positions.copy()
        self.v_sim[6:] = self.joint_velocities.copy()
        self.x_sim[:self.pin_robot.nq] = self.q_sim
        self.x_sim[self.pin_robot.nq:] = self.v_sim
        self.sim_imu_linacc[:] = self.imu_accelerometer.copy()
        self.sim_imu_angvel[:] = self.imu_gyroscope.copy()  

        #________ Compute updated estimate from EKF ________#
        self.estimator.run(self.c_est, self.sim_imu_linacc, self.sim_imu_angvel, 
                           self.q_sim[7:], self.v_sim[6:])
        self.estimator.get_states(self.q_est, self.v_est)
        for i,n in enumerate(self.contact_names):
            self.f_est[i,:] = self.estimator.get_force(n)
        self.c_est = self.estimator.get_detected_contact()
        
        self.x_est[:self.pin_robot.nq] = self.q_est
        self.x_est[self.pin_robot.nq:] = self.v_est
        #________ Compute Control Based on EKF ________#
        self.tau[6:] = self.cntrl(self.t, self.d, self.x_est)

        #________ Increment Counter ________# 
        self.d += 0.1 
        if (self.d - 1.)**2 <= 1.e-8:
            self.d = 0. 
            self.t += 1 

        #________ Send Control Command ________#                
        thread.head.set_control('ctrl_joint_torques', self.tau[6:])

    # def log_data(self):
    #     self.logger.begin_timestep()
    #     self.logger.log(self.id_time, )
    #     self.logger.log(self.sim_q, )
    #     self.logger.log(self.sim_v, )
    #     self.logger.log(self.est_q, )
    #     self.logger.log(self.est_v, )
    #     self.logger.log(self.sim_imu_linacc, )
    #     self.logger.log(self.sim_imu_angvel, )
    #     self.logger.log(self.est_imu_linacc, )
    #     self.logger.log(self.est_imu_angvel, )
    #     self.logger.log()
    #     self.logger.log()
    #     self.logger.end_timestep()
