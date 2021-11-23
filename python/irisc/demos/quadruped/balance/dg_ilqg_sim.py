""" simulate all controllers once this is executed """

import numpy as np 
from dynamic_graph_head import ThreadHead, SimHead, SimVicon, HoldPDController, SimForcePlate
import numpy as np
import matplotlib.pylab as plt
from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
import pybullet as p 

import os, sys, time
src_path = os.path.abspath('../../../')
sys.path.append(src_path)

from utils.controllers import dg_sim_controllers


if __name__ == "__main__":
    #________ Create Simulation Environment ________#
    bullet_env = BulletEnvWithGround()
    robot = Solo12Robot()
    bullet_env.add_robot(robot) 
    pin_robot = Solo12Config.buildRobotWrapper() 

    p.resetDebugVisualizerCamera(1.6, 50, -35, (0.0, 0.0, 0.0))
    p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
    #________ Initialze Thread ________#
    head = SimHead(robot, vicon_name='solo12')
    thread_head = ThreadHead(
        0.001, # dt.
        HoldPDController(head, 3., 0.05, True), # Safety controllers.
        head, # Heads to read / write from.
        [     # Utils.
            ('vicon', SimVicon(['solo12/solo12'])),
            ('force_plate', SimForcePlate([robot]))
        ], 
        bullet_env # Environment to step.
    )

    path = 'solutions/ddp'
    xs = np.load(path+'_xs.npy')
    us = np.load(path+'_us.npy')
    feedback = np.load(path+'_K.npy')

    q0 = xs[0][:pin_robot.nq]
    v0 = xs[0][pin_robot.nq:]

    slider_pd_controller = dg_sim_controllers.SliderPDController(head, 'solo12/solo12', robot, 3., 0.05, q0, v0) 
    

    
    thread_head.head.reset_state(q0, v0)
    thread_head.switch_controllers(slider_pd_controller)


    # thread_head.start_streaming()
    # thread_head.start_logging()

    thread_head.sim_run(10000)

    # thread_head.stop_streaming()
    # thread_head.stop_logging()

    # Plot timing information.
    thread_head.plot_timing()