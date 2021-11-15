import os, sys, time 
from os import path
import numpy as np
import pybullet
import pinocchio

src_path = os.path.abspath('../../../')
sys.path.append(src_path)

from bullet_utils.env import BulletEnvWithGround
from robot_properties_solo.solo12wrapper import Solo12Robot, Solo12Config
from utils.simulation import controllers 
from demos.quadruped.utils import control_problem_solo12 
import config as problemConfig 

"""
TODO:
    1. initialize simulation env in pybullet 
    2. load ddp solutions into controller class 
    3. run simulation 
"""

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
    gaits = control_problem_solo12.Solo12Gaits(robot_config.pin_robot , contact_names)
    optModels, _ = gaits.createBalanceProblem(problemConfig)
    cntrl = controllers.DDPController(optModels,  xs ,us, feedback)

    