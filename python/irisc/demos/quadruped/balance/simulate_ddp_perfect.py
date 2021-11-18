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
import matplotlib.pyplot as plt 
from mim_data_utils import DataLogger, DataReader


# create contact names for solo 
leg = ["FL", "FR", "HL", "HR"]
contact_names = []
for li in leg:
    contact_names +=[li+"_ANKLE"]

## saving path 
abs_path = os.path.abspath("../log_files")
print(abs_path) 



# if __name__ == "__main__":

#     # create bullet environment 
#     np.set_printoptions(precision=2, suppress=True)
#     env = BulletEnvWithGround(pybullet.GUI)
#     # add the robot 
#     robot = Solo12Robot()
#     env.add_robot(robot)
#     robot_config = Solo12Config() 

#     # load ddp controller 
#     solution_path = "solutions/ddp"
#     xs = np.load(solution_path+'_xs.npy')
#     us = np.load(solution_path+'_us.npy')
#     feedback = np.load(solution_path+'_K.npy')
#     pin_robot = robot_config.pin_robot
#     gaits = control_problem_solo12.Solo12Gaits(pin_robot , contact_names)
#     contact_ids = gaits.contact_ids # will need this later 
#     optModels, _ = gaits.createBalanceProblem(problemConfig)
#     cntrl = controllers.DDPController(optModels,  xs ,us, feedback)

#     # rest robot state 
#     robot.reset_state(xs[0][:pin_robot.nq],
#                       xs[0][pin_robot.nq:])

#     time.sleep(2.)

#     # create arrays to store 
#     x_hat = np.zeros([problemConfig.horizon + 1, pin_robot.nq + pin_robot.nv])
#     u_act = np.zeros([problemConfig.horizon,pin_robot.nv-6])
#     f_act = np.zeros([problemConfig.horizon,4,3]) # 4 feet and 3 directions 
#     x_hat[0] = xs[0].copy()
#     tau = np.zeros(pin_robot.nv)
#     for t in range(problemConfig.horizon):
#         for i in range(problemConfig.control_steps): 
#             # get feedback 
#             q, dq = robot.get_state()
#             xi_hat = np.hstack([q,dq]) 
#             # estimate state x_har 
#             # for now assumed to be perfect observation 
#             # compute control 
#             ui = cntrl(t, .1*i, xi_hat)
#             robot.send_joint_command(ui)
            
#             env.step(sleep=False) 
#             active_ids, active_forces = robot.get_force()
#             if i == 0:
#                 # log stuff 
#                 x_hat[t] = xi_hat.copy()
#                 u_act[t] = ui.copy()
#                 for k, id in enumerate(active_ids):
#                     if id in contact_ids:
#                         f_act[t, contact_ids.index(id), :] = active_forces[k][:3]



#     # now lets plot stuff 



#     plt.figure("contact forces")
#     for i, cnt in enumerate(contact_names):
#         plt.plot(1.e-2*np.arange(problemConfig.horizon), f_act[:, i, 2], label=cnt)
#     plt.legend()
#     plt.title("Contact Forces")
#     plt.show()