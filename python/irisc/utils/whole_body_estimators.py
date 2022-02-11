""" This class includes a somehow more complex whole body estimator that fuses IMU and joint measurements 
to construct a full state of the robot """

import numpy as np 
import pinocchio as pin 





class ContactDetection:
    def __init__(self) -> None:
        """ a class that implements a contact detection heuristic that decides which feet are in contact """
        pass



class WholeBodyEKF:
    def __init__(self):
        """ An extended kalman filter for the whole body of the robot, i.e. it spits out xhat = [q_base, q_joints, v_base, v_joints] 
        where q_base = [x, y, z, quat1, quat2, quat3, quat4], the base velocities are expressed in the base frame following 
        the pinocchio convention. 
        At each estimation step, the first thing used is contact detection/estimator, then the active contacts are used to decide the whole 
        body model to use.  
        """
        pass 






