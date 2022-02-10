# import pybullet as p
# from bullet_utils.env import BulletEnvWithGround
# from robot_properties_kuka.config import IiwaConfig

# kuka_config = IiwaConfig()

import pinocchio as pin 
from robot_properties_solo.config import Solo12Config 

if __name__=="__main__":
    model, collision_model, visual_model = pin.buildModelsFromUrdf(Solo12Config.urdf_path, Solo12Config.meshes_path, pin.JointModelFreeFlyer())
 

    viz = pin.visualize.MeshcatVisualizer(model, collision_model, visual_model)
    viz.initViewer(open=True)
    # viz.loadViewerModel() 