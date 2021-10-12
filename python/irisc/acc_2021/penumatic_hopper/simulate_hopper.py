""" Runs a Hopper Simulator with parameters from hopper_config """

from hopper_config import * 
import crocoddyl 
import matplotlib.pyplot as plt 
import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 
from utils.action_models import penumatic_hopper 
from utils.simulation import controllers, simulator

models = []
for t in range(horizon): 
    diff_hopper = penumatic_hopper.DifferentialActionModelHopper(t, horizon, False) 
    models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 
diff_hopper = penumatic_hopper.DifferentialActionModelHopper(horizon, horizon, True) 
models += [crocoddyl.IntegratedActionModelEuler(diff_hopper, dt)] 



if __name__ == "__main__": 
    ddp_soln = solutions_path  + '/ddp'

    xs = np.load(ddp_soln+'_xs.npy')
    us = np.load(ddp_soln+'_us.npy')
    feedback = np.load(ddp_soln+'_K.npy')
    controller = controllers.DDPController(models, xs ,us, feedback)

    hopper_dynamics = penumatic_hopper.PenumaticHopped1D()
    estimator = None 
    p_noise = None 
    m_noise = None 
    sim = simulator.HopperSimulator(hopper_dynamics, controller, estimator, 
                                    p_noise, m_noise, x0, horizon)


    print("n integration steps = %s"%sim.n_steps)

    sim.simulate()
    
    trajectory = np.array(sim.xsim)

    contact_pt_height = trajectory[:,0] - trajectory[:,1] - .5*np.ones_like(time_array)
    foot_planned = xs[:,0] - xs[:,1] - .5*np.ones_like(time_array)
    ground_hieght = sim.env*np.ones_like(time_array)
    plt.figure("trajectory plot")
    plt.plot(time_array,trajectory[:,0], label="Mass DDP")
    plt.plot(time_array,xs[:,0], '--', label="Mass Planned")
    plt.plot(time_array,contact_pt_height, label="Foot DDP")
    plt.plot(time_array, foot_planned, '--', label="foot Planned")
    plt.plot(time_array,ground_hieght, '--k',linewidth=2.)
    plt.legend()

    fine_time_array = np.arange(0., dt*horizon, 1.e-4)
    plt.figure("Contact Forces")
    plt.plot(fine_time_array, sim.fsim)



    plt.show()