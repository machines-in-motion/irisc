import numpy as np 
import matplotlib.pyplot as plt 
from hopper_config import *
import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 

from utils.action_models import penumatic_hopper

if __name__ == "__main__": 
    dynamics = penumatic_hopper.PenumaticHopped1D()
    xs_ddp = np.load("solutions/ddp_xs.npy")
    us_ddp = np.load("solutions/ddp_us.npy")
    K_ddp = np.load("solutions/ddp_K.npy")
    print(K_ddp.shape)

    xs_risk_seeking = np.load("solutions/risk_seeking/uRiSC_xs.npy")
    us_risk_seeking = np.load("solutions/risk_seeking/uRiSC_us.npy")
    K_risk_seeking = np.load("solutions/risk_seeking/uRiSC_K.npy")

    xs_risk_senstive = np.load("solutions/risk_sensitive/uRiSC_xs.npy")
    us_risk_senstive = np.load("solutions/risk_sensitive/uRiSC_us.npy")
    K_risk_senstive = np.load("solutions/risk_sensitive/uRiSC_K.npy")

    print(K_risk_senstive.shape)

    feet_ddp = xs_ddp[:,0] - xs_ddp[:,1] - .5*np.ones_like(time_array)
    feet_seeking = xs_risk_seeking[:,0] - xs_risk_seeking[:,1] - .5*np.ones_like(time_array)
    feet_averse = xs_risk_senstive[:,0] - xs_risk_senstive[:,1] - .5*np.ones_like(time_array)



    plt.figure("Planned Trajectory")
    plt.plot(time_array, xs_ddp[:,0], label="Hip DDP")
    plt.plot(time_array, xs_risk_seeking[:,0], label="Hip $\sigma=0.1$")
    plt.plot(time_array, xs_risk_senstive[:,0], label="Hip $\sigma=-0.1$")

    plt.plot(time_array, feet_ddp,'--' ,label="Foot DDP")
    plt.plot(time_array, feet_seeking,'--', label="Foot $\sigma=0.1$")
    plt.plot(time_array, feet_averse,'--', label="Foot $\sigma=-0.1$")
    plt.legend()


    plt.figure("Control Trajectory")
    plt.plot(time_array[:-1], us_ddp, label="u[t] DDP")
    plt.plot(time_array[:-1], us_risk_seeking, label="u[t] $\sigma=0.1$")
    plt.plot(time_array[:-1], us_risk_senstive, label="u[t] $\sigma=-0.1$")
    plt.legend()

    style = ['-','--','-.',':']
    state_names = ['x1', 'x2', 'x3', 'x4']
    Ks = [K_ddp, K_risk_seeking, K_risk_senstive]
    names = ["DDP", "$\sigma=0.1$", "$\sigma=-0.1$"]
    ig, ax = plt.subplots(1, 4,figsize=(30,15))
    for i in range(4):
        for k, name in enumerate(names):
            if "DDP" in name:
                ax[i].plot(time_array[:-1], Ks[k][:,i], label=name + " K %s "%state_names[i])
            else:
                ax[i].plot(time_array[:-1], Ks[k][:,0,i], label=name + " K %s "%state_names[i])
        ax[i].legend()

    # ax[3].legend(loc="center left", bbox_to_anchor=(1., -.1), ncol= 1)
    # for i in range(2):
    #     for j in range(4): 
    #         for k, solver in enumerate(solvers): 
    #             if "ddp" in solver_names[k]:
    #                 ax[i,j].plot(time_array[:-1], -np.array(solver.K)[:,i,j], 'k', linewidth=5., label=solver_names[k])
    #             else:
    #                 ax[i,j].plot(time_array[:-1], np.array(solver.K)[:,i,j],linewidth=2., label=solver_names[k])
    # ax[0,3].legend(loc="center left", bbox_to_anchor=(1., -.1), ncol= 1)

# if SAVE_FIGS:
#     plt.savefig(SAVE_PATH+'feedback_trajectory.png')


    # for i in range(4):
    #     plt.plot(time_array[:-1], K_ddp[:,i], style[i], label="K %s DDP"%state_names[i])
    #     plt.plot(time_array[:-1], K_risk_seeking[:,0,i], style[i], label="K %s $\sigma=0.1$"%state_names[i])
    #     plt.plot(time_array[:-1], K_risk_senstive[:,0,i], style[i], label="K %s $\sigma=-0.1$"%state_names[i])
    # plt.legend()

    plt.show()