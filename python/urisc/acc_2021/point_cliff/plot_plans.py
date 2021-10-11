import numpy as np 
import matplotlib.pyplot as plt 

import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 
from config_point_cliff import *

if __name__ == "__main__": 
    horizon = 100 
    dt = 1.e-2 
    time_array = dt*np.arange(horizon+1)
    xs_ddp = np.load("solutions/ddp_xs.npy")
    us_ddp = np.load("solutions/ddp_us.npy")
    K_ddp = np.load("solutions/ddp_K.npy")
    print(K_ddp.shape)

    xs_risk_seeking = np.load("solutions/risk_seeking/iRiSC_xs.npy")
    us_risk_seeking = np.load("solutions/risk_seeking/iRiSC_us.npy")
    K_risk_seeking = np.load("solutions/risk_seeking/iRiSC_K.npy")

    xs_risk_senstive = np.load("solutions/risk_averse/iRiSC_xs.npy")
    us_risk_senstive = np.load("solutions/risk_averse/iRiSC_us.npy")
    K_risk_senstive = np.load("solutions/risk_averse/iRiSC_K.npy")



    plt.figure("Planned Trajectory")
    plt.plot(xs_ddp[:,0], xs_ddp[:,1], label="DDP")
    plt.plot(xs_risk_seeking[:,0], xs_risk_seeking[:,1], label="$\sigma=%s$"%abs(sensitivity))
    plt.plot(xs_risk_senstive[:,0], xs_risk_senstive[:,1], label="$\sigma=-%s$"%abs(sensitivity))
    plt.legend()


    plt.figure("Control Trajectory")
    plt.plot(time_array[:-1], us_ddp, label="u[t] DDP")
    plt.plot(time_array[:-1], us_risk_seeking, label="u[t] $\sigma=%s$"%abs(sensitivity))
    plt.plot(time_array[:-1], us_risk_senstive, label="u[t] $\sigma=-%s$"%abs(sensitivity))
    plt.legend()

    style = ['-','--','-.',':']
    state_names = ['x1', 'x2', 'x3', 'x4']
    Ks = [K_ddp, K_risk_seeking, K_risk_senstive]
    names = ["DDP", "$\sigma=%s$"%abs(sensitivity), "$\sigma=-%s$"%abs(sensitivity)]
    ig, ax = plt.subplots(2, 4,figsize=(30,15))
    for i in range(2):
        for j in range(4):
            for k, name in enumerate(names):
                label = name + " " + state_names[j]
                ax[i,j].plot(time_array[:-1], Ks[k][:,i,j], label=label)
                    
            ax[i,j].legend()

    plt.show()