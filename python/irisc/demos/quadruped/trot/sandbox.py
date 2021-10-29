import numpy as np 
import matplotlib.pyplot as plt 




if __name__ == "__main__": 
    reference = np.load("trot.npz")
    momentum_plan = reference["mom_opt"]
    forces_plan = reference["F_opt"]
    ik_com_plan = reference["ik_com_opt"]
    ik_momentum_plan = reference["ik_mom_opt"]
    xs_plan = reference["xs"]
    us_plan = reference["us"]
    contact_plan = reference["cnt_plan"]

    for i in reference.keys():
        if i == "cnt_plan" or i == "F_opt":
            continue
        plt.figure(i)
        for k in range(reference[i].shape[1]):
            plt.plot(np.arange(reference[i].shape[0]), reference[i][:,k])
    
    plt.show()