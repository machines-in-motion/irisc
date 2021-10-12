import numpy as np 
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    ddp_xsim = np.load("results/ddp/ddp_xsim.npy")
    ddp_xhsim= np.load("results/ddp/ddp_xhsim.npy") 
    ddp_usim= np.load("results/ddp/ddp_usim.npy")
    ddp_Ksim = np.load("results/ddp/ddp_Ksim.npy") 
    ddp_fsim = np.load("results/ddp/ddp_fsim.npy") 



    print(ddp_xsim.shape)
    print(ddp_xhsim.shape)
    print(ddp_usim.shape)
    print(ddp_Ksim.shape)
    print(ddp_fsim.shape)