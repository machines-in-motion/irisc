import numpy as np 
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    averse_xsim = np.load("results/risk_averse/averse_xsim.npy")
    averse_xhsim= np.load("results/risk_averse/averse_xhsim.npy") 
    averse_usim= np.load("results/risk_averse/averse_usim.npy")
    averse_Ksim = np.load("results/risk_averse/averse_Ksim.npy") 
    averse_fsim = np.load("results/risk_averse/averse_fsim.npy") 



    print(averse_xsim.shape)
    print(averse_xhsim.shape)
    print(averse_usim.shape)
    print(averse_Ksim.shape)
    print(averse_fsim.shape)