import numpy as np 
import matplotlib.pyplot as plt 


if __name__ == "__main__":
    seeking_xsim = np.load("results/risk_seeking/seeking_xsim.npy")
    seeking_xhsim= np.load("results/risk_seeking/seeking_xhsim.npy") 
    seeking_usim= np.load("results/risk_seeking/seeking_usim.npy")
    seeking_Ksim = np.load("results/risk_seeking/seeking_Ksim.npy") 
    seeking_fsim = np.load("results/risk_seeking/seeking_fsim.npy") 



    print(seeking_xsim.shape)
    print(seeking_xhsim.shape)
    print(seeking_usim.shape)
    print(seeking_Ksim.shape)
    print(seeking_fsim.shape)