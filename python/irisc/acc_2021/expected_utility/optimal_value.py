""" for the simple case where x = u + w and L = x^2 we can plot how the cost varies as a function of u, vs the optimal value """

import numpy as np 
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

sigmas = [1.e-6, 5.e-6, 1.e-5, 5.e-5, 1.e-4, 5.e-4, 1.e-3, 5.e-3, 1.e-2]
us = np.arange(-1., 1., .02)
# gammas = [-.1, -.08, -.06, -.04,-.02, .02, .04, .06, .08, .1]
# gammas = [2.7**(-1.*.5**i) for i in range(6)]
gammas = [1., 2., 3., 4., 5., 6., 7., 8., 9.]

LINEWIDTH = 100 

def simple_cost_expectation(sensitivity, noise, control): 
    sig_bar_2 = 1./(-sensitivity + 1/noise) 
    sig_bar = np.sqrt(sig_bar_2)
    r = 2 + sensitivity*sig_bar_2
    cost = (2/sensitivity)*np.log(sig_bar/noise) + r*control**2 + control 
    return cost 

def optimal_control(sensitivity, noise):
    sig_bar_2 = 1./(-sensitivity + 1/noise) 
    control = -.5/(2 + sig_bar_2*sensitivity)
    return control


if __name__ == "__main__":
    print("Plotting cost vs u".center(LINEWIDTH,"="))
    plt.figure(" Cost vs Control ")
    opt_c = []
    opt_u = []
    for g in gammas:
        # for sig in sigmas:
        c = []
        
        for ui in us:
            c += [simple_cost_expectation(g, sigmas[-1], ui)] 
        plt.plot(us, c, linewidth=2., label="$\gamma$=%s"%g)
        opt_u += [optimal_control(g, sigmas[-1])]
        opt_c += [simple_cost_expectation(g, sigmas[-1], opt_u[-1])]
    
    plt.scatter(opt_u, opt_c)
    plt.legend()
    
    print("Plotting cost vs u different noise ".center(LINEWIDTH,"="))
    plt.figure(" Cost vs Control Different Noise ")
    opt_c = []
    opt_u = []
    for sig in sigmas:
        # for sig in sigmas:
        c = []
        
        for ui in us:
            c += [simple_cost_expectation(gammas[-1], sig, ui)] 
        plt.plot(us, c, linewidth=2., label="$\sigma$=%s"%sig)
        opt_u += [optimal_control(gammas[-1], sig)]
        opt_c += [simple_cost_expectation(gammas[-1], sig, opt_u[-1])]
    
    plt.scatter(opt_u, opt_c)
    plt.legend()


    plt.figure(" Opt. Control vs Noise ")
    for g in gammas:
        u_opt = []
        for sig in sigmas:
            u_opt += [optimal_control(g, sig)]
        plt.plot(sigmas, u_opt, linewidth=2.5, label="$\gamma$ = %s"%g)

    plt.legend()




    fig = plt.figure("U opt fcn of both")
    ax = fig.add_subplot(111, projection='3d')

    X, Y = np.meshgrid(sigmas, gammas)

    u_opt = np.zeros(X.shape)

    for i, gam in enumerate(gammas):
        for j, sig in enumerate(sigmas):
            u_opt[i,j] = optimal_control(gam, sig)

    surf = ax.plot_surface(X, Y, u_opt, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    





    plt.show()

