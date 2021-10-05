import os, sys
src_path = os.path.abspath('../../') 
sys.path.append(src_path) 
from utils.simulation import visualize



import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as m_lines
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection



if __name__ == "__main__":
    ax = None  

    if ax is None:
        fig1 = plt.figure()
        ax = fig1.add_subplot(111)
    else:
        ax.clear()
    
    endpoint = 5. 
    start_time_offset = 0.

    ax.set_xlim([-0.2, endpoint + 0.2])
    ax.set_ylim([-0.1, 0.6])

    xticks = np.linspace(0., np.ceil(endpoint), np.ceil(endpoint+1.) )
    # ax.set_xticks(xticks)
    # ax.set_xticklabels((xticks + start_time_offset*0.05) * 0.2)
    # ax.set_aspect('equal', 'box-forced')
    

    # # Plot the ground.
    # ax.add_patch(m_patches.Rectangle(
    #     (-10, -0.1), 99., 0.1,
    #     fill=False,
    #     hatch='/',
    # ))


    plt.show()