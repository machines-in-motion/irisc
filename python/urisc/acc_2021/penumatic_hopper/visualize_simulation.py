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
    
    endpoint = 3. 
    start_time_offset = 0.

    ax.set_xlim([-0.2, endpoint + 0.2])
    ax.set_ylim([-0.1, 2.])

    xticks = np.linspace(0., np.ceil(endpoint), int(np.ceil(endpoint+1.)))
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks + start_time_offset*0.05) * 0.2)
    ax.set_aspect('equal', 'box')
    

    # Plot the ground.
    ax.add_patch(m_patches.Rectangle(
        (-10, -0.1), 99., 0.1,
        fill=False,
        hatch='/',
    ))

    x = 1. 
    y = .5

    hip_size = .2 
    cylinder_length = .25 
    cylinder_diameter = .07 
    piston_length = .25 
    piston_width = cylinder_diameter
    piston_offset = .5 
    foot_length =  hip_size - .05
    y2 = .0 

    # Plot the mass 
    xy = (x - .5*hip_size, y - .5*hip_size)
    
    ax.add_patch(m_patches.Rectangle(
    xy, hip_size, hip_size,
    fill=True,
    linewidth=2.,
    edgecolor='k', 
    facecolor='c',
    alpha=1., 
    ))

    # plot the cylinder 
    xy = (x - .5*cylinder_diameter, y - .5*hip_size - cylinder_length)
    ax.add_patch(m_patches.Rectangle(
    xy, cylinder_diameter, cylinder_length,
    fill=False,
    linewidth=2.,
    edgecolor='k',
    alpha=1., 
    ))

    # plot the gas 
    yg = y - y2 - piston_offset + piston_length
    xg = x - .5*cylinder_diameter
    gas_length = y - .5*hip_size - yg 
    gas_width = cylinder_diameter 
    ax.add_patch(m_patches.Rectangle(
    (xg,yg), gas_width, gas_length,
    fill=True,
    color='b',
    alpha=.35, 
    ))

    # plot the foot  
    y_foot = y - y2 - piston_offset
    # xf = [x - .5*foot_length, x + .5*foot_length]
    # yf = [y_foot, y_foot]
    # ax.add_line(m_lines.Line2D(xf, yf, lw=4., solid_capstyle='round', color='k'))

    xy = (x - .5*foot_length, y_foot)
    ax.add_patch(m_patches.Rectangle(
    xy, foot_length, .025,
    fill=True,
    color='k',
    ))

    # plot piston shank 
    xshank = [x, x]
    yshank = [y_foot + 5.e-3, y_foot+ piston_length]
    ax.add_line(m_lines.Line2D(xshank, yshank, lw=4.5, color='k'))

    # plot piston cap 
    xp = x - .5 * piston_width 
    yp = y_foot+ piston_length
    xy = (xp, yp)
    ax.add_patch(m_patches.Rectangle(
    xy, cylinder_diameter, .02,
    fill=True,
    color='k',
    alpha=1., 
    ))
    # ax.add_line(m_lines.Line2D(xp, yp, lw=3., color='k'))
#                
#   
    
    plt.show()