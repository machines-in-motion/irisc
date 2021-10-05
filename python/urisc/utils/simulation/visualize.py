""" cool visualization for various simulations """
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib.lines as m_lines
import matplotlib.patches as m_patches
from matplotlib.collections import PatchCollection


class PenumaticHopper1DViz:
    def __init__(self, xs, us): 
        self.xs = xs 
        self.us = us 



    def plot_hip(self):
        pass 


    def update_visual(self): 
        pass 

# def gviz_traj(self, traj, ax=None, start_time_offset=0., draw_full_interval=1., draw_endeffector_point=False):
#         dX = self.dX
#         hip_size = .1
#         shank_length = .1
#         lower_leg_length = .1

#         if ax is None:
#             fig1 = plt.figure()
#             ax = fig1.add_subplot(111)
#         else:
#             ax.clear()

#         # ax.set_aspect('auto', 'box-forced')
#         ax.set_aspect('equal', 'box-forced')
#         endpoint = traj.shape[0] * 0.5 * hip_size * (self.dt/1e-2)
#         ax.set_xlim([-0.2, endpoint + 0.2])
#         ax.set_ylim([-0.1, 0.6])

#         xticks = np.linspace(0, np.ceil(endpoint), np.ceil(endpoint + 1))
#         ax.set_xticks(xticks)
#         ax.set_xticklabels((xticks + start_time_offset*0.05) * 0.2)

#         # Plot the ground.
#         ax.add_patch(m_patches.Rectangle(
#             (-10, -0.1), 99., 0.1,
#             fill=False,
#             hatch='/',
#         ))

#         patches = []

#         points = traj[::int(2.*(1e-2/self.dt))]
#         leg_points = np.zeros((len(points), 3, 2))

#         if draw_endeffector_point:
#             endeffector_traj = self.endeffector_traj(points)

#         for i, x in enumerate(points):
#             offset = np.array((hip_size * i, -0.035))

#             legs = offset + (0., x[0]) + ((
#                 (0., 0.),
#                 (-np.sin(x[2]) * shank_length, -np.cos(x[2]) * shank_length),
#                 (
#                     -np.sin(x[2]) * shank_length - np.sin(x[2] - x[4]) * lower_leg_length,
#                     -np.cos(x[2]) * shank_length - np.cos(x[2] - x[4]) * lower_leg_length
#                 )
#             ))
#             leg_points[i] = legs
#             leg_points[i, 0, 1] += 0.1

#             if i % draw_full_interval == 0 or i == len(points) - 1:
#                 ax.add_line(m_lines.Line2D(
#                         legs[0:3, 0], legs[0:3, 1], lw=3.5, solid_capstyle='round', color='black'))
#                 ax.add_line(m_lines.Line2D(
#                         legs[0:3, 0], legs[0:3, 1], lw=3.0, solid_capstyle='round', color='lightgray'))


#                 ax.add_patch(m_patches.Rectangle(
#                 # patches.append(m_patches.Rectangle(
#                         offset + (-hip_size/2, x[0]), hip_size, hip_size, facecolor='lightgray', linewidth=1., edgecolor='black'))

#                 # Action visualization
#                 for y in [-.0375, 0., 0.0375]:
#                     flines = offset + (0, x[0] + 0.05) + (
#                         (-0.05, y),
#                         (-0.04, y)
#                     )
#                     ax.add_line(m_lines.Line2D(flines[:, 0], flines[:, 1]))

#                 for y in [-.0375, 0., 0.0375]:
#                     flines = offset + (0, x[0] + 0.05) + (
#                         (0.05, y),
#                         (0.04, y)
#                     )
#                     ax.add_line(m_lines.Line2D(flines[:, 0], flines[:, 1]))

#                 if draw_endeffector_point:
#                     # Draw the endeffector point from the simulation explicit.
#                     pos = (
#                         offset[0] + endeffector_traj[i, 0],
#                         endeffector_traj[i, 2]
#                     )
#                     ax.add_patch(m_patches.Circle(pos, radius=0.01, color='red', zorder=9))

#             for i in range(2):
#                 # patches.append(m_patches.Rectangle(
#                 ax.add_patch(m_patches.Rectangle(
#                     offset + (-0.025 + i*0.03, hip_size/2 + x[0]), 0.02, 0.0375 * x[dX + i],
#                     fill=True,
#                     zorder=10
#             ))

#         if draw_full_interval > 1:
#             for i in range(3):
#                 ax.plot(leg_points[:, i, 0], leg_points[:, i, 1], 'b-' if i == 0 else 'b-o',
#                         markersize=3., linewidth=0.5, zorder=-99)

#         # ax.add_collection(PatchCollection(patches))
#         return traj
#         # return ax