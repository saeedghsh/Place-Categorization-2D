'''
Copyright (C) Saeed Gholami Shahbandi. All rights reserved.
Author: Saeed Gholami Shahbandi (saeed.gh.sh@gmail.com)

This file is part of Arrangement Library.
The of Arrangement Library is free software: you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public License as published
by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public License along
with this program. If not, see <http://www.gnu.org/licenses/>
'''
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

################################################################################
def plot_ray_cast(image, pose,
                  raycast_config,
                  r,t):
    ''''''
    r_in_pixel = np.array( r / raycast_config['mpp'], dtype=np.int)

    fig, axis = plt.subplots(1,1, figsize=(10,10))
    axis.imshow( image, cmap = 'gray', interpolation='nearest', origin='lower')
    axis.plot(pose[0], pose[1], 'r*')
    x = pose[0] + r_in_pixel*np.cos(t)
    y = pose[1] + r_in_pixel*np.sin(t)
    axis.plot(x, y, 'b.-')
    plt.show()



################################################################################
def plot_rays_arr_xy(image, pose, raycast_config, raxy):
    ''''''
    fig, axis = plt.subplots(1,1, figsize=(10,10))
    axis.imshow( image, cmap = 'gray', interpolation='nearest', origin='lower')
    # axis.plot(pose[0], pose[1], 'r*')   
    axis.plot(raxy[0,:,:], raxy[1,:,:], 'b.')
    plt.show()


# ################################################################################
# def plot_point_sets (src, dst=None):
#     ''''''
#     fig, axis = plt.subplots(1,1, figsize=(20,12))
#     axis..plot(src[:,0] ,  src[:,1], 'b.')
#     if dst is not None: axis.plot(dst[:,0] ,  dst[:,1], 'r.')
#     axis.axis('equal')
#     plt.show()     # fig.show() # using fig.show() won't block the code!
