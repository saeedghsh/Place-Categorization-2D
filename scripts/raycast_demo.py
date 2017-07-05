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
import sys
sys.path.append( u'../' )

import cv2
import numpy as np
# import matplotlib.pyplot as plt

import place_categorization.place_categorization as plcat
import place_categorization.plotting as pcplt
################################################################################
################################################################################
################################################################################
def get_random_pose (image, occupancy_thr):
    ''''''
    while True:
        pose = np.array([ np.random.random_integers(0, image.shape[1]-1),
                           np.random.random_integers(0, image.shape[0]-1) ])
        if image[int(pose[1]), int(pose[0])] > occupancy_thr:
            return pose


################################################################################
################################################################################
################################################################################
if __name__ == '__main__':
    '''
    usage:
    python raycast_demo.py --filename 'filename.ext'
    '''
  
    args = sys.argv
    # fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument   
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = next( listiterator )
            if item[:2] == '--':
                exec(item[2:] + ' = next( listiterator )')
        except:
            break   
    

    ### load image
    image = np.flipud( cv2.imread( file_name, cv2.IMREAD_GRAYSCALE) )

    ### raycasting config
    raycast_config = {
        'mpp': 0.02, # meter per pixel
        'range_meter': 4, # meter
        'length_range': 200, #range_meter_ / mpp_
        'length_steps': 200, #int(length_range_)
        'theta_range': 2*np.pi,
        'theta_res': 1/1, # step/degree
        'occupancy_thr': 220,
    }

    ### constructing the rays_array_xy
    # constructing the rays_array_xy is time consuming, almost as much as raycasting
    # by creating it once and copying it for each new reaycast, it will speed up two-fold
    # no need to mention, this is not necessary in this demo and one could pass None
    # to raycast_bitmap for rays_array_xy, by it is a good practice in general to construct
    # this template once and pass it to raycast_bitmap method.
    rays_array_xy = plcat.construct_raycast_array(np.array([0,0]),
                                                  raycast_config['length_range'],
                                                  raycast_config['length_steps'], 
                                                  raycast_config['theta_range'],
                                                  raycast_config['theta_res'])
    raxy = rays_array_xy
    # raxy.shape = (2 x n x m) n: number of rays (theta), m: number of points in each ray (range)
    # raxy[0,:,:] x-values (column index to image)
    # raxy[1,:,:] y-values (row index to image)
  
    ### find a random pose in the open space
    pose_ = get_random_pose (image, raycast_config['occupancy_thr'])
    
    ### raycast
    r,t = plcat.raycast_bitmap(pose_, image,
                               raycast_config['occupancy_thr'],
                               raycast_config['length_range'],
                               raycast_config['length_steps'], 
                               raycast_config['theta_range'],
                               raycast_config['theta_res'],
                               rays_array_xy=raxy)

    ### plot the result
    pcplt.plot_ray_cast(image, pose_, r,t)
