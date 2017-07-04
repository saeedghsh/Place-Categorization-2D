'''
'''
from __future__ import print_function
import sys
sys.path.append( u'../' )

import cv2
import numpy as np
import matplotlib.pyplot as plt

import place_categorization.place_categorization as plcat


################################################################################
################################################################################
################################################################################

# ################################################################################
# def plot_point_sets (src, dst=None):
#     ''''''
#     fig, axis = plt.subplots(1,1, figsize=(20,12))
#     axis..plot(src[:,0] ,  src[:,1], 'b.')
#     if dst is not None: axis.plot(dst[:,0] ,  dst[:,1], 'r.')
#     axis.axis('equal')
#     plt.show()     # fig.show() # using fig.show() won't block the code!


def plot_ray_cast(image, pose, r,t):
    ''''''
    fig, axis = plt.subplots(1,1, figsize=(10,10))
    axis.imshow( image, cmap = 'gray', interpolation='nearest', origin='lower')
    axis.plot(pose[0], pose[1], 'r*')
    x = pose[0] + r*np.cos(t)
    y = pose[1] + r*np.sin(t)
    axis.plot(x, y, 'b.-')
    plt.show()

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
    r,t = plcat.raycast_bitmap(image, pose_,
                               raycast_config['occupancy_thr'],
                               raycast_config['length_range'],
                               raycast_config['length_steps'], 
                               raycast_config['theta_range'],
                               raycast_config['theta_res'],
                               rays_array_xy=raxy)

    ### plot the result
    plot_ray_cast(image, pose_, r,t)
