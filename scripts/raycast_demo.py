from __future__ import print_function

import os
import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

new_paths = [
    u'../'
    # u'/home/saesha/Dropbox/myGits/arrangement/',
    # u'/home/saesha/Dropbox/myGits/Python-CPD/',
    # u'/home/saesha/Dropbox/myGits/dev/',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import time
import cv2
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt
import plyfile

from core import place_categorization as plcat
reload(plcat)


################################################################################
############################################################### functions' lobby
################################################################################

########################################
def plot_point_sets (src, dst=None):
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    fig.axes[0].plot(src[:,0] ,  src[:,1], 'b.')
    if dst is not None:
        fig.axes[0].plot(dst[:,0] ,  dst[:,1], 'r.')
    fig.axes[0].axis('equal')
    # fig.show() # using fig.show() won't block the code!
    plt.show()

######################################## Working with bitmap file
image_name = '/home/saesha/Dropbox/myGits/place_categorization_2D/kpt4a_full.png'
image_name = '/home/saesha/Desktop/Kyushu_Indoor/patched_saeed/patched_0063.png'

image_name = '/home/saesha/Dropbox/myGits/sample_data/HH/E5_06.png'
# image_name = '/home/saesha/Dropbox/myGits/sample_data/HH/F5_04.png'
# image_name = '/home/saesha/Dropbox/myGits/sample_data/HH/HIH_04.png'
# image_name = '/home/saesha/Dropbox/myGits/sample_data/sweet_home/kpt4a.png'



image = np.flipud( cv2.imread( image_name, cv2.IMREAD_GRAYSCALE) )
occupancy_thr = 180 #200

if 0:
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    fig.axes[0].imshow( image, cmap = 'gray', interpolation='nearest', origin='lower')
    plt.show()

######################################## raycasting

# raycasting config
length_range_  = 400
length_steps_  = length_range_ 
theta_range_   = 2*np.pi
theta_res_     = 1/1
occupancy_thr_ = 180 # 210

####### constructing the rays_array_xy
pose_ = np.array([0,0,0]) # x,y,theta
rays_array_xy = plcat.construct_raycast_array(pose_,
                                              length_range_, length_steps_, 
                                              theta_range_, theta_res_)
raxy = rays_array_xy


if 0:
    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    fig.axes[0].plot(rays_array_xy[0,:,:], rays_array_xy[1,:,:], 'b.')
    fig.axes[0].axis('equal')
    plt.show()
    

###### raycasting in bitmap image - example
if 1:
    pose_ = np.array([117,230,0]) # x,y,theta
    pose_ = np.array([690,510,0]) # for demo4.gfs-0063.png
    pose_ = np.array([150,200,0]) # for 20170131163311_.png

    pose_ = np.array([500,250,0]) # for kptgn4a
    pose_ = np.array([150,50,0]) # for HIH layout
    pose_ = np.array([375, 750,0]) # for F5 layout
    pose_ = np.array([400, 600,0]) # for E5 layout


    r,t = plcat.raycast_bitmap(image, pose_,
                               occupancy_thr_,
                               length_range_, length_steps_, 
                               theta_range_, theta_res_,
                               rays_array_xy=raxy)

    fig = plt.figure()
    fig.add_axes([0, 0, 1, 1])
    fig.axes[0].imshow( image, cmap = 'gray', interpolation='nearest', origin='lower')
    fig.axes[0].plot(pose_[0], pose_[1], 'r*')
    x = pose_[0] + r*np.cos(t)
    y = pose_[1] + r*np.sin(t)
    fig.axes[0].plot(x, y, 'b.-')
    plt.show()

    
# file_list = [ file_name 
#               for file_name in os.listdir( '/home/saesha/Desktop/Kyushu_Indoor/patched_saeed/' )
#               if file_name.split('.')[-1] == 'png' ]
# file_list.sort()



