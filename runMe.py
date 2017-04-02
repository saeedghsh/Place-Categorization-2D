from __future__ import print_function

import sys
if sys.version_info[0] == 3:
    from importlib import reload
elif sys.version_info[0] == 2:
    pass

import time
import cv2
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt
from core import place_categorization as plcat
reload(plcat)


################################################################################
################################################################## loading image 
################################################################################
image_name = 'map_sample/E5_06.png'
image_name = 'map_sample/F5_04.png'
image_name = 'map_sample/E5_layout.png'

image = np.flipud( cv2.imread( image_name, cv2.IMREAD_GRAYSCALE) )

fig = plt.figure()
fig.add_axes([0, 0, 1, 1])
fig.axes[0].imshow( image, cmap = 'gray', interpolation='nearest', origin='lower')
plt.show()
################################################################################
############################################################### raycasting setup
################################################################################
'''
The "kpt4a_full.png" map was created with: mpp=.02 (meter per pixel)
for a raycast of range of M meter: length_range_= M/mpp
resolution of the ray is up to user
'''

# raycasting config
mpp_ = 0.02 # meter per pixel
range_meter_ = 4 # meter
length_range_  = 200 #range_meter_ / mpp_
length_steps_  = 200 #int(length_range_)
theta_range_   = 2*np.pi
theta_res_     = 1/1 # step/degree
occupancy_thr_ = 210
gapThreshold_  = [1.0]

# constructing the rays_array_xy
pose_ = np.array([0,0,0]) # x,y,theta
raxy = plcat.construct_raycast_array(pose_,
                                     length_range_, length_steps_, 
                                     theta_range_, theta_res_)

# finding free space (unoccupied pixels) from which to raycast
open_cells = np.transpose(np.nonzero(image>occupancy_thr_)) # row, col

# single processing vs. multi processing
multi_prc = [False, 4][1]


################################################################################
############################################## raycasting and feature extraction
################################################################################

if multi_prc:
    import multiprocessing as mp
    import contextlib as ctx
    from functools import partial
    features_extraction_parial = partial(plcat.features_extraction,
                                         image=image,
                                         rays_array_xy=raxy,
                                         mpp=mpp_,
                                         range_meter=range_meter_,
                                         length_range=length_range_,
                                         length_steps=length_steps_,
                                         theta_range=theta_range_,
                                         theta_res=theta_res_,
                                         occupancy_thr=occupancy_thr_,
                                         gapThreshold=gapThreshold_)

    tic = time.time()
    with ctx.closing(mp.Pool(processes=multi_prc)) as p:
        features = p.map( features_extraction_parial, open_cells)
    features = np.array(features)
    print (time.time()-tic)

else:

    tic = time.time()
    features = np.ones((open_cells.shape[0], 17))
    for idx, (row, col) in enumerate(open_cells):

        pose_ = np.array([col,row,0])

        r,t = plcat.raycast_bitmap(image, pose_,
                                   occupancy_thr_,
                                   length_range_, length_steps_, 
                                   theta_range_, theta_res_,
                                   rays_array_xy=raxy)

        features[idx,:] = plcat.raycast_to_features(t,r,
                                                    mpp=mpp_,
                                                    RLimit=range_meter_,
                                                    gapThreshold=gapThreshold_)

    print (time.time()-tic)

np.save(image_name[:-4]+'_{:s}.npy'.format('features'), features)
np.save(image_name[:-4]+'_{:s}.npy'.format('open_cells'), open_cells)


# features = np.load('map_sample/E5_06_features.npy')
# open_cells = np.load('map_sample/E5_06_open_cells.npy')


################################################################################
############################################# feature plotting and normalization
################################################################################
X = features 
ind_ = open_cells

if 0:
    # box_plot all
    fig, ax = plt.subplots()
    ax.boxplot( [X[:,i] for i in range(X.shape[1])] )
    ax.set_xlabel('feaures')
    plt.show()


########### Normalizing 
for i in range(X.shape[1]):
    X[:,i] /= X[:,i].mean()
# print (X_ == X/X.mean(axis=0))

if 0:
    #  box_plot all after "normalization"
    fig, ax = plt.subplots()
    ax.boxplot( [X[:,i] for i in range(X.shape[1])] )
    ax.set_xlabel('feaures')
    plt.show()

################################################################################
##################################################################### clustering
################################################################################
n_categories = 2
kmean = sklearn.cluster.KMeans(n_clusters=n_categories,
                               precompute_distances=False,
                               n_init=20, max_iter=500).fit(X)
labels = kmean.labels_


# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
print('number of estimated clusters: {:d}'.format(n_clusters_) )

# Size of each cluster:
msg = '\t cluster {:d}: {:d} instances'
for label in list(set(labels)):
    print (msg.format(label, np.count_nonzero(labels==label)))

################################################################################    
###################################################### Plotting place categories
################################################################################
## creating a colored image, according to labels
colors = plt.cm.Spectral(np.linspace(0, 1, len(set(labels))))
ogm_label = np.ones((image.shape[0],image.shape[1],3))
for i in range(len(labels)):
    if labels[i] == -1:
        ogm_label[ind_[i][0],ind_[i][1]] = np.array([0.,0.,0.])
    else:
        ogm_label[ind_[i][0],ind_[i][1]] = colors[labels[i]][0:3]

if 1:
    import scipy.misc
    scipy.misc.imsave(image_name[:-4]+'_labels_km{:s}.png'.format(str(n_categories)), np.flipud(ogm_label) )


fig, ax = plt.subplots()
ax.imshow(ogm_label, interpolation='nearest', origin='lower')
plt.tight_layout()
plt.show()
