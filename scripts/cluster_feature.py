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
new_paths = [
    u'../',
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import time
import cv2
import numpy as np
import sklearn.cluster
import matplotlib.pyplot as plt


# import place_categorization.place_categorization as plcat
# import place_categorization.plotting as pcplt
# import place_categorization.utilities as utls


################################################################################
def _lock_n_load(file_name, k_size=3):
    '''
    '''    
    image = np.flipud( cv2.imread( file_name, cv2.IMREAD_GRAYSCALE) )

    # converting to binary, for the layout-images
    thr1,thr2 = [127, 255]
    ret, image = cv2.threshold(image.astype(np.uint8) , thr1,thr2 , cv2.THRESH_BINARY)

    # erode to make the ogm suitable for raycasting    
    kernel = np.ones((k_size,k_size),np.uint8)
    image = cv2.erode(image, kernel, iterations = 3)
    image = cv2.medianBlur(image, k_size)

    return image

################################################################################
def _visualize_save(label_image, visualize=True, save_to_file=False):
    '''
    '''
    if visualize and save_to_file:
        np.save( save_to_file, label_image)

        fig, axes = plt.subplots(1,1, figsize=(10,10))
        axes.imshow(label_image, origin='lower')
        plt.tight_layout()
        plt.show()


    elif visualize:
        fig, axes = plt.subplots(1,1, figsize=(10,10))
        axes.imshow(label_image, origin='lower')
        plt.tight_layout()
        plt.show()

    if save_to_file:
        np.save( save_to_file, label_image)


################################################################################
################################################################################
################################################################################
if __name__ == '__main__':
    '''
    options
    -------
    -normalize (normalizes the features by removing bias and deviding by std)
    -visualize
    -save_to_file

    parameters
    ----------
    --n_category 2

    python cluster_feature.py --image_name ../map_sample/test_.png --n_category 2 -visualize
    '''
    args = sys.argv

    ###### fetching options from input arguments
    # options are marked with single dash
    options = []
    for arg in args[1:]:
        if len(arg)>1 and arg[0] == '-' and arg[1] != '-':
            options += [arg[1:]]

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

    if 'image_name' not in locals(): raise( StandardError('at least give an image file name!') )

    visualize = True if 'visualize' in options else False
    save_to_file = True if 'save_to_file' in options else False
 
    out_file_name = '.'.join( image_name.split('.')[:-1] ) + '_place_categories.npy'
    save_to_file = out_file_name if save_to_file==True else False
    n_category = int(n_category) if 'n_category' in locals() else 2
    normalize = True if 'normalize' in options else False


    ### loading and processing image
    # image_name = '../map_sample/kpt4a_full.png'
    image = _lock_n_load(image_name, k_size=3)

    ### loading features
    # raycasts_name = '.'.join( image_name.split('.')[:-1] ) + '_raycasts.npy'
    feature_name = '.'.join( image_name.split('.')[:-1] ) + '_features.npy'
    features = np.atleast_1d( np.load(feature_name) )[0]
    
    open_cells = features['open_cells']
    X = features['features']

    ### Normalizing
    if normalize:
        print ('\t *********** normalizing features ***********')
        X_mean = np.atleast_2d( np.mean( X, axis=1 ) ).T
        X_std = np.atleast_2d( np.std( X, axis=1 ) ).T
        X -= np.concatenate([X_mean for _ in range(X.shape[1]) ],axis=1)
        X /= np.concatenate([X_std for _ in range(X.shape[1]) ],axis=1)

    ### clustering
    if 1:
        print ('\t Kmean... ')
        clusterer = sklearn.cluster.KMeans(n_clusters=n_category,
                                           precompute_distances=False,
                                           n_init=20, max_iter=500)
    else:
        print ('\t DBSCAN... ')
        clusterer = sklearn.cluster.DBSCAN(eps=1.,
                                           min_samples = open_cells.shape[0]/10)

    clusterer.fit(X)
    labels = clusterer.labels_

    ### generating and storing labeling image
    label_image = np.ones(image.shape) *-1 # initializing every pixel as outlier
    label_image[open_cells[:,1],open_cells[:,0]] = labels

    _visualize_save(label_image, visualize, save_to_file)
