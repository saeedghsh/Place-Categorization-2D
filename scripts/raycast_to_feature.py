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

import multiprocessing as mp
import contextlib as ctx
from functools import partial

import place_categorization.place_categorization as plcat
import place_categorization.plotting as pcplt
import place_categorization.utilities as utls
reload(utls)

################################################################################
def lock_n_load(file_name, k_size=3):
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
################################################################################
################################################################################
if __name__ == '__main__':
    '''
    python raycast_map.py --image_name ../map_sample/test_.png
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

    # ### 
    # multiprocessing = 4

    ### loading and processing image
    # image_name = '../map_sample/kpt4a_full.png'
    # image_name = '../map_sample/test_.png'
    image = lock_n_load(image_name, k_size=3)
    raycasts_name = '.'.join( image_name.split('.')[:-1] ) + '_raycasts.npy'
    raycasts = np.atleast_1d( np.load(raycasts_name) )[0]

    ### raycasting parametere
    raycast_config = raycasts['config']
    open_cells = raycasts['open_cells']
    R = raycasts['range_vecs']
    t = raycasts['theta_vecs']

    # idx = np.random.randint(open_cells.shape[0])
    # pcplt.plot_ray_cast(image, open_cells[idx,:], R[idx,:], T[idx,:])       

    print('\t **************** TODO ****************')
    print('\t what is the metric of R?')
    print('\t what is the threshold for relative gap in A1?')
    print('\t what is the EFD_coefficient_degree in A2?')
    print('\t what to do with complext values in A2?')
    print('\t **************************************')

    tic = time.time()
    # if memory is short, do one by one. But the function expects 2d array for R, so it goes like:
    # F = plcat.feature_set_A1(np.atleast_2d(R[0,:]), t, [50, 100, 150], 0.5)
    FA1 = plcat.feature_set_A1(R, t, [50, 100, 150], 0.5)
    print ('time to extract features (A1) for {:d} opencells: {:f}'.format(open_cells.shape[0],time.time()-tic))

    tic = time.time()
    FA2 = plcat.feature_set_A2(R, t, EFD_degree=10)
    print ('time to extract features (A2) for {:d} opencells: {:f}'.format(open_cells.shape[0],time.time()-tic))

    features = {
        'open_cells': open_cells,
        'features': np.concatenate( (FA1, FA2), axis=1)
    }

    output_name = '.'.join( image_name.split('.')[:-1] ) + '_featurs.npy'
    np.save(output_name, features)
