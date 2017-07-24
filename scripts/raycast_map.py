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

################################################################################
def lock_n_load(file_name, occupancy_thr=220, k_size=3):
    '''
    '''    
    image = np.flipud( cv2.imread( file_name, cv2.IMREAD_GRAYSCALE) )

    # converting to binary, for the layout-images
    thr1,thr2 = [occupancy_thr, 255]
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
    python raycast_map.py --image_name ../map_sample/test.png
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

    ### raycasting parameters
    # NOTE: metric resolution of raycast (~accuracy) is range_res*mpp, e.g range_res=5, mpp=.02 -> 0.1m 
    raycast_config = {
        'mpp': 0.02, # meter per pixel
        'range_meter': 4, # meter
        'range_res': 1//1, # pixel per point  (must be > 1, otherwise doesn't make much sense)
        'theta_range': 2*np.pi,
        'theta_res': 1//1, # step/degree
        'occupancy_thr': 220,
    }
    raycast_config['length_range'] = raycast_config['range_meter'] // raycast_config['mpp'] # in pixels
    raycast_config['length_steps'] = raycast_config['length_range'] // raycast_config['range_res'] # 

    ### loading and processing image
    # image_name = '../map_sample/test.png'
    image = lock_n_load(image_name, occupancy_thr=raycast_config['occupancy_thr'], k_size=3)

    ### constructing the rays_array_xy template
    raxy = plcat.construct_raycast_array( np.array([0,0]),
                                          raycast_config['length_range'],
                                          raycast_config['length_steps'],
                                          raycast_config['theta_range'],
                                          raycast_config['theta_res'] )
    
    ### finding free space (unoccupied pixels) from which to raycast
    open_cells = np.transpose(np.nonzero(image>raycast_config['occupancy_thr'])) # [(row, col),..]
    open_cells = np.roll(open_cells, 1, axis=1) # [(col, row),..]

    ### type casting
    raxy = raxy.astype(np.int16)
    open_cells = open_cells.astype(np.int16)

    # pcplt.plot_rays_arr_xy(image, np.array([0,0]), raycast_config, raxy)

    # import matplotlib.pyplot as plt
    # plt.imshow(image)
    # plt.show()

    ### partialing raycast_method and leave out only pose argument
    raycast_bitmap_par = partial(plcat.raycast_bitmap,
                                 image=image,
                                 occupancy_thr=raycast_config['occupancy_thr'],
                                 length_range=raycast_config['length_range'],
                                 length_steps=raycast_config['length_steps'],
                                 mpp=raycast_config['mpp'],
                                 range_res=raycast_config['range_res'],
                                 theta_range=raycast_config['theta_range'],
                                 theta_res=raycast_config['theta_res'],
                                 rays_array_xy=raxy)
    
    ### raycasting with multi-processing
    multiprocessing = mp.cpu_count()
    tic = time.time()
    with ctx.closing(mp.Pool(processes=multiprocessing)) as p:
        r_t_s = p.map( raycast_bitmap_par, open_cells)
    print ('time to raycast "{:s}" with {:d} opencells: {:f}'.format(image_name,open_cells.shape[0],time.time()-tic))

    ### saving indices to open_cells, their corresponding raycasts, and raycast config to file
    # the type casting here is not necessary since it is already done in raycast_bitmap,
    # but it's once and not too much, so let's be on the safe side?
    t = r_t_s[0][1].astype(np.float16) # since all theta vectors are the same, save onle one of them
    R = np.array([rt[0] for rt in r_t_s], dtype=np.float16)
    del r_t_s
    raycasts = {
        'config': raycast_config,
        'open_cells': open_cells,
        'theta_vecs': t,
        'range_vecs': R,
    }
    del R, t, raycast_config, open_cells

    output_name = '.'.join( image_name.split('.')[:-1] ) + '_raycasts.npy'
    np.save(output_name, raycasts)

    # ### plot raycast from a random cell
    # import place_categorization.plotting as pcplt
    # idx = np.random.randint(raycasts['open_cells'].shape[0])
    # pcplt.plot_ray_cast(image,
    #                     raycasts['open_cells'][idx,:],
    #                     raycasts['config'],
    #                     raycasts['range_vecs'][idx,:],
    #                     raycasts['theta_vecs'])


