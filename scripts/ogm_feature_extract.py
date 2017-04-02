from __future__ import print_function


new_paths = [
    u'../',
    # u'/home/saesha/Dropbox/myGits/Python-CPD/'
]
for path in new_paths:
    if not( path in sys.path):
        sys.path.append( path )

import sys
import time
import cv2
import numpy as np

import multiprocessing as mp
import contextlib as ctx
from functools import partial

from core import place_categorization as plcat


def load_process_image(file_name, k_size=3):
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

def main(file_name,
         # raycasting config
         mpp = 0.02, # meter per pixel
         range_meter   = 8, # meter
         length_range  = 400, #range_meter_ / mpp_
         length_steps  = 400, #int(length_range_)
         theta_range   = 2*np.pi,
         theta_res     = 1/1, # step/degree
         occupancy_thr = 210,
         gapThreshold  = [1.0] ):

    '''
    Note
    ----
    open_cells is in (row,col) format
    '''

    ### loading and processing image
    image = load_process_image(file_name, k_size=3)
    # image = np.flipud( cv2.imread( file_name, cv2.IMREAD_GRAYSCALE) ) 

    ########################################
    ###### raycasting and feature extraction
    ########################################
    ### constructing the rays_array_xy
    pose = np.array([0,0,0]) # x,y,theta
    raxy = plcat.construct_raycast_array(pose,
                                         length_range, length_steps, 
                                         theta_range, theta_res)

    ### finding free space (unoccupied pixels) from which to raycast
    open_cells = np.transpose(np.nonzero(image>occupancy_thr)) # row, col

    ### feature extraction 
    features_extraction_parial = partial(plcat.features_extraction,
                                         image=image,
                                         rays_array_xy=raxy,
                                         mpp=mpp,
                                         range_meter=range_meter,
                                         length_range=length_range,
                                         length_steps=length_steps,
                                         theta_range=theta_range,
                                         theta_res=theta_res,
                                         occupancy_thr=occupancy_thr,
                                         gapThreshold=gapThreshold)

    tic = time.time()
    with ctx.closing(mp.Pool(processes=4)) as p:
        features = p.map( features_extraction_parial, open_cells)
    features = np.array(features)
    print (time.time()-tic)

    ### saving indices to open_cells and their corresponding features
    np.save(file_name[:-4]+'_{:s}.npy'.format('features'), features)
    np.save(file_name[:-4]+'_{:s}.npy'.format('open_cells'), open_cells)




if __name__ == '__main__':
    '''
    python ogm_feature_extract.py --file_name test.png
    '''
    args = sys.argv
    file_name = None

    # fetching parameters from input arguments
    # parameters are marked with double dash,
    # the value of a parameter is the next argument   
    listiterator = args[1:].__iter__()
    while 1:
        try:
            item = listiterator.next()
            if item[:2] == '--':
                exec(item[2:] + ' = listiterator.next()')
        except:
            break

    # if file name is not provided, set to default
    if file_name is None:
        raise (NameError('no ply file is found'))

    main(file_name)
