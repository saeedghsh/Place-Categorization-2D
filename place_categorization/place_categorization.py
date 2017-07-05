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

import utilities

################################################################################
def construct_raycast_array(pose=[0,0],
                            length_range=30, length_steps=30, 
                            theta_range=2*np.pi, theta_res=1/1):
    '''
    This method constructs a 3d array as a discrete representiion of raycast.
    The output of this method is used in raycasting.

    Note that this method could be (and will be if not available) constructed
    in raycast_bitmap method, but this is as much time consuming as the raycast
    itself. By providing a template and making a copy of it in raycast_bitmap
    method, the speed would improve almost two-fold.
    

    Input:
    ------
    pose: the location of the sensor [x,y] (default: [0,0])

    Parameters:
    -----------
    length_range: the length of a ray in pixel (default: 30)
    length_steps: number of steps in every ray (default: same as length_range)
    theta_range: the field of view of scanner in radian (default: 2pi)
    theta_res: resolution of the the angle [step/deg] (default: 1))
    
    Output:
    -------
    rays_array_xy: a 3d array of coordinated of all point for raycast
    rays_array_xy.shape = (2, theta_range, lngth_range)

    Note on indexing the output:
    ----------------------------
    >>> rays_array_xy[0,:,:] (and rays_array_xy[1,:,:])
    meshgrid of x-values (and y-values) for all points
    rays_array_xy[0,:,:].shape=(theta_range, lngth_range)

    >>> rays_array_xy[0,i,:] (and rays_array_xy[1,i,:])
    x-value (and y-value) of the ray at angle theta[i]
    rays_array_xy[0,i,:].shape=(length_range,)

    >>> rays_array_xy[0,:,i] (and rays_array_xy[1,:,i])
    x-value (and y-value) of points of all angles with a length[i] distance from center.
    increasing the index 'i' moves to out concentric circle.
    rays_array_xy[0,:,i].shape=(theta_range,)

    Note:
    -----
    The field of view is equal to the interval:
    [-theta_range/2 theta_range/2]

    Note:
    -----
    The output rays_array_xy contains coordinates to points in the map for raycasting.
    Before indexing the map with rays_array_xy, it must be cast from float to integer.
    It is not converted here, to avoid error of type cast for visualization.

    Note:
    -----
    If the ray cast is not for simulating a robot's sensor,
    for instance if the target is feature extraction towards place categorization,
    it is more convinient (and slightly faster) to only contruct the rays_array_xy once,
    with pose=[0,0] and at every new location just change it by:
    >>> rays_array_xy[0,:,:] += pose[0]
    >>> rays_array_xy[1,:,:] += pose[1]
    '''
    
    theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )

    # creating mesh grids along two axes: theta and length
    length = np.linspace(0, length_range, length_steps+1, endpoint=True)[1:]
    theta = np.linspace(0, theta_range, theta_steps, endpoint=False)
    rays_l, rays_t = np.meshgrid(length, theta)
    # rays_l.shape:(theta_range, length_range)
    # ->[[1., 2., ..., 30.], ..., [1., 2., ..., 30.]]
    # rays_t.shape:(theta_range, lngth_range)
    # ->[[0., 0., ..., 0.], ..., [2pi, 2ip, ..., 2pi]]


    # computing the coordinate of points 
    rays_array_xy = np.stack((rays_l, rays_l), axis=0)
    rays_array_xy *= np.stack((np.cos(rays_t), np.sin(rays_t)), axis=0)
    # if np.__version__ <= 1.10.0:
    #     rays_array_xy = np.array(np.vsplit(np.concatenate((rays_l, rays_l), axis=0),2))
    #     rays_array_xy *= np.array(np.vsplit(np.concatenate((np.cos(rays_t), np.sin(rays_t)), axis=0),2))

    # adjusting the points' coordinate wrt the pose 
    rays_array_xy[0,:,:] += pose[0]
    rays_array_xy[1,:,:] += pose[1]
    
    return rays_array_xy

################################################################################
def raycast_bitmap_batch(open_cells, image,
                         occupancy_thr=127,
                         length_range=30,
                         length_steps=30, 
                         theta_range=2*np.pi,
                         theta_res=1/1,
                         rays_array_xy=None):
    '''
    This method takes a bitmap iamge and returns a raycast from the specided pose

    Input:
    ------
    image:
    An occupancy grid map (bitmap image), where the value of open cells is
    max (255), and min (0) for occupied cells.
    
    open_cells
    
    rays_array_xy:
    see the output of "construct_raycast_array" for more details

    Parameters:
    -----------
    occupancy_thr:
    a pixel with value less-equal to this threshold is considered as occupied (default: 127)

    length_range:
    the length of a ray in pixel (default: 30)

    length_steps:
    number of steps in every ray (default: same as length_range)

    theta_range:
    the field of view of scanner in radian (default: 2pi)

    theta_res:
    resolution of the the angle [step/deg] (default: 1))


    Output:
    -------
    t:
    this array stores the value for the angle of each ray
    shape=(theta_steps,) where: 
    theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )
    

    Note:
    -----
    It's a safe practice to set the parameters once and use the same values
    for this method and the "construct_raycast_array" method.

    Note:
    -----
    If the "rays_array_xy" is provided, the heading is assumed to be zero,
    and only the location is adjusted wrt "pose".
    If the heading is not zero, don't provide the "rays_array_xy" and let
    this method construct it inside, which will take into account the heading.
    '''

    all_inside = True
    all_inside = all_inside & np.all(0<=open_cells[:,0]) & np.all(open_cells[:,0]<image.shape[1])
    all_inside = all_inside & np.all(0<=open_cells[:,1]) & np.all(open_cells[:,1]<image.shape[0])
    if not all_inside:
        raise Exception('open cells out of image!!!')
 
    # stack the ray array 
    rays_arr_xy = np.stack([rays_array_xy for _ in range(open_cells.shape[0])], axis=0)

    # adjust the location
    rays_arr_xy[:,0,:,:] += np.swapaxes( np.atleast_3d(open_cells[:,0]),axis1=0, axis2=1)
    rays_arr_xy[:,1,:,:] += np.swapaxes( np.atleast_3d(open_cells[:,1]),axis1=0, axis2=1)

    # fixing the indices wrt image size
    # making sure all points in the rays_arr_xy are inside the image
    x_min = np.zeros(rays_arr_xy.shape[2:4])
    y_min = np.zeros(rays_arr_xy.shape[2:4])
    x_max = (image.shape[1]-1) *np.ones(rays_arr_xy.shape[2:4])
    y_max = (image.shape[0]-1) *np.ones(rays_arr_xy.shape[2:4])

    rays_arr_xy[:,0,:,:] = np.where(rays_arr_xy[:,0,:,:] > x_min, rays_arr_xy[:,0,:,:], x_min)
    rays_arr_xy[:,0,:,:] = np.where(rays_arr_xy[:,0,:,:] < x_max, rays_arr_xy[:,0,:,:], x_max)

    rays_arr_xy[:,1,:,:] = np.where(rays_arr_xy[:,1,:,:] > y_min, rays_arr_xy[:,1,:,:], y_min)
    rays_arr_xy[:,1,:,:] = np.where(rays_arr_xy[:,1,:,:] < y_max, rays_arr_xy[:,1,:,:], y_max)

    # finding the values in the image, corresponding to each point
    rays_image_val = image[rays_arr_xy[:,1,:,:].astype(int),
                           rays_arr_xy[:,0,:,:].astype(int)]

    # making sure there is at least one occupied pixels per ray
    # it is set at the end, so it is equivalent to the range limit
    rays_image_val[:,:,-1] = 0

    # return rays_image_val


    theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )
    t = np.linspace(0, theta_range, theta_steps, endpoint=False)


    # nonzero only returns the indices to entries that satisfy the condition
    # this means t_idx is not uniform, different angles (rays) have different nubmer of nonzeros
    # this is why reshape won't work!
    # instead the change from one ray to the next is detected by the change of angle index (t_idx)
    # i.e. np.nonzero(np.diff(t_idx)!=0)[0]+1
    # since this ignores the first ray (no change of index), it is added manually at first: r_idx[0]
    cell_idx, t_idx , r_idx = np.nonzero(rays_image_val<=occupancy_thr)
    # r_idx = r_idx.reshape(open_cells.shape[0], theta_steps, length_steps)[:,:,0]
    r_idx = np.concatenate((np.atleast_1d(r_idx[0]), r_idx[np.nonzero(np.diff(t_idx)!=0)[0]+1]))
    r_idx = r_idx.reshape(open_cells.shape[0], theta_steps)

    # scaling (converting) range values from index to distance
    R = r_idx * float(length_range)/length_steps

    return R,t


################################################################################
def raycast_bitmap(pose, image,
                   occupancy_thr=127,
                   length_range=30,
                   length_steps=30, 
                   theta_range=2*np.pi,
                   theta_res=1/1,
                   rays_array_xy=None):
    '''
    This method takes a bitmap iamge and returns a raycast from the specided pose

    Input:
    ------
    image:
    An occupancy grid map (bitmap image), where the value of open cells is
    max (255), and min (0) for occupied cells.
    
    pose:
    the location the sensor [x,y]
    
    rays_array_xy:
    see the output of "construct_raycast_array" for more details

    Parameters:
    -----------
    occupancy_thr:
    a pixel with value less-equal to this threshold is considered as occupied (default: 127)

    length_range:
    the length of a ray in pixel (default: 30)

    length_steps:
    number of steps in every ray (default: same as length_range)

    theta_range:
    the field of view of scanner in radian (default: 2pi)

    theta_res:
    resolution of the the angle [step/deg] (default: 1))


    Output:
    -------
    t:
    this array stores the value for the angle of each ray
    shape=(theta_steps,) where: 
    theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )

    r:
    distance to the first occupied cell in ray
    shape=(theta_steps,) where: 
    theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )
    

    Note:
    -----
    It's a safe practice to set the parameters ones and use the same values
    for this method and the "construct_raycast_array" method.


    Note:
    -----
    If the "rays_array_xy" is provided, the heading is assumed to be zero,
    and only the location is adjusted wrt "pose".
    If the heading is not zero, don't provide the "rays_array_xy" and let
    this method construct it inside, which will take into account the heading.
    '''

    if not(0<=pose[0]<image.shape[1]) or not(0<=pose[1]<image.shape[0]):        
        # print ( image.shape, pose )
        raise Exception('pose is out of map')
 
    if rays_array_xy is None:
        # if the rays array is not provided, construct one
        rays_arr_xy = construct_raycast_array(pose,
                                              length_range, length_steps, 
                                              theta_range, theta_res)
    else:
        # if the rays array is provided, adjust the location        
        rays_arr_xy = rays_array_xy.copy()
        # moving rays_array wrt new pose
        rays_arr_xy[0,:,:] += pose[0]
        rays_arr_xy[1,:,:] += pose[1]

    # fixing the indices wrt image size
    # making sure all points in the rays_arr_xy are inside the image
    x_min = np.zeros(rays_arr_xy.shape[1:])
    y_min = np.zeros(rays_arr_xy.shape[1:])
    x_max = (image.shape[1]-1) *np.ones(rays_arr_xy.shape[1:])
    y_max = (image.shape[0]-1) *np.ones(rays_arr_xy.shape[1:])

    rays_arr_xy[0,:,:] = np.where(rays_arr_xy[0,:,:] > x_min, rays_arr_xy[0,:,:], x_min)
    rays_arr_xy[0,:,:] = np.where(rays_arr_xy[0,:,:] < x_max, rays_arr_xy[0,:,:], x_max)

    rays_arr_xy[1,:,:] = np.where(rays_arr_xy[1,:,:] > y_min, rays_arr_xy[1,:,:], y_min)
    rays_arr_xy[1,:,:] = np.where(rays_arr_xy[1,:,:] < y_max, rays_arr_xy[1,:,:], y_max)

    # finding the values in the image, corresponding to each point
    rays_image_val = image[rays_arr_xy[1,:,:].astype(int),
                           rays_arr_xy[0,:,:].astype(int)]

    # making sure there is at least one occupied pixels per ray
    # it is set at the end, so it is equivalent to the range limit
    rays_image_val[:,-1] = 0


    theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )
    t = np.linspace(0, theta_range, theta_steps, endpoint=False)
    
    # new way ~ 2.5715110302 Second for 1000 raycast
    # nonzero only returns the indices to entries that satisfy the condition
    # this means t_idx is not uniform, different angles (rays) have different nubmer of nonzeros
    # this is why reshape won't work!
    # instead the change from one ray to the next is detected by the change of angle index (t_idx)
    # i.e. np.nonzero(np.diff(t_idx)!=0)[0]+1
    # since this ignores the first ray (no change of index), it is added manually at first: r_idx[0]
    t_idx , r_idx = np.nonzero(rays_image_val<=occupancy_thr)
    r_idx = np.concatenate((np.atleast_1d(r_idx[0]), r_idx[np.nonzero(np.diff(t_idx)!=0)[0]+1]))
    # old way ~3.64616513252 Second for 1000 raycast
    # r_idx = np.array([ np.nonzero(ray<=occupancy_thr)[0][0] for ray in rays_image_val ]).astype(float)
    
    # scaling (converting) range values from index to distance
    r = r_idx * float(length_range)/length_steps

    return r,t

################################################################################
def fit_with_gaussian(data):
    '''
    <<GAUSSIAN DISTRIBUTION ESTIMATION>>

    Input:
    ------
    input to this class is a numpy.ndarray with dimensions NxD, where:
    N= number of sample points
    D= space dimension of distribution

    Outputs:
    --------
    FitGaussian.mean      (1xD) numpy.ndarray representing center of the distribution (ie. estimated gaussian).
    FitGaussian.stddev    (1xD) numpy.ndarray representing standard deviation of the distribution in direction of PCAs.
    FitGaussian.rotation  (DxD) numpy.ndarray representing the orientation of estimated gaussian.    
    '''

    N = data.shape[0]
    Dimension = data.shape[1]

    # Compute the center of gravity = center of the gaussian
    mean = np.mean(data,0)

    # Translate the distribution to center its CoG at origin of the 
    zero_mean_data = np.float64(data.copy())
    for i in range(Dimension):
        zero_mean_data[:,i] -= mean[i]
        
    # Compute the singular value decomposition
    U,S,V = np.linalg.svd(zero_mean_data)
        
    stddev = S / np.sqrt(N)
    rot = V
            
    return mean, stddev, rot

################################################################################
def feature_set_A1(R,t, gap_t, rel_gap_t):
    '''
    Statistical Feature Set
    -----------------------
    Feature set A1 from "Semantic labeling of places with mobile robots"
    For details type `feature_set_by_oscar_description()`

    Inputs
    ------
    t: 1darray (1xn)
    n is the is the number of rays in each raycast, or the number of angles
    
    R: 2darray (mxn)
    m is the number of open_cells and hence the number of raycast vectors
    n is the number of rays in each raycast (i.e. the length of theta vector)

    Paramters
    ---------
    gap_t: list
    gap threshold (if di-dj > gap_t[k])
   
    rel_gap_t: scaler (0,1)
    relateive gap threshold (if di/dj > gap_t[k])
    
    Output
    ------
    F: 2darray (mxl)
    m is the number of open_cells and hence the number of raycast vectors
    l is the length of fature vector (it depends on feature set)

    '''

    assert R.ndim == 2

    diff = np.abs( np.diff(R, axis=1) )
    # Average Difference Between the Length of Two Consecutive Beams
    A11 = np.atleast_2d( diff.mean(axis=1) ).T
    # Standard Deviation of the Difference Between the Length of Two Consecutive Beams
    A12 = np.atleast_2d( diff.std(axis=1) ).T

    # since we use raycast and not real laser beams, the max-range is already included
    # # Average Difference Between the Length of Consecutive Beams Considering Max-Range
    # A13 =
    # # Standard Deviation of the Difference Between the Length of Two Consecutive Beams Considering Max-Range
    # A14 =

    # The Average Beam Length
    A15 = np.atleast_2d( R.mean(axis=1) ).T
    # The Standard Deviation of the Beam Length
    A16 = np.atleast_2d( R.std(axis=1) ).T

    # Number of Gaps
    A17 = np.concatenate( [ np.atleast_2d( np.count_nonzero(diff>gt, axis=1) ).T
                            for gt in gap_t ],
                          axis=1 )

    # For two reason I won't include this, 1) too much work, 2) I don't the point of it anyway!
    # # Number of Beams Lying on Lines Extracted from the Range
    # A18 = method from \cite{sack2004comparison}

    # for door frame detection
    # filt = np.vstack( np.hannig(window_size), R.shape[0])
    # R_smooth = np.apply_along_axis(lambda m: np.convolve(m, filt, mode='full'), axis=1, arr=R)
    # Global_Minima = 
    # # Euclidean Distance Between the Two Points Corresponding to Two Consecutive Global Minima
    # A19 =
    # # The Angular Distance Between the Two Points Corresponding to Two Consecutive Global Minima
    # A110 =

    rel = R / np.roll(R, 1, axis=1)
    # Average of the Relation Between Two Consecutive Beams
    A111 = np.atleast_2d( rel.mean(axis=1) ).T
    # Standard Deviation of the Relation Between the Length of Two Consecutive Beams
    A112 = np.atleast_2d( rel.std(axis=1) ).T

    nrm = R / np.atleast_2d( R.max(axis=1) ).T
    # Average of Normalized Beam Length
    A113 = np.atleast_2d( nrm.mean(axis=1) ).T
    # Standard Deviation of Normalized Beam Length
    A114 = np.atleast_2d( nrm.std(axis=1) ).T

    # Number of Relative Gaps
    A115 = np.atleast_2d( np.count_nonzero( rel > rel_gat_t, axis=1) ).T
    # todo: since it\'s relative, shouldn\'t the condition be ((1/rel_gap_t)< rel <rel_gap_t)?

    # Kurtosis
    A116 = np.atleast_2d( (np.sum((R-A15)**4,axis=1) / (R.shape[1]* A16**4))-3 ).T

    F = np.concatenate( (A11,A12, A15,A16, A17, A111,A112, A113,A114, A115, A116),
                        axis=1)


    return F

################################################################################
def feature_set_A2(R,t):
    '''
    Shape Feature Set
    -----------------
    Feature set A2 from "Semantic labeling of places with mobile robots"
    For details type `feature_set_by_oscar_description()`

    Inputs
    ------
    t: 1darray (1xn)
    n is the is the number of rays in each raycast, or the number of angles
    
    R: 2darray (mxn)
    m is the number of open_cells and hence the number of raycast vectors
    n is the number of rays in each raycast (i.e. the length of theta vector)
   

    Output
    ------
    F: 2darray (mxl)
    m is the number of open_cells and hence the number of raycast vectors
    l is the length of fature vector (it depends on feature set)
    '''

    # # Area of P(z)
    # A21 = 

    # # Perimeter of P(z)
    # A22 = 

    # # Mean Distance Between the Centroid and the Shape Boundary
    # A23 =

    # # Standard Deviation of the Distances Between the Centroid and the Shape Boundary
    # A24 = 

    # # Invariant Descriptors Based on the Fourier Transformation
    # A25 = 

    # # Major Axis Ma of the Ellipse that Approximates P(z)
    # A26 = 

    # # Minor Axis Mi of the Ellipse that Approximates P(z)
    # A27 = 

    # # Invariant Moments of P(z)
    # A28 = 

    # # Normalized Feature of Compactness of P(z)
    # A29 = 

    # # Normalized Feature of Eccentricity of P(z)
    # A210 = 

    # # Form Factor of P(z)
    # A211 = 

    # # Circularity of P(z)
    # A212 = 

    # # Normalized Circularity of P(z)
    # A213 = 

    # # Average Normalized Distance Between the Centroid and the Shape Boundary
    # A214 = 
    
    # # Standard Deviation of the Normalized Distances Between the Centroid and the Shape Boundary
    # A215 = 

    # F = np.concatenate( (),
    #                     axis=1)

    return F

################################################################################
def feature_set_mix(R,t):
    '''
    A feature set from "Semantic labeling of places with mobile robots" (See note below)

    Inputs
    ------
    t: 1darray (1xn)
    n is the is the number of rays in each raycast, or the number of angles
    
    R: 2darray (mxn)
    m is the number of open_cells and hence the number of raycast vectors
    n is the number of rays in each raycast (i.e. the length of theta vector)
   
    Output
    ------
    F: 2darray (mxl)
    m is the number of open_cells and hence the number of raycast vectors
    l is the length of fature vector (it depends on feature set)

    Note
    ----
    {:s}
    '''#.format(feature_set_by_oscar(print_out=False))

    A1 = feature_set_A1(R)
    A2 = feature_set_A2(R)
    F = np.concatenate( (A1,A2), axis=1 )

    return F

################################################################################
def raycast_to_features(t,r,
                        mpp, RLimit,
                        gapThreshold= [1.0]):
    '''
    Insipred by [1] this method delivers a feature vector that is a descriptor
    for the shape of the surrounding, represented by the raycast in input

    Inputs:
    -------
    t:
    r:
    

    Parameters:
    -----------
    mpp: meter per pixel
    RLimit:
    gapThreshold:
    default: [1.0]
    
    Output:
    -------

    Note:
    -----
    Elements of the feature vectore: 
    1. Euclidean distance between the two points corresponding to the two smallest local minima.
    2. The angular distance between the beams corresponding to the local minima in previous.
    3. 200 similarity invariant descriptors based in the Fourier transformation.
    4. Major axis Ma of the ellipse that approximates P(z) using the first two Fourier coefficients.
    5. Minor axis Mi of the ellipse that approximate P(z) using the first two Fourier coefficients.
    6. Ma/Mi.
    7. Seven invariants calculated from the central moments of P(z).
    8. Normalized feature of compactness of P(z).
    9. Normalized feature of eccentricity of P(z).
    10. Form factor of P(z)

    [1] Mozos, O. Martinez, and Wolfram Burgard.
    "Supervised learning of topological maps using semantic information extracted from range data."
    Intelligent Robots and Systems, 2006 IEEE/RSJ International Conference on. IEEE, 2006.
    [2] R.M. Haralick and L.G. Shapiro. Computer and Robot Vision. Addison-Wesley Publishing Inc., 1992.
    [3] S. Loncaric. A survey of shape analysis techniques. Pattern Recognition,31(8), 1998.
    [4] R.C. Gonzalez and P. Wintz. Digital Image Processing. Addison-Wesley Publishing Inc., 1987.
    '''    
    f = [] # the problem is the dynamic number of gaps np.zeros(17)
    # [ r_mean_norm,
    #   r_stdv_norm,
    #   diff_mean_norm,
    #   diff_stdv_norm,    
    #   A, P, C,
    #   [gap_count]
    #   kurtosis,
    #   [svdStddev],
    #   svdStddev[0]/svdStddev[1],
    #   np.sqrt(svdMean[0]**2+svdMean[1]**2),
    #   [svdStddev]
    #   svdStddev[0]/svdStddev[1]
    #   np.sqrt(svdMean[0]**2+svdMean[1]**2)
    # ]
    

    ###### normalized mean and std of ranges
    r_mean = np.mean(r)
    r_stdv = np.sqrt( np.sum((r-r_mean)**2) /len(r) )
    r_mean_norm = r_mean / RLimit
    r_stdv_norm = r_stdv / RLimit
    f.append(r_mean_norm), f.append(r_stdv_norm)

    ###### normalized mean and std of differences of ranges    
    diff_mean = np.mean(np.abs(np.diff(r)))
    diff_stdv = np.sqrt( np.sum((np.abs(np.diff(r))-diff_mean)**2) /len(r) )
    diff_mean_norm = diff_mean / RLimit
    diff_stdv_norm = diff_stdv / RLimit
    f.append(diff_mean_norm), f.append(diff_stdv_norm)
    
    ###### 
    dt = t[1]-t[0]
    # p = np.zeros(len(r))
    # a = np.zeros(len(r))
    # for i in range(len(r)):
    #     r1,r2 = r[i-1], r[i]
    #     r3,r4 = r1*np.sin(dt), r1*np.cos(dt)
    #     p[i] = np.sqrt(r3**2 + (r2-r4)**2)
    #     a[i] = r2*r3/2.0
    p = np.sqrt((np.roll(r,1)*np.sin(dt))**2 + (r-np.roll(r,1)*np.cos(dt))**2)    
    a = r *np.roll(r,1) *np.sin(dt) /2.0

    A = np.sum(a)
    P = np.sum(p)
    C = P**2/A
    f.append(A), f.append(P), f.append(C)

    ###### counting thresholds
    for gt in gapThreshold:
        gap_count = len(np.nonzero(np.abs(np.diff(r))>gt)[0])
        f.append(gap_count)
    
    ###### the kurtosis
    if np.abs(r_stdv) < np.spacing(10**10):
        f.append( 0 )
    else:
        kurtosis = np.sum((r - r_mean)**4) / (len(r)*r_stdv**4) #-3
        f.append(kurtosis)
        
    ###### conversion of raycasts  ######
    ###### from polar to cartesian ######
    endpoints =  (1/mpp) * np.array( [r*np.cos(t), r*np.sin(t)] ).T

    # resampling the endpoints - rejecting close points
    # dis_mat = get_distance_matrix(endpoints)
    # pts_idx = np.arange(endpoints.shape[0])
    # dis_mat[pts_idx,pts_idx] = dis_mat.max(axis=0)    
    endpoints_resampled = list(endpoints)
    for i in range(len(endpoints)-1,0,-1):
        p1,p2, = endpoints_resampled[i],endpoints_resampled[i-1]
        if np.sqrt( (p1[0]-p2[0])**2 + (p1[1]-p2[1])**2  ) < 0.5:#(1.0*mpp):
            endpoints_resampled.pop(i)        
    endpoints_resampled = np.array(endpoints_resampled)
    #####################################
    #####################################
    
    ###### parameters of the gaussian model of the endpoints 
    svdMean, svdStddev, svdRot = fit_with_gaussian(endpoints)
    if len( svdStddev ) == 2:
        f.extend( svdStddev ) # principle components
        f.append( svdStddev[0]/svdStddev[1] ) # ratio between principle components
        f.append( np.sqrt(svdMean[0]**2+svdMean[1]**2) ) # the bias of the center
    else:
        f.extend( [0,0,0,0] )
    ###### parameters of the gaussian model of the resampled_endpoints 
    svdMean, svdStddev, svdRot = fit_with_gaussian(endpoints_resampled)
    if len( svdStddev ) == 2:
        f.extend( svdStddev ) # principle components
        f.append( svdStddev[0]/svdStddev[1] ) # ratio between principle components
        f.append( np.sqrt(svdMean[0]**2+svdMean[1]**2) ) # the bias of the center
    else:
        f.extend( [0,0,0,0] )

    ###### TODO:
    # ellipse shape stuff

    return np.array(f)

################################################################################
def features_extraction(row_col,
                        image,
                        rays_array_xy,
                        mpp,
                        range_meter,
                        length_range,
                        length_steps,
                        theta_range,
                        theta_res,
                        occupancy_thr,
                        gapThreshold):
    '''
    This method takes the pose (row,col), an image and some other pameters,
    returns the feature descriptor for that pose. It's binds together the
    following methods:
    raycast_bitmap    
    raycast_to_features

    It will become partial, for the purpose of multi-processing.
    '''
    pose = np.array([row_col[1],row_col[0]])
    r,t = raycast_bitmap(pose,
                         image,
                         occupancy_thr,
                         length_range,
                         length_steps, 
                         theta_range,
                         theta_res,
                         rays_array_xy)

    features = raycast_to_features(t,r,
                                   mpp,
                                   range_meter,
                                   gapThreshold)
    
    return features



################################################################################
################################################################################
################################################################################

# ################################################################################
# def get_distance_matrix(points):
#     '''
#     This is useless now!
#     even if need could use: scipy.spatial.distance.cdist instead
    
#     Returns a matrix "distances" that is the distance between pairs of points
    
#     Input:
#     ------
#     points:
#     a 2d numpy array of N points coordinates (Nx2)

#     Output:
#     -------
#     a 2d numpy array of of distances between points (NxN)
#     '''
#     xh = np.repeat( [points[:,0]], points.shape[0], axis=0)
#     xv = np.repeat( [points[:,0]], points.shape[0], axis=0).T
#     dx = xh - xv
    
#     yh = np.repeat( [points[:,1]], points.shape[0], axis=0)
#     yv = np.repeat( [points[:,1]], points.shape[0], axis=0).T
#     dy = yh - yv
    
#     distances = np.sqrt( dx**2 + dy**2)

#     return distances

# ################################################################################
# def raycast_pointcloud(pointcloud, pose,
#                        dist_thr,
#                        length_range, length_steps, 
#                        theta_range, theta_res,
#                        rays_array_xy=None):
#     '''
#     This method takes a pointcloud and returns a raycast from the specided pose

#     Input:
#     ------
#     pointcloud:
    
#     pose:
#     the location of the sensor [x,y]

#     rays_array_xy:
#     see the output of "construct_raycast_array" for more details


#     Parameters:
#     -----------
#     dist_thr:
#     if the distance between a point in  raycast and a point from pointcloud is 
#     less than this threshold, the point is considered occupided


#     Output:
#     -------
#     t:
#     this array stores the value for the angle of each ray
#     shape=(theta_steps,) where: 
#     theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )

#     r:
#     distance to the first occupied cell in ray
#     shape=(theta_steps,) where: 
#     theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )
    

#     Note:
#     -----
#     It's a safe practice to set the parameters once and use the same values
#     for this method and the "construct_raycast_array" method.

#     Note:
#     -----
#     If the "rays_array_xy" is provided, the heading is assumed to be zero,
#     and only the location is adjusted wrt "pose".
#     If the heading is not zero, don't provide the "rays_array_xy" and let
#     this method construct it inside, which will take into account the heading.
#     '''
 
#     if rays_array_xy is None:
#         # if the rays array is not provided, construct one
#         rays_arr_xy = construct_raycast_array(pose,
#                                               length_range, length_steps, 
#                                               theta_range, theta_res)
#     else:
#         # if the rays array is provided, adjust the location        
#         rays_arr_xy = rays_array_xy.copy()
#         # moving rays_array wrt new pose
#         rays_arr_xy[0,:,:] += pose[0]
#         rays_arr_xy[1,:,:] += pose[1]


#     # finding the min distance between raycasts and pointcloud
#     x_ = rays_array_xy[0,:,:].flatten()
#     y_ = rays_array_xy[1,:,:].flatten()
#     rc = np.stack( (x_, y_), axis=1)    

#     dist = scipy.spatial.distance.cdist(rc, pointcloud, 'euclidean') # shape = (rc x pointcloud)
#     dist = np.min(dist, axis=1)
#     dist = dist.reshape( rays_array_xy[0,:,:].shape )
#     dist = np.where (dist< dist_thr, True, False) 
#     # TODO: find the first True value along the axis=1
#     # note: each row is a different angle, and along columns the range increases


#     # constructing the template for the output 
#     theta_steps = np.int(theta_range *theta_res *360/(2*np.pi) )
#     t = np.linspace(0, theta_range, theta_steps, endpoint=False)
#     r = length_range * np.ones(rays_arr_xy.shape[1])
    
#     # updating the range vector  
#     # TODO: 

#     return r,t

################################################################################
def feature_set_by_oscar_description(print_out=True):
    '''
    The set of features are implemented according to:
    Oscar Martinez Mozos "Semantic labeling of places with mobile robots", 2010, Springer Berlin Heidelberg

    The followings are copied from the abovementioned document:

    A.1 Simple Features Extracted from Laser Beams
    A.1.1 Average Difference Between the Length of Two Consecutive Beams
    A.1.2 Standard Deviation of the Difference Between the Length of Two Consecutive Beams
    A.1.3 Average Difference Between the Length of Consecutive Beams Considering Max-Range
    A.1.4 Standard Deviation of the Difference Between the Length of Two Consecutive Beams Considering Max-Range
    A.1.5 The Average Beam Length
    A.1.6 The Standard Deviation of the Beam Length
    A.1.7 Number of Gaps
    A.1.8 Number of Beams Lying on Lines Extracted from the Range
    A.1.9 Euclidean Distance Between the Two Points Corresponding to Two Consecutive Global Minima
    A.1.10 The Angular Distance Between the Two Points Corresponding to Two Consecutive Global Minima
    A.1.11 Average of the Relation Between Two Consecutive Beams
    A.1.12 Standard Deviation of the Relation Between the Length of Two Consecutive Beams
    A.1.13 Average of Normalized Beam Length
    A.1.14 Standard Deviation of Normalized Beam Length
    A.1.15 Number of Relative Gaps
    A.1.16 Kurtosis

    A.2 Simple Features Extracted from a Polygon Approximation
    A.2.1 Area of P(z)
    A.2.2 Perimeter of P(z)
    A.2.3 Mean Distance Between the Centroid and the Shape Boundary
    A.2.4 Standard Deviation of the Distances Between the Centroid and the Shape Boundary
    A.2.5 Invariant Descriptors Based on the Fourier Transformation
    A.2.6 Major Axis Ma of the Ellipse that Approximates P(z)
    A.2.7 Minor Axis Mi of the Ellipse that Approximates P(z)
    A.2.8 Invariant Moments of P(z)
    A.2.9 Normalized Feature of Compactness of P(z)
    A.2.10 Normalized Feature of Eccentricity of P(z)
    A.2.11 Form Factor of P(z)
    A.2.12 Circularity of P(z)
    A.2.13 Normalized Circularity of P(z)
    A.2.14 Average Normalized Distance Between the Centroid and the Shape Boundary
    A.2.15 Standard Deviation of the Normalized Distances Between the Centroid and the Shape Boundary
    '''
    if print_out:
        print (feature_set_by_oscar.__doc__)
    else:
        return feature_set_by_oscar.__doc__
