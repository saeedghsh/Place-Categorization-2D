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
# import utilities
################################################################################
def construct_raycast_array(pose=[0,0],
                            length_range=30,
                            length_steps=30, 
                            theta_range=2*np.pi,
                            theta_res=1/1):
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
    x-value (and y-value) of points of all angles with a length[i] distance from
    center. increasing the index 'i' moves to out concentric circle.
    rays_array_xy[0,:,i].shape=(theta_range,)

    Note:
    -----
    The field of view is equal to the interval:
    [0 theta_range] (should it be [-theta_range/2 theta_range/2]?)

    Note:
    -----
    The output rays_array_xy contains coordinates to points in the map for
    raycasting. Before indexing the map with rays_array_xy, it must be cast from
    float to integer. It is not converted here, to avoid error of type cast for
    visualization.

    Note
    ----
    If the ray cast is not for simulating a robot's sensor, for instance if the
    target is feature extraction towards place categorization, it is more
    convinient (and slightly faster) to only contruct the rays_array_xy once,
    with pose=[0,0] and at every new location just change it by:
    >>> rays_array_xy[0,:,:] += pose[0]
    >>> rays_array_xy[1,:,:] += pose[1]

    Note
    ----
    Intended to keep the data type compact to save memory and use float16 for
    rays_array_xy, but np.meshgrid does not accept dtype and returns float64, so
    I'll keep this simple and just type cast in the main script.
    '''
    # number of steps in the angle vector (ie. number of rays per scan)
    theta_steps = np.int( theta_range *theta_res *360/(2*np.pi) )

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
    #     rays_array_xy = np.array( np.vsplit( np.concatenate((rays_l, rays_l),
    #                                                         axis=0), 2))
    #     rays_array_xy *= np.array( np.vsplit( np.concatenate((np.cos(rays_t),
    #                                                           np.sin(rays_t)),
    #                                                          axis=0), 2))

    # adjusting the points' coordinate wrt the pose 
    rays_array_xy[0,:,:] += pose[0]
    rays_array_xy[1,:,:] += pose[1]
    
    return rays_array_xy

################################################################################
def raycast_bitmap(pose, image,
                   occupancy_thr=127,
                   length_range=30,
                   length_steps=30,
                   mpp= 0.02,
                   range_res =1,
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

    mpp
    metric resolution of the occumpancy map ( meter per pixel )

    range_res
    pixel distance between points in the rays (pixel per point)

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

    ###### fixing the indices wrt image size
    # making sure all points in the rays_arr_xy are inside the image
    x_min = np.zeros(rays_arr_xy.shape[1:])
    y_min = np.zeros(rays_arr_xy.shape[1:])
    x_max = (image.shape[1]-1) *np.ones(rays_arr_xy.shape[1:])
    y_max = (image.shape[0]-1) *np.ones(rays_arr_xy.shape[1:])

    # although the following adjustment potentially messess up the alignment
    # of points of a ray, but that doesn't matter. that's because we pick the
    # index to the first occupied cell of the ray and use that, not image coordinate,
    # to calculate the range. Now if the border of image is occupied this is just fine.
    # otherewise, the last point is returned! since we'll `rays_image_val[:,-1] = 0`
    rays_arr_xy[0,:,:] = np.where(rays_arr_xy[0,:,:] > x_min, rays_arr_xy[0,:,:], x_min)
    rays_arr_xy[0,:,:] = np.where(rays_arr_xy[0,:,:] < x_max, rays_arr_xy[0,:,:], x_max)

    rays_arr_xy[1,:,:] = np.where(rays_arr_xy[1,:,:] > y_min, rays_arr_xy[1,:,:], y_min)
    rays_arr_xy[1,:,:] = np.where(rays_arr_xy[1,:,:] < y_max, rays_arr_xy[1,:,:], y_max)

    ###### finding the values in the image, corresponding to each point
    # assuming rays_arr_xy is already type casted to int
    # rays_image_val = image[rays_arr_xy[1,:,:], rays_arr_xy[0,:,:]]
    rays_image_val = image[rays_arr_xy[1,:,:].astype(int), rays_arr_xy[0,:,:].astype(int)]

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
    
    # scaling (converting) range values from ray point index to distance in meter

    # todo: this 
    print ('WARNING: beware, something is not right here...')
    print ('WARNING: r was multiplied with mpp, but I had to remove it for interactive raycasting for mesh to ogm')
    print ('WARNING: I don\'t know how sould it be for ray casting and place cat')
    r = r_idx * range_res #* mpp

    # to save memory, I typecast them here
    return r.astype(np.float16), t.astype(np.float16)
    # return r, t

################################################################################
def feature_subset(R,t, raycast_config, gap_t, point_closeness_thr=0.5):
    '''

    Inputs
    ------
    t: 1darray (1xn)
    n is the is the number of rays in each raycast, or the number of angles
    
    R: 2darray (mxn)
    m is the number of open_cells and hence the number of raycast vectors
    n is the number of rays in each raycast (i.e. the length of theta vector)


    Parameters
    ----------
    raycast_config: dictionary
    gap_t: [type:list, unit:meter]
    point_closeness_thr: [type:scalar, unit:meter]


    Output
    ------
    F: 2darray (mxl)
    m is the number of open_cells and hence the number of raycast vectors
    l is the length of fature vector (it depends on the parameters, eg. number of gap thresholds)

    F = [A13, A14, A11, A12, A21, A22, C, A116, STD, RAT, BIA, A17]
    F[:,0:2] = A13, A14: normalized (/raycast_config['range_meter']) mean and std of ranges
    F[:,2:4] = A11, A12: normalized (/raycast_config['range_meter']) mean and std of differences of ranges
    F[:,4] = A21: Area of P(z)
    F[:,5] = A22: Perimeter of P(z)
    F[:,6] = C: A22**2 / A21
    F[:,7] = A116: the kurtosis
    F[:,8:10] = PCA: principle components
    F[:,10] = RAT: ratio between principle components
    F[:,11] = BIA: the bias of the center

    no resampling:
    F[:,12:] = A17: number of gaps

    with resampling
    F[:,12:14] = PCA: principle components
    F[:,14] = RAT: ratio between principle components
    F[:,15] = BIA: the bias of the center

    F[:,16:] = A17: number of gaps
    '''

    assert R.ndim == 2

    ###### normalized mean and std of ranges - A13, A14
    A13 = np.atleast_2d( R.mean(axis=1) ).T / raycast_config['range_meter']
    # The Standard Deviation of the Beam Length
    A14 = np.atleast_2d( R.std(axis=1) ).T / raycast_config['range_meter']

    ###### normalized mean and std of differences of ranges  - A11, A12 (Normalized)
    diff = np.abs( np.diff(R, axis=1) ) / raycast_config['range_meter']
    # Average Difference Between the Length of Two Consecutive Beams
    A11 = np.atleast_2d( diff.mean(axis=1) ).T
    # Standard Deviation of the Difference Between the Length of Two Consecutive Beams
    A12 = np.atleast_2d( diff.std(axis=1) ).T

    ########################################
    RR = np.stack( (R,R), axis=2)
    T = np.stack( [t for _ in range(R.shape[0])], axis=0 )
    CS = np.stack( [np.cos(T), np.sin(T)], axis=2)
    XY = RR * CS
    X = XY[:,:,0]
    Y = XY[:,:,1]

    del RR, T, CS
    ########################################
    ### Area of P(z)
    A21 = np.atleast_2d( np.sum( X*np.roll(Y,-1,axis=1) - np.roll(X,-1,axis=1)*Y, axis=1) /2. ).T     
    del X,Y

    ########################################
    # PD: the distance between consecutive points in each point set (minus the first to last point distance)
    PD = np.sqrt(np.diff(XY[:,:,0],axis=1)**2 + np.diff(XY[:,:,1],axis=1)**2) 
    ### Perimeter of P(z)
    A22 = np.atleast_2d( np.sum(PD, axis=1) ).T
    del PD
    ### 
    C = A22**2 / A21

    ###### counting gaps - A17
    # Number of Gaps
    # TODO: shouldn't the number of gaps be integer? in the result they look weird
    # it was because diff is normalized to raycast_config['range_meter'], but gap_t
    # was still in meter, it should be fixed now
    A17 = np.concatenate( [ np.atleast_2d( np.count_nonzero(diff>gt/raycast_config['range_meter'], axis=1) ).T
                            for gt in gap_t ],
                          axis=1 )
    del diff

    ###### the kurtosis - A116
    A15 = np.atleast_2d( R.mean(axis=1) ).T # The Average Beam Length
    A16 = np.atleast_2d( R.std(axis=1) ).T # The Standard Deviation of the Beam Length
    A116 = (np.sum((R-A15)**4,axis=1) / (R.shape[1]*A16[:,0]**4)) - 3
    A116 = np.atleast_2d( np.where(np.isnan(A116), np.zeros(A116.shape[0]), A116  ) ).T
    del A15, A16

    ###### Of the ORIGINAL point set: PCA, the bias of center of gravity, etc. 
    # Compute the center of gravity = center of the gaussian
    Center_of_Gravity = XY.mean(axis=1)
    # Translate the distribution to center its CoG at origin of the
    XY_centered = XY - np.stack([Center_of_Gravity for _ in range(XY.shape[1])], axis=1)
    # Compute the singular value decomposition (Note: np.linalg does not accept float16)
    S = np.linalg.svd(XY_centered.astype(np.float32), full_matrices=False,compute_uv=False)
    del XY_centered

    ### principle components
    PCA = S/ np.sqrt(XY.shape[1]) 
    del S #, XY

    ### ratio between principle components
    RAT = np.atleast_2d(PCA[:,0]/PCA[:,1]).T
    ### the bias of the center
    BIA = np.atleast_2d( np.sqrt(Center_of_Gravity[:,0]**2+Center_of_Gravity[:,1]**2)).T 
    del Center_of_Gravity


    ########################################
    ###### Of the RESAMPLED point set: PCA, the bias of center of gravity, etc.
    # Df: forward distance is the distance between the current point [i] and the next point [i+1]
    dx = XY[:,:,0] - np.roll(XY[:,:,0], -1, axis=1)
    dy = XY[:,:,1] - np.roll(XY[:,:,1], -1, axis=1)
    Df = np.sqrt(dx**2 + dy**2)
    del dx, dy
    # Db: backward distance is the distance between the current point [i] and the previous point [i-1]
    dx = XY[:,:,0] - np.roll(XY[:,:,0], 1, axis=1)
    dy = XY[:,:,1] - np.roll(XY[:,:,1], 1, axis=1)
    Db = np.sqrt(dx**2 + dy**2)
    del dx, dy
    # those point that have both backward and forward distance less than the threshold are invalid (masked True)
    # that is to say, a point is valid (masked False) only if atleast one of the distances is more than threshold
    # Note that what is masked True is ignored in operation
    mask = np.logical_and( Db<point_closeness_thr, Df<point_closeness_thr )
    del Db, Df

    XY_ma = np.ma.array(XY, mask=np.stack( (mask,mask),axis=2) )
    del XY, mask

    # Compute the center of gravity = center of the gaussian
    Center_of_Gravity_ma = XY_ma.mean(axis=1)
    # Translate the distribution to center its CoG at origin of the
    XY_centered_ma = XY_ma - np.stack([Center_of_Gravity_ma for _ in range(XY_ma.shape[1])], axis=1)
    del XY_ma
    
    # Compute the singular value decomposition (Note: np.linalg does not accept float16)
    # The problem is here! I can't get it right :(
    XY_centered_ma.fill_value = 0
    S_ma = np.linalg.svd(XY_centered_ma.filled().astype(np.float32), full_matrices=False,compute_uv=False)
    # np.array([ np.linalg.svd( np.ma.compress_rowcols(XY_ma[idx,:,:], axis=0).T,
    #                           full_matrices=False, compute_uv=False )
    #            for idx in range(XY_ma.shape[0]) ]) 

    ### principle components
    PCA_ma = S_ma/ np.sqrt( XY_centered_ma.count(axis=1) ) # np.sqrt(XY_ma.shape[1]) 
    del S_ma, XY_centered_ma

    ### ratio between principle components
    RAT_ma = np.atleast_2d(PCA_ma[:,0]/PCA_ma[:,1]).T
    ### the bias of the center
    BIA_ma = np.atleast_2d( np.sqrt(Center_of_Gravity_ma[:,0]**2+Center_of_Gravity_ma[:,1]**2)).T
    del Center_of_Gravity_ma
    ########################################
    F = np.concatenate( (A13, A14, A11, A12, A21, A22, C, A116,
                         PCA, RAT, BIA,
                         PCA_ma, RAT_ma, BIA_ma,
                         A17),
                        axis=1)
    return F


################################################################################
def feature_set_A1(R, t, raycast_config, gap_t, rel_gap_t):
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


    F = [A11,A12, A15,A16, A111,A112, A113,A114, A115, A116, A17]

    F[:,0:2] = A11,A12: mean and std of Difference Between the Length of Two Consecutive Beams
    F[:,2:4] = A15,A16: mean and std of Beam Length
    F[:,4:6] = A111,A112: mean and std of the Relation Between Two Consecutive Beams
    F[:,6:8] = A113,A114: mean and std of Normalized Beam Length (/max(rays))
    F[:,8] = A115: number of relative gaps
    F[:,9] = A116: Kurtosis
    F[:,10:] = A17: Number of Gaps

    '''
    # check this link:
    # https://docs.scipy.org/doc/scipy-0.19.0/reference/stats.html
    # like: scipy.stats.describe

    assert R.ndim == 2
    
    diff = np.abs( np.diff(R, axis=1) )
    # Average Difference Between the Length of Two Consecutive Beams
    A11 = np.atleast_2d( diff.mean(axis=1) ).T
    # Standard Deviation of the Difference Between the Length of Two Consecutive Beams
    A12 = np.atleast_2d( diff.std(axis=1) ).T
    
    # Number of Gaps
    A17 = np.concatenate( [ np.atleast_2d( np.count_nonzero(diff>gt, axis=1) ).T
                            for gt in gap_t ],
                          axis=1 )
    del diff

    # The Average Beam Length
    A15 = np.atleast_2d( R.mean(axis=1) ).T
    # The Standard Deviation of the Beam Length
    A16 = np.atleast_2d( R.std(axis=1) ).T

    # TODO: double check if this is exactly what is suggested in Oscar's thesis
    # Average Difference Between the Length of Consecutive Beams Considering Max-Range
    A13 = A15 / raycast_config['range_meter']
    # Standard Deviation of the Difference Between the Length of Two Consecutive Beams Considering Max-Range
    A14 = A16 / raycast_config['range_meter']

    # # For two reason I won't include this:
    # # 1) too much work
    # # 2) I don't the point of it anyway!
    # A18 = Number of Beams Lying on Lines Extracted from the Range (method from \cite{sack2004comparison})

    # # # for door frame detection
    # # A19: Euclidean Distance Between the Two Points Corresponding to Two Consecutive Global Minima
    # # A110: The Angular Distance Between the Two Points Corresponding to Two Consecutive Global Minima
    # # These features are specifically designed to find doorways.
    # # I find the the definition of "Global Minima" very subjective, that requires an ad-hoc minima detection
    # # instead I take the mean and std of Angular Distances (and skip Euclidean conversion :D)
    # # UPDATE:  I decide to ignore this too, it's too slow and still is not working well
    # window_size = R.shape[1]//36
    # S = utilities.smooth_along_axis(R, window_size, window='hanning')
    # minima = utilities.find_minima_along_axis(S, window_size)
    # T_diff = [ np.diff( t[m_idx] ) for m_idx in minima ]
    # A19a = np.atleast_2d( [T_diff[idx].mean() for idx in range(R.shape[0])] ).T
    # A19b = np.atleast_2d( [T_diff[idx].std() for idx in range(R.shape[0])] ).T 

    rel = R / (np.roll(R, 1, axis=1)+np.spacing(1)) # np.spacing helps avoiding division by zero
    # In the following line the 1./rel might encounter division by zero, but it should matter
    # Because if rel was original zero (hence division by zero), then its zero value is copied to final rel
    rel = np.where (rel<=1, rel, 1./rel) # making sure all relative length are (min/max) ie. 0<rel<=1
    # Average of the Relation Between Two Consecutive Beams
    A111 = np.atleast_2d( rel.mean(axis=1) ).T
    # Standard Deviation of the Relation Between the Length of Two Consecutive Beams
    A112 = np.atleast_2d( rel.std(axis=1) ).T

    # Number of Relative Gaps
    A115 = np.atleast_2d( np.count_nonzero( rel > rel_gap_t, axis=1) ).T
    del rel

    nrm = R / np.atleast_2d( R.max(axis=1) ).T
    # Average of Normalized Beam Length
    A113 = np.atleast_2d( nrm.mean(axis=1) ).T
    # Standard Deviation of Normalized Beam Length
    A114 = np.atleast_2d( nrm.std(axis=1) ).T

    del nrm

    # Kurtosis
    # A16 should be indexed to its only column, otherwise it will be 2d and
    # the result of division will become a matrix instead of a vector
    A116 = np.atleast_2d( (np.sum((R-A15)**4,axis=1) / (R.shape[1]*A16[:,0]**4)) - 3 ).T 
    
    F = np.concatenate( (A11,A12, A15,A16, A111,A112, A113,A114, A115, A116, A17),
                        axis=1)

    # F = np.concatenate( (A11,A12, A15,A16, A17, A19a,A19b, A111,A112, A113,A114, A115, A116),
    #                     axis=1)

    return F

################################################################################
def feature_set_A2(R,t, EFD_degree=10):
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

    F = [A21, A22, A23,A24, A214,A215, A26,A27, A28, A29,A210, A211, A212, A213, A25]

    F[:,0] = A21: Area of P(z)
    F[:,1] = A22: Perimeter of P(z)
    F[:,2:4] = A23,A24: mean and std of Distance Between the Centroid and the Shape Boundary
    F[:,4:6] = A214,A215: mean and std of Normalized Distance Between the Centroid and the Shape Boundary
    F[:,6:8] = A26,A27: Major and Minor Axis Ma/Mi of the Ellipse that Approximates P(z)
    F[:,8:15] = A28: 7 Invariant Moments of P(z)
    F[:,15:17] = A29,A210: Normalized Feature of Compactness and Eccentricity of P(z)
    F[:,17] = A211: Form Factor of P(z)
    F[:,18] = A212: Circularity of P(z)
    F[:,19] = A213: Normalized Circularity of P(z)
    F[:,20:] = A25: EFD - Invariant Descriptors Based on the Fourier Transformation
    '''
    assert R.ndim == 2

    RR = np.stack( (R,R), axis=2)
    T = np.stack( [t for _ in range(R.shape[0])], axis=0 )
    CS = np.stack( [np.cos(T), np.sin(T)], axis=2)
    XY = RR * CS
    del RR, T, CS # leave R, XY

    ########################################
    X = XY[:,:,0]
    Y = XY[:,:,1]
    ### Area of P(z)
    A21 = np.atleast_2d( np.sum( X*np.roll(Y,-1,axis=1) - np.roll(X,-1,axis=1)*Y, axis=1) /2. ).T     
    # leave R, A21, XY, X, Y

    ########################################
    # PD: the distance between consecutive points in each point set (minus the first to last point distance)
    PD = np.sqrt(np.diff(XY[:,:,0],axis=1)**2 + np.diff(XY[:,:,1],axis=1)**2) 
    ### Perimeter of P(z)
    A22 = np.atleast_2d( np.sum(PD, axis=1) ).T
    del PD # leave R, A21, A22, XY, X, Y

    ########################################
    ### centeroid of P(z)
    # Cx, Cy: 1d arrays containg the x and y coordinates of centroids
    tmp = (X*np.roll(Y,-1,axis=1)-np.roll(X,-1,axis=1)*Y)
    Cx = np.sum( (X+np.roll(X,-1,axis=1)) * tmp, axis=1) / (A21 *6)[:,0]
    Cy = np.sum( (Y+np.roll(Y,-1,axis=1)) * tmp, axis=1) / (A21 *6)[:,0]
    # C: 2d array containg the coordinates (x,y) of centroids
    C = np.stack( (Cx,Cy), axis=1) 
    # CX, CY: 2d arrays, that are stacked version of Cx and Cy
    CX = np.stack( [C[:,0] for _ in range(R.shape[1])], axis=1)
    CY = np.stack( [C[:,1] for _ in range(R.shape[1])], axis=1)
    # CD: 2d array, distance of every points in each point set to the centroid of that set
    CD = np.sqrt( (X - CX)**2 + (Y - CY)**2 ) 
    del tmp, Cx, Cy, CX, CY # leave R, A21, A22, XY, X, Y, C, CD

    # Mean Distance Between the Centroid and the Shape Boundary
    A23 = np.atleast_2d( CD.mean(axis=1) ).T 
    # Standard Deviation of the Distances Between the Centroid and the Shape Boundary
    A24 = np.atleast_2d( CD.std(axis=1) ).T 

    ########################################
    # Average Normalized Distance Between the Centroid and the Shape Boundary
    A214 = np.atleast_2d( A23[:,0] / CD.max(axis=1) ).T 
    # Standard Deviation of the Normalized Distances Between the Centroid and the Shape Boundary
    A215 = np.atleast_2d( A24[:,0] / CD.max(axis=1) ).T
    del CD # leave R, A21, A22, A23, A24, XY, X, Y, C

    ########################################
    ### Fourier Transform
    CMPX = X + 1j*Y
    FFT = np.fft.fft(CMPX, axis=1, norm=[None, 'ortho'][1])
    # FFT = {c[0], c[1], ..., c[n-1], c[-n], ..., c[-1]} -> we should shift befor truncating
    if (EFD_degree > FFT.shape[1]): raise( StandardError('EFD degree can\'t be larger than {:d}'.format(FFT.shape[1]) ) )
    idx_s, idx_e = (FFT.shape[1] - EFD_degree)//2 , (FFT.shape[1] + EFD_degree)//2
    FFT = np.fft.ifftshift( np.fft.fftshift(FFT)[:, idx_s:idx_e] ) # truncate
    ### Elliptic Fourier Descriptor
    idx = np.stack( [range(FFT.shape[1]//2)+range(-FFT.shape[1]//2,0) for _ in range(FFT.shape[0])], axis=0 )
    angl = np.angle( FFT )
    phi_1 = np.stack( [angl[:,1] for _ in range(angl.shape[1])], axis=1 )
    phi_2 = np.stack( [angl[:,2] for _ in range(angl.shape[1])], axis=1 )
    absl = np.abs( FFT )
    abs_1 = np.stack( [absl[:,1] for _ in range(angl.shape[1])], axis=1 )
    EFD = (absl/abs_1) * np.exp((angl +(1-idx)*phi_2 -(2-idx)*phi_1)*1j)

    # Invariant Descriptors Based on the Fourier Transformation
    # TODO:
    # EFD values are complex and I take their absolute value to match other features' type
    # not sure if this is correct!
    A25 = np.abs( EFD )
    del CMPX, idx_s, idx_e, idx, angl, phi_1, phi_2, abs_1, FFT, EFD 

    # Major Axis Ma of the Ellipse that Approximates P(z)
    A26 = np.atleast_2d( absl[:,1]+absl[:,-1] ).T 
    # Minor Axis Mi of the Ellipse that Approximates P(z)
    A27 = np.atleast_2d( np.abs(absl[:,1]-absl[:,-1]) ).T  
    del absl # leave R, A21,A22,A23,A24,A25,A26,A27, XY, X, Y, C

    ########################################
    ### Normalized Central Moments
    Xh = np.mean(X, axis=1)
    Yh = np.mean(Y, axis=1)
    dX = X - np.stack([Xh for _ in range(X.shape[1])], axis=1)
    dY = Y - np.stack([Yh for _ in range(Y.shape[1])], axis=1)
    del Xh, Yh
    
    # since this is a point set, there is one summation over points
    # rather two summation over rows and columns (x,y) as for images
    # also points have equal value, so there is not f(x,y) to multiply
    # (as in image we do!)
    MU = lambda p,q: np.sum(dX**p * dY**q, axis=1) / X.shape[1]
    MU00 = MU(0,0)
    ETA = lambda p,q: MU(p,q) / MU00**((p+q)/2 +1)

    ETA_11 = ETA(1,1)
    ETA_02 = ETA(0,2)
    ETA_20 = ETA(2,0)
    ETA_12 = ETA(1,2)
    ETA_21 = ETA(2,1)
    ETA_30 = ETA(3,0)
    ETA_03 = ETA(0,3)
    
    PHI_1 = ETA_20 + ETA_02
    PHI_2 = (ETA_20 - ETA_02)**2 + 4* ETA_11**2
    PHI_3 = (ETA_30 - 3*ETA_12)**2 + (3*ETA_21 - ETA_03)**2
    PHI_4 = (ETA_30 + ETA_12)**2 + (ETA_21 + ETA_03)**2

    PHI_5 = (ETA_30 - 3*ETA_12) * (ETA_30 + ETA_12) * ( (ETA_30 + ETA_12)**2 - 3*(ETA_21 + ETA_03)**2 )
    PHI_5 += (3*ETA_21 - ETA_03) * (ETA_21 + ETA_03) * ( 3*(ETA_30 - ETA_12)**2 - (ETA_21 + ETA_03)**2 )

    PHI_6 = (ETA_20 - ETA_02) * ( (ETA_30 + ETA_12)**2 - (ETA_21 + ETA_03)**2 )
    PHI_6 += 4 * ETA_11 * (ETA_30 - ETA_12) * (ETA_21 - ETA_03)

    PHI_7 = (3*ETA_21 - ETA_03) * (ETA_30 - ETA_12) * ( (ETA_30 - ETA_12)**2 - 3*(ETA_21 + ETA_03)**2 )
    PHI_7 += (3*ETA_12 - ETA_30) * (ETA_21 + ETA_03) * ( 3*(ETA_30 + ETA_12)**2 - (ETA_21 + ETA_03)**2 )

    # Invariant Moments of P(z)
    A28 = np.stack( (PHI_1, PHI_2, PHI_3, PHI_4, PHI_5, PHI_6, PHI_7), axis=1 )
    del PHI_1, PHI_2, PHI_3, PHI_4, PHI_5, PHI_6, PHI_7
    del ETA_11, ETA_02, ETA_20, ETA_12, ETA_21, ETA_30, ETA_03

    ########################################
    # A21 is the area
    MU_11, MU_20, MU_02 = MU(1,1), MU(2,0), MU(0,2)
    # Normalized Feature of Compactness of P(z)
    A29 = np.atleast_2d( A21[:,0] / (MU_20+MU_02) ).T
    if not ( np.all( 0<=A29 ) and np.all( A29<=1 ) ): print('A29 - oops')
    # Normalized Feature of Eccentricity of P(z)
    A210 = np.atleast_2d( np.sqrt( (MU_20+MU_02)**2 + 4*MU_11**2 ) / (MU_20+MU_02) ).T
    if not ( np.all( 0<=A210 ) and np.all( A210<=1 ) ): print('A210 - oops')
    del MU_11, MU_20, MU_02

    ########################################
    # A21 and A22 are the area and perimeter
    # Form Factor of P(z)
    A211 = np.atleast_2d( 4 * np.pi * A21[:,0] / np.sqrt(A22[:,0]) ).T
    # Circularity of P(z)
    A212 = np.atleast_2d( A22[:,0]**2 / A21[:,0] ).T 
    # Normalized Circularity of P(z)
    A213 = np.atleast_2d( 4 * np.pi * A21[:,0] / A22[:,0]**2 ).T

    ########################################

    F = np.concatenate( (A21, A22, A23,A24, A214,A215, A26,A27,
                         A28, A29,A210, A211, A212, A213, A25),
                        axis=1)
    return F


################################################################################
####################################################################### obsolete
################################################################################

################################################################################
def raycast_bitmap_batch(open_cells, image,
                         occupancy_thr=127,
                         length_range=30,
                         length_steps=30,
                         mpp= 0.02,
                         range_res =1,
                         theta_range=2*np.pi,
                         theta_res=1/1,
                         rays_array_xy=None):
    '''
    This method takes a bitmap iamge and returns a raycast from the specided
    pose.

    Not very practical, it requires huge memory. Instead use raycast_bitmap()

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
    occupancy_thr: (default: 127)
    a pixel with value less-equal to this threshold is considered as occupied

    mpp
    metric resolution of the occumpancy map ( meter per pixel )

    range_res
    pixel distance between points in the rays (pixel per point)

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
    if not all_inside: raise Exception('open cells out of image!!!')
 
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
    r_idx = r_idx.astype(np.int16)
    r_idx = np.concatenate((np.atleast_1d(r_idx[0]), r_idx[np.nonzero(np.diff(t_idx)!=0)[0]+1]))
    r_idx = r_idx.reshape(open_cells.shape[0], theta_steps)

    # scaling (converting) range values from ray point index to distance in meter
    R = r_idx * range_res * mpp

    # return R,t
    return R.astype(np.float16), t.astype(np.float16)



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
def raycast_to_features(t,r,
                        mpp, RLimit,
                        gapThreshold= [1.0]):
    '''
    Insipred by [1] this method delivers a feature vector that is a descriptor
    for the shape of the surrounding, represented by the raycast in input

    Note:
    This is old implementation (not vectorized)
    for new version see feature_subset()

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

    Elements of the feature vector
    ------------------------------
    [x] 1. Euclidean distance between the two points corresponding to the two smallest local minima.
    [x] 2. The angular distance between the beams corresponding to the local minima in previous.
    [x] 3. 200 similarity invariant descriptors based in the Fourier transformation.
    [v] 4. Major axis Ma of the ellipse that approximates P(z) using the first two Fourier coefficients.
    [v] 5. Minor axis Mi of the ellipse that approximate P(z) using the first two Fourier coefficients.
    [v] 6. Ma/Mi.
    [x] 7. Seven invariants calculated from the central moments of P(z).
    8. Normalized feature of compactness of P(z).
    9. Normalized feature of eccentricity of P(z).
    10. Form factor of P(z)

    references
    ----------
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
    

    ###### normalized mean and std of ranges - A15, A16
    r_mean = np.mean(r)
    r_stdv = np.sqrt( np.sum((r-r_mean)**2) /len(r) )
    r_mean_norm = r_mean / RLimit
    r_stdv_norm = r_stdv / RLimit
    f.append(r_mean_norm), f.append(r_stdv_norm)

    ###### normalized mean and std of differences of ranges  - A11, A12
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

    ###### counting gaps - A17
    for gt in gapThreshold:
        gap_count = len(np.nonzero(np.abs(np.diff(r))>gt)[0])
        f.append(gap_count)
    
    ###### the kurtosis - A116
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
###################################################### In progess - not finished
################################################################################

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
