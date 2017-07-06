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
import scipy.signal

################################################################################
def smooth(x, window_size=11, window='hanning'):
    '''
    smooth the data using a window with requested size.
    http://scipy-cookbook.readthedocs.io/items/SignalSmooth.html
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_size: the len of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    '''

    if window_size < 3:  return x

    if x.ndim != 1: raise (StandardError('smooth only accepts 1 dimension arrays.'))
    if x.size < window_size:  raise (StandardError('Input vector needs to be bigger than window size.'))
    win_type = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    if window not in win_type: raise( StandardError( 'Window type is unknown'))

    s = np.r_[x[window_size-1:0:-1],x,x[-2:-window_size-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=np.ones(window_size,'d')
    else:
        w=eval('np.'+window+'(window_size)')

    y=np.convolve(w/w.sum(),s,mode='valid')

    # saesha modify
    ds = y.shape[0] - x.shape[0] # difference of shape
    dsb = ds//2 # [almsot] half of the difference of shape for indexing at the begining
    dse = ds - dsb # rest of the difference of shape for indexing at the end
    y = y[dsb:-dse]

    return y


################################################################################
def smooth_along_axis(x, window_size=11, window='hanning'):
    '''
    each rows of x is a separate signal to be smoothed
    '''

    if window_size < 3:  return x

    if x.ndim != 2: raise (StandardError('smooth_along_axis only accepts 2 dimension arrays.'))
    if x.shape[1] < window_size:  raise (StandardError('Input vector needs to be bigger than window size.'))
    win_type = ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']
    if window not in win_type: raise( StandardError( 'Window type is unknown'))


    s = np.concatenate( (x[:, window_size-1:0:-1], x, x[:,-2:-window_size-1:-1]), axis=1)

    if window == 'flat': #moving average
        w=np.ones(window_size,'d')
    else:
        w=eval('np.'+window+'(window_size)')

    y = np.apply_along_axis(lambda m: np.convolve(m, w, mode='valid'), axis=1, arr=s)

    ds = y.shape[1] - x.shape[1] # difference of shape
    dsb = ds//2 # [almsot] half of the difference of shape for indexing at the begining
    dse = ds - dsb # rest of the difference of shape for indexing at the end
    y = y[:,dsb:-dse]

    return y


################################################################################
def find_minima(x, window_size=10):
    '''
    '''

    # if condistion is '<', it will not detect plateau-like minima
    # if condistion is '<=', it will return all points in plateau in a plateau-like minima

    # let all the points to be minima in the plateau
    min_mask = np.stack( [np.r_[[True]*ws, x[ws:] <= x[:-ws]] & np.r_[x[:-ws] <= x[ws:], [True]*ws]
                           for ws in range(1, window_size+1)], axis=0 )
    peak_idx = np.nonzero( np.all(min_mask, axis=0) )[0]

    # detect plateau by finding close minima and average their location
    split_points = np.r_[0, np.nonzero( np.diff(peak_idx)> window_size )[0], peak_idx.shape[0]]

    
    # TODO: this first zero entry is very fishy!
    # as a matter of fact I know there are cases with two zeros at the begining
    # but that shouldn't be a problem though, since array[0:0] is empty!
    # maybe add the following to the loop?
    # if split_points[idx+1]- split_points[idx]>0])

    if split_points[0] == split_points[1]: split_points = split_points[1:]
    
    peak_idx = np.array([ int(peak_idx[split_points[idx]:split_points[idx+1]].mean())
                          for idx in range(split_points.shape[0]-1) ] )
                          
    return peak_idx


################################################################################
def find_minima_along_axis(x, window_size=10):
    '''
    incomplete

    test: create a series of segnals in a 2D array
    arr = [[signal1], [signal2],.. ]
    compare the results of this function, signal by signal, to the result of the
    find_minima method above
    There seems to be something fishy about the plateau detection
    There is a good change

    
    Test passed :D
    minim = utls.find_minima_along_axis(R, window_size=10)
    for idx in range(R.shape[0]):
        m1 = utls.find_minima(R[idx,:], window_size=10)
        m2 = np.array( minim[idx] )

        if m1.shape != m2.shape:
            print('diff_size:{:d}'.format(idx) )
    
        elif np.any(m1-m2):
            print('diff_value:{:d}'.format(idx) )
    '''

    # This method of finding local minima is based on comparing each point with its neighbors
    # to that end, the signal is shifted by values according to the size of the window, and 
    # the signal is compared to its shifted version
    # 'min_mask_p' and 'min_mask_n' represents the comparison for all shifting values (axis=1)
    # in different shifting direction.
    # this means, for instance, min_mask_p[i,:,:] represents the coparison of each point to its
    # (i+1)-step shifted version.
    # Then these two arrays, representing different shift directions are concatenated and 
    # that is followed by using np.all as a "logical and" operator (since np.logical_and won't work)
    # the final result (min_mask) is true for any point that is smaller than all neighbors
    min_mask_p = np.stack( [ np.concatenate( (np.ones((x.shape[0],ws)), x[:,ws:] <= x[:,:-ws]),axis=1)
                             for ws in range(1, window_size+1) ], axis=0)
    min_mask_n = np.stack( [ np.concatenate( (x[:,:-ws] <= x[:,ws:], np.ones((x.shape[0],ws))),axis=1)
                             for ws in range(1, window_size+1) ], axis=0 )
    min_mask = np.concatenate((min_mask_p, min_mask_n), axis=0)
    min_mask = np.all(min_mask, axis=0)
    
    ## since the length of minima is different for each row, longer we can maintain numpy.array
    # finding the indices to minima
    minima_indices = [np.nonzero(a)[0] for a in min_mask]
    
    # detect plateau by finding close minima and average their location
    split_points = [ np.r_[0, np.nonzero( np.diff(m_idx)> window_size )[0], len(m_idx)]
                     for m_idx in minima_indices ]

    for idx in range(len(split_points)):
        if split_points[idx][0] == split_points[idx][1]:
            split_points[idx] = split_points[idx][1:]
    
    minima_indices = [ np.array([ int(m_idx[spl_pts[idx]:spl_pts[idx+1]].mean())
                                   for idx in range(len(spl_pts)-1) ])
                       for m_idx,spl_pts in zip(minima_indices, split_points) ]
    

    return minima_indices


################################################################################
def find_maxima(x, window_size=10):
    '''
    '''
    return find_minima(-x, window_size)


################################################################################
################################################################################
################################################################################

# t = np.arange(0,10*np.pi, .1)
# a = np.sin(t) + np.random.random(t.shape)
# min_a_idx = find_minima(a, window_size=10)
# max_a_idx = find_maxima(a, window_size=10)
# s = smooth(a,window_size=21)
# min_s_idx = find_minima(s, window_size=10)
# max_s_idx = find_maxima(s, window_size=10)

# import matplotlib.pyplot as plt
# plt.plot(t,a, 'b')
# plt.plot(t[min_a_idx],a[min_a_idx], 'b*')
# plt.plot(t[max_a_idx],a[max_a_idx], 'b^')
# plt.plot(t,s, 'g')
# plt.plot(t[min_s_idx],s[min_s_idx], 'g*')
# plt.plot(t[max_s_idx],s[max_s_idx], 'g^')
# plt.show()
