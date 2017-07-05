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
    """
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
    """

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
def find_peaks_min(x, window_size=10):
    '''
    '''

    # if condistion is '<', it will not detect plateau-like minima
    # if condistion is '<=', it will return all points in plateau in a plateau-like minima

    # let all the points to be minima in the plateau
    min_stack = np.stack( [np.r_[[True]*ws, x[ws:] <= x[:-ws]] & np.r_[x[:-ws] <= x[ws:], [True]*ws]
                           for ws in range(1, window_size+1)], axis=0 )
    peak_idx = np.nonzero( np.all(min_stack,axis=0) )[0]

    # detect plateau by finding close minima and average their location
    split_points = np.r_[0, np.nonzero( np.diff(peak_idx)> window_size )[0], peak_idx.shape[0]]
    peak_idx = np.array([ int(peak_idx[spl:spl+1].mean()) for spl in split_points[:-1]])
    
    return peak_idx


################################################################################
def find_peaks_max(x, window_size=10):
    '''
    '''
    return find_peaks_min(-x, window_size)


################################################################################
################################################################################
################################################################################

# t = np.arange(0,10*np.pi, .1)
# a = np.sin(t) + np.random.random(t.shape)
# min_a_idx = find_peaks_min(a, window_size=10)
# max_a_idx = find_peaks_max(a, window_size=10)
# s = smooth(a,window_size=21)
# min_s_idx = find_peaks_min(s, window_size=10)
# max_s_idx = find_peaks_max(s, window_size=10)

# import matplotlib.pyplot as plt
# plt.plot(t,a, 'b')
# plt.plot(t[min_a_idx],a[min_a_idx], 'b*')
# plt.plot(t[max_a_idx],a[max_a_idx], 'b^')
# plt.plot(t,s, 'g')
# plt.plot(t[min_s_idx],s[min_s_idx], 'g*')
# plt.plot(t[max_s_idx],s[max_s_idx], 'g^')
# plt.show()
