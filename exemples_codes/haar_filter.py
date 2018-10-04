# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:29:29 2018

@author: GD5264
"""

from pylab import *
import numpy as np
import os
import pywt


def haar_filter(data, thres_percentile):
    """
    Filters image by applying following steps
    1.transorm data to Haar wavelet space
    2. truncate wavelet coefficients
    3. reconstruct image via truncated wavelet coefficients
    """
    
    (cA2, (cH2, cV2, cD2)) = pywt.dwt2(data, wavelet='haar')
    
    cA1 = np.ravel(cA2)
    cH1 = np.ravel(cH2)
    cV1 = np.ravel(cV2)
    cD1 = np.ravel(cD2)
    coeffs1 = np.concatenate([cA1, cH1, cV1, cD1])
    
    coeffs_thres = np.percentile(coeffs1, thres_percentile)
    truncated_cA2 = cA2.copy()
    truncated_cA2[abs(truncated_cA2)<coeffs_thres] = 0.0
    
    truncated_cH2 = cH2.copy()
    truncated_cH2[abs(truncated_cH2)<coeffs_thres] = 0.0
    truncated_cV2 = cV2.copy()
    truncated_cV2[abs(truncated_cV2)<coeffs_thres] = 0.0
    truncated_cD2 = cD2.copy()
    truncated_cD2[abs(truncated_cD2)<coeffs_thres] = 0.0
    
    truncated_coeffs = (truncated_cA2, (truncated_cH2, truncated_cV2, truncated_cD2))
    smooth_data = pywt.idwt2(truncated_coeffs, wavelet='haar')
    
    return smooth_data

if (__name__=='__main__'):

    io_dir = 'C:\\Users\\gd5264\\Downloads\\TRAINING_SET'
    fname = 'MAT_RAW_realisation_927.txt'

    thres_percentile = 90.

    full_fname = os.path.join(io_dir, fname)
    data = np.loadtxt(full_fname)
    smooth_data = haar_filter(data, thres_percentile)
    
    figure()
    imshow(data)
    title('data')
    
    figure()
    imshow(smooth_data)
    title('smooth_data')