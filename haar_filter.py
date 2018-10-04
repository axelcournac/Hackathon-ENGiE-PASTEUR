# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 13:29:29 2018

@author: GD5264
"""

from pylab import *
import numpy as np
import os
import pywt


def haar_filter(data, thres_percentile, level=1):
    """
    Filters image by applying following steps
    1.transorm data to Haar wavelet space
    2. truncate wavelet coefficients
    3. reconstruct image via truncated wavelet coefficients
    """
    
    if (level==1):
        (cA1, (cH1, cV1, cD1)) = pywt.wavedec2(data, wavelet='haar', level=level)
        ravel_cA1 = np.ravel(cA1)
        ravel_cH1 = np.ravel(cH1)
        ravel_cV1 = np.ravel(cV1)
        ravel_cD1 = np.ravel(cD1)
        coeffs = np.concatenate([ravel_cA1, ravel_cH1, ravel_cV1, ravel_cD1])
    elif (level==2):
        (cA1, (cH1, cV1, cD1), (cH2, cV2, cD2)) = pywt.wavedec2(data, wavelet='haar', level=level)
        ravel_cA1 = np.ravel(cA1)
        ravel_cH1 = np.ravel(cH1)
        ravel_cV1 = np.ravel(cV1)
        ravel_cD1 = np.ravel(cD1)
        ravel_cH2 = np.ravel(cH2)
        ravel_cV2 = np.ravel(cV2)
        ravel_cD2 = np.ravel(cD2)
        coeffs = np.concatenate([ravel_cA1, ravel_cH1, ravel_cV1, ravel_cD1, ravel_cH2, ravel_cV2, ravel_cD2])
    elif (level==3):
        (cA1, (cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3)) = pywt.wavedec2(data, wavelet='haar', level=level)
        ravel_cA1 = np.ravel(cA1)
        ravel_cH1 = np.ravel(cH1)
        ravel_cV1 = np.ravel(cV1)
        ravel_cD1 = np.ravel(cD1)
        ravel_cH2 = np.ravel(cH2)
        ravel_cV2 = np.ravel(cV2)
        ravel_cD2 = np.ravel(cD2)
        ravel_cH3 = np.ravel(cH3)
        ravel_cV3 = np.ravel(cV3)
        ravel_cD3 = np.ravel(cD3)
        coeffs = np.concatenate([ravel_cA1, ravel_cH1, ravel_cV1, ravel_cD1, ravel_cH2, ravel_cV2, ravel_cD2, ravel_cH3, ravel_cV3, ravel_cD3])
    elif (level==4):
        (cA1, (cH1, cV1, cD1), (cH2, cV2, cD2), (cH3, cV3, cD3), (cH4, cV4, cD4)) = pywt.wavedec2(data, wavelet='haar', level=level)
        ravel_cA1 = np.ravel(cA1)
        ravel_cH1 = np.ravel(cH1)
        ravel_cV1 = np.ravel(cV1)
        ravel_cD1 = np.ravel(cD1)
        ravel_cH2 = np.ravel(cH2)
        ravel_cV2 = np.ravel(cV2)
        ravel_cD2 = np.ravel(cD2)
        ravel_cH3 = np.ravel(cH3)
        ravel_cV3 = np.ravel(cV3)
        ravel_cD3 = np.ravel(cD3)
        ravel_cH4 = np.ravel(cH4)
        ravel_cV4 = np.ravel(cV4)
        ravel_cD4 = np.ravel(cD4)
        coeffs = np.concatenate([ravel_cA1, ravel_cH1, ravel_cV1, ravel_cD1, ravel_cH2, ravel_cV2, ravel_cD2, ravel_cH3, ravel_cV3, ravel_cD3, ravel_cH4, ravel_cV4, ravel_cD4])
    else:
        raise ValueError('level=', level, 'not implemented')
    
    coeffs_thres = np.percentile(coeffs, thres_percentile)
    
    if (level>0):
        truncated_cA1 = cA1.copy()
        truncated_cA1[abs(truncated_cA1)<coeffs_thres] = 0.0
        truncated_cH1 = cH1.copy()
        truncated_cH1[abs(truncated_cH1)<coeffs_thres] = 0.0
        truncated_cV1 = cV1.copy()
        truncated_cV1[abs(truncated_cV1)<coeffs_thres] = 0.0
        truncated_cD1 = cD1.copy()
        truncated_cD1[abs(truncated_cD1)<coeffs_thres] = 0.0
    if (level>1):
        truncated_cH2 = cH2.copy()
        truncated_cH2[abs(truncated_cH2)<coeffs_thres] = 0.0
        truncated_cV2 = cV2.copy()
        truncated_cV2[abs(truncated_cV2)<coeffs_thres] = 0.0
        truncated_cD2 = cD2.copy()
        truncated_cD2[abs(truncated_cD2)<coeffs_thres] = 0.0
    if (level>2):
        truncated_cH3 = cH3.copy()
        truncated_cH3[abs(truncated_cH3)<coeffs_thres] = 0.0
        truncated_cV3 = cV3.copy()
        truncated_cV3[abs(truncated_cV3)<coeffs_thres] = 0.0
        truncated_cD3 = cD3.copy()
        truncated_cD3[abs(truncated_cD3)<coeffs_thres] = 0.0
    if (level>3):
        truncated_cH4 = cH4.copy()
        truncated_cH4[abs(truncated_cH4)<coeffs_thres] = 0.0
        truncated_cV4 = cV4.copy()
        truncated_cV4[abs(truncated_cV4)<coeffs_thres] = 0.0
        truncated_cD4 = cD4.copy()
        truncated_cD4[abs(truncated_cD4)<coeffs_thres] = 0.0
    if (level==1):
        truncated_coeffs = (truncated_cA1, (truncated_cH1, truncated_cV1, truncated_cD1))
    elif (level==2):
        truncated_coeffs = (truncated_cA1, (truncated_cH1, truncated_cV1, truncated_cD1), (truncated_cH2, truncated_cV2, truncated_cD2))
    elif (level==3):
        truncated_coeffs = (truncated_cA1, (truncated_cH1, truncated_cV1, truncated_cD1), (truncated_cH2, truncated_cV2, truncated_cD2), (truncated_cH3, truncated_cV3, truncated_cD3))
    elif (level==4):
        truncated_coeffs = (truncated_cA1, (truncated_cH1, truncated_cV1, truncated_cD1), (truncated_cH2, truncated_cV2, truncated_cD2), (truncated_cH3, truncated_cV3, truncated_cD3), (truncated_cH4, truncated_cV4, truncated_cD4))
        
    smooth_data = pywt.waverec2(truncated_coeffs, wavelet='haar')
    
    return smooth_data

if (__name__=='__main__'):

    io_dir = 'C:\\Users\\gd5264\\Downloads\\TRAINING_SET'
    fname = 'MAT_RAW_realisation_927.txt'

    thres_percentile = 90.

    level = 4

    full_fname = os.path.join(io_dir, fname)
    data = np.loadtxt(full_fname)
    smooth_data = haar_filter(data, thres_percentile, level=level)
    
    figure()
    imshow(data)
    title('data')
    
    figure()
    imshow(smooth_data)
    title('smooth_data')