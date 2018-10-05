# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 21:45:06 2018
@author: axel c
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import imageio
import os
from skimage import measure 
from scipy.ndimage import gaussian_filter, generate_binary_structure, maximum_filter, label

# .............................. #
# ... Normalisation function ... #
# .............................. #

def scn_func(A,threshold=0):    
    n1 = A.shape[0];
    n_iterations=10;
    keep = np.zeros((n1, 1));
    
    for i in range(0,n1) :
        if np.sum(A[i,]) > threshold:
            keep[i] = 1
        else :
            keep[i] = 0
    
    indices1=np.where(keep >0 )
    indices2=np.where(keep <=0 )
    
    for n in range(0,n_iterations) :
        for i in range(0,n1) :
            A[indices1[0],i]=A[indices1[0],i]/ np.sum(A[indices1[0],i])
            A[indices2[0],i]=0   
        A[np.isnan(A)] = 0.0 
        
        for i in range(0,n1) :    
            A[i,indices1[0]]=A[i,indices1[0]]/ np.sum(A[i,indices1[0]])
            A[i,indices2[0]]=0  
        A[np.isnan(A)] = 0.0    
        
    return A

# ................................ #
# ... Detrend with Genomic law ... #
# ................................ #

def distance_law_human(A):
    n1 = A.shape[0]
    dist = np.zeros((n1, 1))
    for nw in range(n1):  # scales
        somme = []
        for i in range(n1):
                kp = i - nw
                if (kp >= 0) and (kp < n1):
                    somme.append(A[i, kp])
        dist[nw] = np.mean(somme)
    return dist

# ........................ #
# ... Find local maxima... #
# ........................ #
    
def local_conv_max(convolution_rescaled):
    

    
    thr = 0.12 # Noise threshold
    peakthr = 0.22 # Peak threshold
    
    # De-noising
    convo2 = np.array(convolution_rescaled)
    convo2[convo2<thr] = thr
    
    # Apply Gaussian Blur
    convo_blur = gaussian_filter(convo2, 2)
    convo_blur2 = convo_blur - min(np.ndarray.flatten(convo_blur))
    blurnorm = convo_blur2 / max(np.ndarray.flatten(convo_blur2))
    
    # Find local maxima, pixel value indicate the magnitude of the maximum
    neighborhood = generate_binary_structure(2,2)
    localmax = (maximum_filter(blurnorm, footprint=neighborhood) == blurnorm) * blurnorm
    
    # Display peaks overlaid on blurred matrix
    disp = np.array(blurnorm)
    disp[localmax>=peakthr] = 3
    
    # Apply threshold to the peaks and label the regions
    peaks = localmax >= peakthr
    labels, numregions = label(peaks) # labels is the labelled image
    
    props = measure.regionprops(labels)
    centroids = [i.centroid for i in props]
    centroids[(np.array(centroids)[:,1]-np.array(centroids)[:,0])>12,:] # to be reshaped


    return np.array(centroids)
    

# .................... #
# ... Find suspect ... #
# .................... #

def pattern_finder(realisation, pattern, with_plots=False):

    matscn = scn_func(realisation)
    dist = distance_law_human(matscn)
    n1 = np.shape(matscn)[0]

    # Computation of genomic distance law matrice:
    MAT_DIST = np.zeros((n1, n1))
    for i in range(n1):
        for j in range(n1):
            MAT_DIST[i, j] =  dist[abs(j-i)]

    MAT_DETREND = matscn/MAT_DIST
    MAT_DETREND[np.isnan(MAT_DETREND)] = 1.0

    if with_plots:
        plt.figure()
        plt.imshow(matscn**0.15, interpolation = "none", cmap ="afmhot_r" )
        for l in loops:
            plt.scatter(l[1], l[0], s=80, facecolors='none', edgecolors='yellow')

    # pattern = np.loadtxt(os.path.join(path_input, "pattern_loop_median_detrended.txt"))
    area = np.shape(pattern)[0]

    if with_plots:
        plt.figure()
        plt.imshow(pattern**0.2, interpolation="none", cmap="afmhot_r")

    # Convert to .png
    imageio.imsave('matrice.png', MAT_DETREND)
    imageio.imsave('template.png', pattern)
    
    # Read .png for openCV
    img = cv2.imread('matrice.png')
    tp = cv2.imread('template.png')
    # print(np.shape(img))
    # print(np.shape(tp))
    res = cv2.matchTemplate(img, tp, cv2.TM_CCOEFF_NORMED)
    n2 = np.shape(res)[0]
    
    # Rescale output convolution to initial image
    res_rescaled = np.zeros(np.shape(matscn))
    res_rescaled[np.ix_(range(int(area/2), n2+int(area/2)), 
                        range(int(area/2),n2+int(area/2)))] = res
    
    # vect_values = np.reshape(res_rescaled , (1,n1*n1) )
    # indices_max = np.where(res_rescaled > (0.5*res_rescaled.max()))
    # indices_max = np.where(res_rescaled > np.median(vect_values) + 4 * np.std(vect_values))
    # indices_max = np.array(indices_max)
    
    # vect_values = np.reshape(res_rescaled , (1,n1*n1) )
    indices_max = np.where(res_rescaled > (0.5*res_rescaled.max()))
    # indices_max = np.where(res_rescaled > np.median(vect_values) + 4 * np.std(vect_values))
    indices_max = np.array(indices_max)
    
    # mat_res is a binary image such that
    # 1 <-> loop
    # 0 <-> no-loop
    mat_res = np.zeros(np.shape(res_rescaled))
    for ind in range(np.shape(indices_max)[1]):
        mat_res[indices_max[0, ind], indices_max[1, ind]] = 1

    if with_plots:
        plt.figure()
        plt.imshow(mat_res, interpolation="none", cmap="Greys")
        plt.plot(np.arange(np.shape(mat_res)[0]), np.arange(np.shape(mat_res)[0]))
        for l in loops:
            plt.scatter(l[1], l[0], s=80, facecolors='none', edgecolors='yellow')
            
    is_res, js_res = np.where(mat_res)
    ijs_res = np.vstack([is_res, js_res]).T
    
    mask = np.array(abs(ijs_res[:,0] - ijs_res[:,1])) > 15
    
    ijr_res_final = ijs_res[mask,:]
    
    """
    ijs_res = local_conv_max(res_rescaled)
    """

    
    # ijs is a 2-columns array of integers indicating the loops (i.e.the "1"s in mat_res)
    # (because mat_res is sparse, ijs encoding of loops has low memory footprint)


    return ijr_res_final


if (__name__ == '__main__'):
    indice_map = 1
    path_input = os.path.join(os.getcwd(), 'data')
    
    # Initial images
    loops = np.loadtxt(os.path.join(path_input, 'Raw_training/Loops_realisation_'+str(indice_map)+".txt"))
    realisation = np.loadtxt(os.path.join(path_input, 'Raw_training/MAT_RAW_realisation_'+str(indice_map)+".txt"))
    
    # Best mugshot
    pattern = np.loadtxt(os.path.join(path_input, "pattern_loop_median_detrended.txt"))

    ijs_res = pattern_finder(realisation, pattern, with_plots=True)