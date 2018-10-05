#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct  5 00:41:05 2018

@author: zx5281
"""

# Import libraries
import os
import numpy as np
import matplotlib.pyplot as plt
from pylab import loadtxt
import cv2
from PIL import Image
import imageio

# ...................... #
# ... Display matrix ... #
# ...................... #

def plot_matrix(M, filename, vmax=99):

    del filename
    plt.figure()
    plt.imshow(
        M, vmax=np.percentile(M, vmax), cmap="Reds", interpolation="none"
    )
    plt.colorbar()
    plt.axis("off")
    plt.show()

# .............................. #
# ... Normalisation function ... #
# .............................. #

# A is the matrix to normalise  
# the treshold to remove bins for the normalization (these bins are replaced by zeros vectors)
     
def scn_func(A,threshold):  
    
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
        print(n);
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

# .................... #
# ... Find suspect ... #
# .................... #

def pattern_finder2(loops, realisation, pattern, with_plots=False):

    matscn = scn_func(realisation) # noramlize
    dist = distance_law_human(matscn) # detrend
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

    pattern = np.loadtxt("pattern_loop_median_detrended.txt")

    if with_plots:
        plt.figure()
        plt.imshow(pattern**0.2, interpolation="none", cmap="afmhot_r")

    # conversion
    imageio.imsave('matrice.png', MAT_DETREND)
    imageio.imsave('template.png', pattern)
    img = cv2.imread('matrice.png')
    tp = cv2.imread('template.png')
    res = cv2.matchTemplate(img, tp, cv2.TM_CCOEFF_NORMED)

    indices_max = np.where(res > (0.5*res.max()))
    indices_max = np.array(indices_max)
    # mat_res is a binary image such that
    # 1 <-> loop
    # 0 <-> no-loop
    mat_res = np.zeros(np.shape(res), int)
    for ind in range(np.shape(indices_max)[1]):
        mat_res[indices_max[0, ind], indices_max[1, ind]] = 1

    if with_plots:
        plt.figure()
        plt.imshow(mat_res, interpolation="none", cmap="Greys")
        plt.plot(np.arange(np.shape(mat_res)[0]), np.arange(np.shape(mat_res)[0]))
        for l in loops:
            plt.scatter(l[1], l[0], s=80, facecolors='none', edgecolors='yellow')

    # ijs is a 2-columns array of integers indicating the loops (i.e.the "1"s in mat_res)
    # (because mat_res is sparse, ijs encoding of loops has low memory footprint)
    is_res, js_res = np.where(mat_res)
    ijs_res = np.vstack([is_res, js_res]).T

    return ijs_res


if __name__ == "__main__":
    
    # Select random raw file
    path_input = os.path.join(os.getcwd(), 'data')
    file = os.path.join(path_input, 'Raw_training/MAT_RAW_realisation_1200.txt')
    
    # Study raw file
    #raw_file = loadtxt(file)
    #plot_matrix(raw_file, 'test.txt')
    
    # Normalize file
    #normalized = scn_func(raw_file, 16)
    #plot_matrix(normalized, 'test.txt')
    
    # Detrended file
    #detrended = dist_law(normalized)
    # plot_matrix(detrended, 'test.txt')
    
    # Find suspect
    #file_mugshot = os.path.join(path_input, 'pattern_loop_median.txt')
    #find_suspect(file_mugshot, file)
    #img = file_mugshot = os.path.join(path_input, 'pattern_loop_median.txt')
    #template = file
    #result = cv2.matchTemplate(np.array(img), np.array(template), cv2.TM_SQDIFF)
    
    indice_map = 1
    loops = np.loadtxt(os.path.join(path_input, 'Raw_training/Loops_realisation_'+str(indice_map)+".txt"))
    realisation = np.loadtxt(os.path.join(path_input, 'Raw_training/MAT_RAW_realisation_'+str(indice_map)+".txt"))
    pattern = np.loadtxt(os.path.join(path_input, 'pattern_loop_median.txt'))
    plot_matrix(pattern, 'test.txt')

    #ijs_res = pattern_finder2(realisation, loops, pattern, with_plots=True)

    
