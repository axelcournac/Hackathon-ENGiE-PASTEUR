# -*- coding: utf-8 -*-
"""
Created on %(date)s

@author: %(username)s
"""
%reset
#==============================================================================
# MODULES
#==============================================================================
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

pre = '/media/remi/Disk1/'
sys.path.append(pre+'Juicer/scripts/')
import straw

#==============================================================================
# FUNCTIONS
#==============================================================================
def scn_func(matrix,threshold):    
    """Normalizes a raw contact map (and sets underexpressed positions to 0)
    Input: a raw contact map (numpy array)
           the threshold discriminating under and correctly expressed position (int)
    Output: a normalized contact map (numpy array)"""
    dim = matrix.shape[0]
    #~ print('dim: {0}'.format(dim))
    n_iterations=10
    keep = np.zeros((dim, 1))
    
    for i in range(0,dim):
        if np.sum(matrix[i,]) > threshold:
            keep[i] = 1
        else :
            keep[i] = 0
    
    to_keep=np.where(keep >0 )[0]
    to_remove=np.where(keep <=0 )[0]

    for n in range(0,n_iterations):
#        print(n)
        for i in range(0,dim):
            matrix[to_keep,i]=matrix[to_keep,i]/ np.sum(matrix[to_keep,i])
            matrix[to_remove,i]=0   
        matrix[np.isnan(matrix)] = 0.0 
        
        for i in range(0,dim):    
            matrix[i,to_keep]=matrix[i,to_keep]/ np.sum(matrix[i,to_keep])
            matrix[i,to_remove]=0  
        matrix[np.isnan(matrix)] = 0.0 

    return matrix

def generate_matrix(results):
        size = max(max(result[0])/binnage, max(result[1])/binnage)
        mat = np.zeros((size, size))
        for i in range(len(result[0])):
            x = result[0][i]/binnage -1
            y = result[1][i]/binnage -1
            mat[x, y] = result[2][i]
            mat[y, x] = mat[x, y]
        return mat
        
        
#==============================================================================
# MAIN
#==============================================================================
infile = pre + 'Yeast/data/TADs/AT147/aligned_AT147_S288C/inter_30.hic'
normalization='NONE'
unit = 'BP'
binnage = 5000
output_dir = pre + 'Yeast/data/TADs/AT147/aligned_AT147_S288C/txt_matrices/'
chr_list = [str(i+1) for i in range(16)]

if not os.path.isdir(output_dir): 
    os.mkdir(output_dir)

for chrm in chr_list:
    # Extract data from .hic format
    result = straw.straw(normalization, infile, chrm, chrm, unit, binnage)

    # Put them in an array
    mat = generate_matrix(result)
    
    # SCN of the matrix
    mn = scn_func(mat, 0)
    
    # Save the matrix
    np.savetxt('{0}chr{1}.txt'.format(output_dir, chrm), mn)

#plt.figure()
#plt.imshow(mn**0.2, cmap='afmhot_r', interpolation='none')
#plt.close('all')

