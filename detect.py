#!python3
# coding: utf-8

# In[28]:
import sys
import os
import numpy as np
import scipy.signal as sci
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import glob as glob
from matplotlib.patches import Circle
import matplotlib.patches as patches
# get_ipython().run_line_magic('matplotlib', 'inline')
# plt.rcParams['figure.figsize'] = [10, 10]


from scipy.ndimage import label, generate_binary_structure

from pathlib import Path

# In[4]:


def deletes_small_objects(mat, n):
    """Supprime les objets plus petits que n pixels"""
    # creation du masque binaire
    mask = np.zeros(mat.shape, dtype=int)
    mask[mat>0] = 1
    # label (trouve les objets)
    mat_obj, n_obj = label(mask)
    # test la taille de chaque objet, et le supprime s'il est trop petit
    for i in range(1, n_obj+1):
        if np.sum(mat_obj==i) <= n:
            mat = mat * (mat_obj!=i)
    # suppression digaonale et objets connectés
    return mat


# matrix loading
def deletes_diag(mat):
    """Utilise la matrice de Vitto après thresholding"""
    # creation du masque binaire
    mask = np.zeros(mat.shape, dtype=int)
    mask[mat>0] = 1
    # remplissage de la diagonale (taille 1 ou 3 ou 5)
    for i in range(mask.shape[0]):
        mask[i, i] = 1
    # label
    mat_obj = label(mask, generate_binary_structure(2,2))[0]
    # where (mask binaire des objects connectés à la diagonale)
    cc = mat_obj!=1
    # suppression digaonale et objets connectés
    return cc*mat

def compute_fish_algo(matrix_path,out_fold):

    file_name = matrix_path
    folder = "/pasteur/homes/vscolari/hack"
    N = 289
    count=0


    # In[33]:
    mat = np.loadtxt("TRAINING_SET/" + matrix_path, dtype=np.float)
    mean_matrix = np.loadtxt(folder + "/mean_matrix.txt", dtype=np.float)
    std_matrix = np.loadtxt(folder + "/std_matrix.txt", dtype=np.float)
    mat_centered = np.subtract(mat, mean_matrix)
    mat_centered_reduced = np.divide(mat_centered, std_matrix)
    mat_centered_reduced[np.isnan(mat_centered_reduced)] = 0
    mat2 = mat_centered_reduced


    # In[34]:


    # plt.figure()
    # plt.imshow(mat2, cmap="afmhot_r", interpolation="none")#, vmin=0, vmax=0.5)


    # In[30]:


    x = np.floor(np.random.normal(8, 2, 1000)).astype(int)
    y = np.floor(np.random.normal(8, 2, 1000)).astype(int)
    z = np.zeros((16, 16))
    for i in range(1000):
     if x[i] < 16 and y[i] < 16 and x[i]> 0 and y[i] > 0:
         z[x[i],y[i]] += 1
    # plt.imshow(z, cmap="afmhot_r")


    # In[45]:


    res = np.zeros((289, 289))
    for i in range(289-16):
     for j in range(289-16):
         res[i+8,j+8] = stats.spearmanr(mat2[i:i+16,j:j+16].reshape(-1), z.reshape(-1))[0]
    # plt.figure()
    # plt.imshow(res, cmap="afmhot_r", interpolation="none")#, vmin=0.1)
    # plt.colorbar()#, vmin=0, vmax=0.8)


    # In[46]:


    std = 1./0.675 * np.median(np.abs(res - np.median(res)))


    # In[55]:


    cut = np.copy(res)
    cut[cut < np.median(cut) + 4.*std] = 0


    # In[57]:


    np.save(folder+"thresholded", cut)
    # plt.imshow(cut, cmap="afmhot_r", interpolation="none")#, vmin=0.1)
    # plt.colorbar()#, vmin=0, vmax=0.8)


    # In[54]:
    n = 0
    positions = []
    cut2 = np.copy(cut)
    cut3 = np.copy(cut)
    
    cut2 = deletes_diag(cut)
    new_mat_no_small = deletes_small_objects(cut2,5)

    while n < 100000:
      peak = np.unravel_index(np.argmax(new_mat_no_small), np.array(new_mat_no_small).shape)
      new_mat_no_small[peak[0]-8:peak[0]+8 , peak[1]-8:peak[1]+8 ] = 0
      if len(positions) > 0 and peak[0] == positions[-1][0] and peak[1] == positions[-1][1]:
          break
      if peak[0] < peak[1]:
          positions.append(peak)
      n = n + 1
    peaks = positions
    predict_fname = file_name.replace('MAT_RAW_realisation_','Loops_prediction_')
    predict_name_path = out_fold + "/" + predict_fname
    np.savetxt(predict_name_path, np.matrix(peaks),fmt="%d")


if __name__=='__main__':

    input_matrix = sys.argv[1]
    out_fold = "out_predic"

    compute_fish_algo(input_matrix,out_fold)
