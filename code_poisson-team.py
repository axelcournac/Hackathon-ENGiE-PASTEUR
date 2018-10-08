import numpy as np
import scipy.signal as sci
import scipy.stats as stats
import pandas as pd
import matplotlib.pyplot as plt
import glob as globfolder = "/home/remi/Desktop/Pasteur/Hackaton/Hackathon-ENGiE-PASTEUR/normalized/"
N = 289
count=0

# Compute the mean matrix of all the simulations
mean_matrix = np.zeros((N, N))
for f in glob.glob(folder+"/*"):
   aux = np.loadtxt(f, dtype=np.float)
   mean_matrix = np.add(mean_matrix, aux)
   count += 1
   if count % 500 == 0: print(count)mean_matrix = np.divide(mean_matrix, count)
plt.imshow(matrix**0.2, cmap="afmhot_r", interpolation="none", vmin=0, vmax=0.8)
np.savetxt("/home/remi/Desktop/Pasteur/Hackaton/Hackathon-ENGiE-PASTEUR/mean_matrix.txt", mean_matrix)count = 0

# Compute the std matrix of all the simulations
l = [np.loadtxt(f, dtype=np.float) for f in glob.glob(folder+"/*")]
std_matrix = np.std(l, axis=0, ddof=1)
np.savetxt("/home/remi/Desktop/Pasteur/Hackaton/Hackathon-ENGiE-PASTEUR/std_matrix.txt", std_matrix)

# Center-reduce a matrix:
f = "/home/remi/Desktop/Pasteur/Hackaton/Hackathon-ENGiE-PASTEUR/normalized/MAT_RAW_realisation_1.txt"
mat = np.loadtxt(f, dtype=np.float)
mat_centered = np.subtract(mat, mean_matrix)
mat_centered_reduced = np.divide(mat_centered, std_matrix)
centered_reduced
# TO DO for every matrices: for f in glob.glob(folder+"/*"):
