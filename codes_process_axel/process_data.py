# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 12:01:59 2018
@author: axel
"""

import numpy
import os
import matplotlib.pyplot as plt
os.chdir("/home/axel/repo/Hackathon-ENGiE-PASTEUR/codes_process_axel/")
import scn_human
import distance_law_human

n_patterns = 0
MAT_SUM = np.zeros( (area*2, area*2) )
for indice_map in range(1, 2001) :
    matscn = loadtxt("/home/axel/Bureau/TRAINING_SET/MAT_SCN_realisation_"+str(indice_map)+".txt")
    name = "/home/axel/Bureau/TRAINING_SET/MAT_SCN_realisation_"+str(indice_map)+".txt"
    dist = distance_law_human.dist_law(matscn) 
    n1 = shape(m)[0]
    # Computation of genomic distance law matrice:
    MAT_DIST =  np.zeros((n1, n1))
    for i in range(0,n1) :
        for j in range(0,n1) :
            MAT_DIST[i,j] =  dist[abs(j-i)] 
        
    MAT_DETREND = matscn / MAT_DIST
    MAT_DETREND[np.isnan(MAT_DETREND)] = 1.0
    loops = loadtxt("/home/axel/Bureau/TRAINING_SET/Loops_realisation_"+str(indice_map)+".txt") 
    nl= 0
    area = 8
    for l in loops :
        nl += 1
        p1 = int(l[0])
        p2 = int(l[1])
        if p1-area >= 0 and p1+area < n1 and p2-area >= 0 and p2+area < n1 :
            n_patterns += 1
            MAT_SUB = matscn[np.ix_(range(p1-area,p1+area),  range(p2-area, p2+area) ) ]
            MAT_SUM = MAT_SUM + MAT_SUB
print(n_patterns)    
imshow(MAT_SUB, interpolation = "none", vmin = 0., vmax = 2., cmap ="seismic" )       
    
imshow(MAT_SUB**0.15, interpolation = "none", cmap ="afmhot_r" )

savetxt("pattern_loop1.txt",MAT_SUB)

   
abs(MAT_DETREND.min() )
MAT_DETREND = MAT_DETREND + abs(MAT_DETREND.min() )    

imshow(MAT_DETREND, vmin = 0., vmax = 2.0, cmap = "seismic" , interpolation ="none") 

imshow(MAT_DETREND, vmin = 0., vmax = MAT_DETREND.max()*0.9, cmap = "afmhot_r" , interpolation ="none")    

indice_map = 2000
loops = loadtxt("/home/axel/Bureau/TRAINING_SET/Loops_realisation_"+str(indice_map)+".txt") 
nl= 0
area = 8
MAT_SUM = np.zeros( (area*2, area*2) )
for l in loops :
    nl += 1
    print(nl,  l[0], l[1])
    p1 = int(l[0])
    p2 = int(l[1])
    MAT_SUB = matscn[np.ix_(range(p1-area,p1+area),  range(p2-area, p2+area) ) ]
    MAT_SUM = MAT_SUM + MAT_SUB 
    plt.figure(nl)
    imshow(MAT_SUB, interpolation = "none")

close('all')

imshow(MAT_SUM, interpolation = "none")
imshow(matscn**.2, interpolation = "none", cmap = "afmhot_r")

imshow(m2**.2, interpolation = "none", cmap = "afmhot_r") 

