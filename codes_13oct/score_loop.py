# -*- coding: utf-8 -*-
"""
Created on Wed Oct 10 18:26:00 2018
@author: axel KournaK
To compute a score for predicted loops and simulated loops at pixel scale. 
"""
import numpy as np
# ex:
#list_predited = loadtxt("/home/axel/Bureau/HACK/TRAINING_SET/Loops_prediction_1.txt")
#list_real = loadtxt("/home/axel/Bureau/HACK/TRAINING_SET/Loops_realisation_1.txt")
#n1 = 289
#area = 3

def score_loop(list_predited, list_real, n1, area):
    MAT_PREDICT = np.zeros( (n1+area*2,n1+area*2) )
    for l in list_predited :
        p1 = int(l[0])
        p2 = int(l[1])
        MAT_PREDICT[ np.ix_(range( p1-area, p1+area+1 )  , range( p2-area, p2+area+1)   ) ]  += 1
        
    nb_loops_found = 0    
    for l in list_real :
        p1 = int(l[0])
        p2 = int(l[1])
        bool_find = 0
        MAT_REAL = np.zeros( (n1+area*2,n1+area*2) )
        MAT_REAL[ np.ix_(range( p1-area, p1+area+1 )  , range( p2-area, p2+area+1)   ) ]  += 1
        bool_find = ( MAT_REAL * MAT_PREDICT).sum()
        if bool_find > 0 :
            nb_loops_found +=1
    if  len(list_predited) > 0 :      
        PREC =   nb_loops_found / float( len(list_predited)  )
    else :
        PREC = "NA"    
    if  len(list_real) > 0 :
        RECALL = nb_loops_found / float( len(list_real)      )
    else : 
        RECALL = "NA"
    if  PREC != "NA" and RECALL != "NA"  and PREC != 0 and RECALL != 0:  
        F1 =     2* (PREC * RECALL) / (PREC + RECALL)
    else : 
        F1 = "NA"
    return PREC, RECALL, F1