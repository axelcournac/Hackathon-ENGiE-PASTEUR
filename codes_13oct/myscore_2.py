# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:28:02 2018
@author: GD5264
2: la méthode du calcul de PREC et RECALL est basé sur les évènement de loops et plus les pixels. 
Also: modify to allow zero loops detected
"""
import os
import numpy as np
import time
import itertools
import score_loop

def select_txts(fnames, prefix=None):
    """
    given a list of filenames, return the sublist ending with '.txt'
    (and starting with prefix)
    """
    fnames_txt = []
    for fname in fnames:
        if (len(fname.split('.')) == 2):
            if (fname.split('.')[1] == 'txt'):
                if (prefix is None):
                    fnames_txt.append(fname)
                elif (prefix in fname.split('.')[0]):
                    fnames_txt.append(fname)
    return sorted(fnames_txt)

if (__name__ == '__main__'):
    """
    This script calculates the score assigned to your classification algorithm.
    To do so, it:
        1. compares predicted loops to target loop
        2. builds the confusion matrix
        3. calculates the F1-score of the confusion matrix
    All you have to do, is to correctly define the 4 parameters:
        target_dir, predict_dir, shape, smooth_target_res
    and run this script, et voilà !
    """
    # Set your parameters:MYSTERIOUS_SET_PREDICTION_DONOUGHT
    #target_dir = 'C:\\Users\\gd5264\\Downloads\\TRAINING_SET'
#    target_dir = 'C:\\Users\\gd5264\\Downloads\\MYSTERIOUS_SET_ANSWERS\\loops_repo'  # directory containing the target *.txt files
#    target_dir = '/home/axel/Bureau/HACK/MYSTERIOUS_SET_ANSWERS/Loops_repo'  # directory containing the target *.txt files
    target_dir = '/media/axel/RSG41/Hackathon-ENGiE-PASTEUR-master/TRAINING_SET5'  # directory containing the target *.txt files
#    target_dir = '/home/axel/Bureau/HACK/MYSTERIOUS_SET_ANSWERS/Loops_repo'  
    
    #predict_dir = 'C:\\Users\\gd5264\\Downloads\\MYSTERIOUS_SET_PREDICTION_DONOUGHT'  # directory containing the predict *.txt files
    #predict_dir = 'C:\\Users\\gd5264\\Downloads\\MYSTERIOUS_SET_PREDICTION_MOUSQUETAIRES3'  # directory containing the predict *.txt files
    #predict_dir = 'C:\\Users\\gd5264\\Downloads\\MYSTERIOUS_SET_PREDICTION_FISH'  # directory containing the predict *.txt files
    #predict_dir = 'C:\\Users\\gd5264\\Downloads\\MYSTERIOUS_SET_PREDICTION_MOUSQUETAIRES2'  # directory containing the predict *.txt files
#    predict_dir = 'C:\\Users\\gd5264\\Downloads\\MYSTERIOUS_SET_PREDICTION_LOGISTIC'  # directory containing the predict *.txt files
    predict_dir = '/media/axel/RSG41/Hackathon-ENGiE-PASTEUR-master/TRAINING_SET5'  # directory containing the predict *.txt files
#    predict_dir = '/home/axel/Bureau/HACK/MYSTERIOUS_SET_PREDICTED/'  
    #predict_dir = 'C:\\Users\\gd5264\\Downloads\\TRAINING_SET'
    shape = [289, 289]  # size of images to be compared
    smooth_target_res = 0  # smoothing parameter (no smoothing when set to 0)

    print('your parameters:')
    print('----------------')
    print('target_dir=', target_dir)
    print('predict_dir=', predict_dir)
    print('shape=', shape)
    print('smooth_target_res=', smooth_target_res, '\n')

    target_fnames = select_txts(os.listdir(target_dir), prefix='Loops_realisation_')
    predict_fnames = [target_fname.replace('Loops_realisation_','Loops_prediction_') for target_fname in target_fnames]
    # hack for mousquetaires2
    #predict_fnames = [predict_fname.replace('.txt','_norm.txt') for predict_fname in predict_fnames]

    # sanity check
    listdir_predict_dir = os.listdir(predict_dir)
    for predict_fname in predict_fnames:
        if not(predict_fname in listdir_predict_dir):
            raise ValueError('There is not a one-to-one mapping of files in target_dir and predict_dir')

    # generate confusion matrices for each target file vs. predict file
    print('Comparing predictions to targets...')
    tic = time.time()
    PREC = {}
    RECALL = {}
    F1 = {}
    
    for ff in range(len(target_fnames)):
        # read target data files
        full_target_fname = os.path.join(target_dir, target_fnames[ff])
        #print('READ', full_target_fname)
        if (os.stat(full_target_fname).st_size == 0):
            target = np.zeros([0, 2], int)
        else:
            target = np.loadtxt(full_target_fname, int)
        # sanity check on target file format
#        if not(len(target.shape) == 2):
#            raise ValueError(full_target_fname + ' has corrupted format')
        # sanity check on values
        if target.sum() != 0 :   #  there are positions of loops in the file    
            if np.any(target[:, 0] < 0):
                #raise ValueError('Some targets have i<0')
                print('Some targets in '+ target_fnames[ff] +' have i<0')
                target = target[target[:, 0] >= 0, :]
            if np.any(target[:, 0] >= shape[0]):
                raise ValueError('Some targets have i>=shape[0]')
            if np.any(target[:, 1] < 0):
                raise ValueError('Some targets have j<0')
            if np.any(target[:, 1] >= shape[1]):
                raise ValueError('Some targets have j>=shape[1]')
            if (target.shape[0] > 0):
                if not(target.shape[0] == np.unique(target, axis=0).shape[0]):
                    raise ValueError('There are duplicates in target')
    
            full_predict_fname = os.path.join(predict_dir, predict_fnames[ff])
            #print('READ', full_predict_fname, flush=True)
            if (os.stat(full_predict_fname).st_size == 0):
                predict = np.zeros([0, 2], int)
            else:
                predict = np.loadtxt(full_predict_fname, int)
            # sanity check on file format
            if (predict.size == 2):
                predict = np.reshape(predict, [1, 2])
    #        elif not(len(predict.shape) == 2):
    #            raise ValueError(full_predict_fname + ' has corrupted format')
            # sanity check on values
            if predict.sum() != 0 :   #  there are positions of loops in the file 
                if np.any(predict[:, 0] < 0):
                    raise ValueError('Some predictions have i<0')
                if np.any(predict[:, 0] >= shape[0]):
                    raise ValueError('Some predictions have i>=shape[0]')
                if np.any(predict[:, 1] < 0):
                    raise ValueError('Some predictions have j<0')
                if np.any(predict[:, 1] >= shape[1]):
                    raise ValueError('Some predictions have j>=shape[1]')
                if (predict.shape[0] > 0):
                    if not(predict.shape[0] == np.unique(predict, axis=0).shape[0]):
                        raise ValueError('There are duplicate in predict')      
                PREC[full_predict_fname] , RECALL[full_predict_fname] , F1[full_predict_fname]  = score_loop.score_loop(predict, target, 289, 3)

    toc = time.time()
    print('Parsed', len(target_fnames), 'files in', toc-tic, 'sec')

    # calculate average 
    PREC_avg = np.zeros((1, 1), float)
    RECALL_avg = np.zeros((1, 1), float)
    F1_avg = np.zeros((1, 1), float)
    nn = 0
    for key in PREC:
        if PREC[key] != "NA" and RECALL[key] != "NA" and F1[key] != "NA" and (nn == 0):
            PREC_avg += PREC[key]
            RECALL_avg += RECALL[key]
            F1_avg +=     F1[key]
            nn += 1
        else:
            if PREC[key] != "NA" and RECALL[key] != "NA" and F1[key] != "NA" :
                nn += 1
                PREC_avg = ((nn-1)/nn)*PREC_avg + (1/nn)*PREC[key]
                RECALL_avg = ((nn-1)/nn)*RECALL_avg + (1/nn)*RECALL[key]
                F1_avg = ((nn-1)/nn)*F1_avg + (1/nn)*F1[key]


    # calculate F1 score
#    PREC_avg, REC_avg, F1_avg = confusion2scores(confusion_avg)

    print('*******************************************')
    print('Your PREC Score =', PREC_avg, '(0.0 is bad; 1.0 is perfect)')
    print('Your REC Score =', RECALL_avg, '(0.0 is bad; 1.0 is perfect)')
    print('Your F1 Score =', F1_avg, '(0.0 is bad; 1.0 is perfect)')
    print('*******************************************')
