# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:49:27 2018

@author: GD5264
"""

import os
import time
import numpy as np
import pattern_finder2_v2 as pattern_finder2


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


def myalgo(observe):
    #predict = np.zeros([1,2], int)
    #predict[0,0] = 100
    #predict[0,1] = 100
    predict = pattern_finder2.pattern_finder2(observe)
    return predict

# Set your parameters:
# target_dir = os.path.join(os.getcwd(), 'data/Raw_training')  # directory containing the target *.txt files
# predict_dir = path_input = os.path.join(os.getcwd(), 'data/Raw_training_predict_01')  # directory containing the predict *.txt files

if (__name__ == '__main__'):
    """
    This script calculates the applies your classification algorithm to the test cases in target_dir.
    
    All you have to do, is to correctly define the 4 parameters:
        target_dir, predict_dir
    and run this script, et voilà !
    """
    # Set your parameters:
    path_input = os.path.join(os.getcwd(), 'data')
    observe_dir = os.path.join(os.getcwd(), 'data/Raw_training')  # directory containing the target *.txt files # directory containing the target *.txt files
    predict_dir = os.path.join(os.getcwd(), 'data/Raw_training_predict')  # directory containing the predict *.txt files

    observe_fnames = select_txts(os.listdir(observe_dir), prefix='MAT_RAW_')

    # generate predicted loops
    tic = time.time()
    confusions = {}
    for observe_fname in observe_fnames:

        # read target data files
        full_observe_fname = os.path.join(observe_dir, observe_fname)
        print('READ', full_observe_fname)
        observe = np.loadtxt(full_observe_fname)

        predict = myalgo(observe)

        # sanity check
        if not(len(predict.shape)==2):
            raise ValueError('predict is not a 2 dimensional array')
        predict = np.array(predict, int)
        predict_fname = observe_fname.replace('MAT_RAW_realisation_','Loops_prediction_')
        full_predict_fname = os.path.join(predict_dir, predict_fname)
        print('WRITE', full_predict_fname, flush=True)
        np.savetxt(full_predict_fname, predict, fmt='%i %i')