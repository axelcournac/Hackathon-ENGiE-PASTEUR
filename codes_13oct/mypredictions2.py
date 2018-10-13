# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 20:49:27 2018
@author: GD5264
"""
import os
import time
import numpy as np
import pickle
os.chdir("/home/axel/Bureau/HACK/last_code_axel_breuer")
import pattern_finder33

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


def generate_myalgo():
    dist_min = np.load('dist_min.npy')
    dist_max = np.load('dist_max.npy')
    medians = np.load('medians.npy')
    mads = np.load('mads.npy')
    smooth_target_res = 3
    model = pickle.load(open('model.pkl', 'rb'))
    def myalgo(observe):
        return pattern_finder33.pattern_finder3(observe, dist_min, dist_max, medians, mads, model, smooth_target_res)
    return myalgo

if (__name__ == '__main__'):
    """
    This script calculates the applies your classification algorithm
    to the test cases in observe_dir.
    All you have to do, is to correctly define the 2 parameters:
        target_dir, predict_dir
    and run this script, et voilà !
    """
    # Set your parameters:
    # directory containing the observed contact map  *.txt files
    observe_dir = '/media/axel/RSG41/Hackathon-ENGiE-PASTEUR-master/TRAINING_SET5'
    # directory where the loops predicted by your algo will be stored
    predict_dir = '/media/axel/RSG41/Hackathon-ENGiE-PASTEUR-master/TRAINING_SET5'

    try:
        os.mkdir(predict_dir)
    except:
        pass

    data_source = 'txt'

    # sanity check
    if not(data_source in ['txt', 'npz']):
        raise ValueError('data_source must be txt or npz')
    raws = None
    raws_npz_fname = 'raws.npz'
    # List of all files that will be processed by your loop identification algo
    observe_fnames = select_txts(os.listdir(observe_dir), prefix='MAT_RAW_')
    if (data_source == 'txt'):
        nb_maps = len(observe_fnames)
        tic = time.time()
        for mm in range(nb_maps):
            observe_fname = observe_fnames[mm]
            # read target data files
            full_observe_fname = os.path.join(observe_dir, observe_fname)
            print('READ', full_observe_fname)
            observe = np.loadtxt(full_observe_fname)
            if (raws is None):
                raws = np.zeros([nb_maps, observe.shape[0], observe.shape[1]])
            raws[mm, :, :] = observe
        toc = time.time()
        print('done in', toc-tic, 'sec')
        print('Saving *.npz files ...')
        tic = time.time()
        full_raws_npz_fname = os.path.join(observe_dir, raws_npz_fname)
        np.savez_compressed(full_raws_npz_fname, raws)
        toc = time.time()
        print('done in', toc-tic, 'sec')
    elif (data_source == 'npz'):
        tic = time.time()
        print('Loading *.npz file ...')
        full_raws_npz_fname = os.path.join(observe_dir, raws_npz_fname)
        raws = np.load(full_raws_npz_fname)['arr_0']
        toc = time.time()
        print('done in', toc-tic, 'sec')

    # generate prediction function such that
    # output = myalgo(input)
    # input is the observed contact map
    # output are the coordinates of the identified loops

    myalgo = generate_myalgo()

    # generate predicted loops
    print('Processing maps ...')
    nb_maps = raws.shape[0]
    confusions = {}
    tic = time.time()
    for mm in range(nb_maps):
        observe = raws[mm, :, :]
        # make loop prediction
        predict = myalgo(observe)    #                    !!  Here the heart of the algo   !! 
        # sanity check
        if ( predict != "NA" ):
            predict = np.array(predict, int)
            # write the loops of your algorithm
            observe_fname = observe_fnames[mm]
            predict_fname = observe_fname.replace('MAT_RAW_realisation_','Loops_prediction_')
            full_predict_fname = os.path.join(predict_dir, predict_fname)
            print('WRITE', full_predict_fname)
            # np.save() seems to become unstable on Linux when there are many files (?!?)
            # hence we have commented the line below:
            # np.savetxt(full_predict_fname, predict, fmt='%i %i')
            # same as np.save() but potentially more stable on Linux (read above):
            with open(full_predict_fname, 'w') as fobj:
                for ii in range(predict.shape[0]):
                    fobj.write('%i %i\n' %(predict[ii,0], predict[ii,1]))
        else :
            observe_fname = observe_fnames[mm]
            predict_fname = observe_fname.replace('MAT_RAW_realisation_','Loops_prediction_')
            full_predict_fname = os.path.join(predict_dir, predict_fname)
            print('WRITE', full_predict_fname)
            with open(full_predict_fname, 'w') as fobj:
                fobj.write('00\n')
    toc = time.time()
    print('done in', toc-tic, 'sec')
    
    
    
    
    
    