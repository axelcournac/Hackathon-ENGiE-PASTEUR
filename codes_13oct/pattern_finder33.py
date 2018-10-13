# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 21:45:06 2018
@author: axel c
33: without smoothing
"""
import matplotlib.pyplot as plt
import time
import os
import numpy as np
import pickle
import random

from scipy.ndimage import measurements
from sklearn.linear_model import LogisticRegression
import itertools


def access_data(nb_maps, raw_dir, loop_dir=None, data_source='txt'):
    if (loop_dir is None):
        loop_dir = raw_dir
    # sanity check
    if not(nb_maps > 0):
        raise ValueError('nb_maps must be greater than 0')
    if not(data_source in ['txt', 'npz']):
        raise ValueError("data_source must be 'txt' or 'npz'")
    if not(os.path.isdir(raw_dir)):
        raise ValueError('raw_dir does not exist')
    if not(os.path.isdir(loop_dir)):
        raise ValueError('loop_dir does not exist')
    raws_npz_fname = 'raws_'+str(nb_maps)+'.npz'
    loops_npz_fname = 'loops_'+str(nb_maps)+'.npz'
    if (data_source == 'txt'):
        raws = np.zeros([nb_maps, 289, 289])
        loops = np.zeros([nb_maps, 289, 289], bool)
        print('Loading *.txt files ...')
        tic = time.time()
        for mm in range(nb_maps):
            # read raw data
            fname = "MAT_RAW_realisation_"+str(mm+1)+".txt"
            full_fname = os.path.join(raw_dir, fname)
            raw = np.loadtxt(full_fname)
            raws[mm, :, :] = raw
            # read loop data
            fname = "Loops_realisation_"+str(mm+1)+".txt"
            full_fname = os.path.join(loop_dir, fname)
            loop = np.loadtxt(full_fname, int)
            for ll in range(loop.shape[0]):
                ii, jj = loop[ll, :]
                if ((jj >= ii) and (0<= ii <289) and (0<= jj <289)):
                    loops[mm, ii, jj] = True
        toc = time.time()
        print('done in', toc-tic, 'sec')
        print('Saving *.npz files ...')
        tic = time.time()
        np.savez_compressed(raws_npz_fname, raws)
        np.savez_compressed(loops_npz_fname, loops)
        toc = time.time()
        print('done in', toc-tic, 'sec')
    elif (data_source == 'npz'):
        tic = time.time()
        print('Loading *.npz files ...')
        raws = np.load(raws_npz_fname)['arr_0']
        loops = np.load(loops_npz_fname)['arr_0']
        toc = time.time()
        print('done in', toc-tic, 'sec')
    return raws, loops


def extract_vignets(raws, loops, V):
    """
    vignet extraction for logistic regression
    """
    # (maximal) number of vignets WITH loops
    nb_loops = loops.sum()
    # (maximal) number of vignets WITHOUT loops
    nb_noloops = 6 * nb_loops
    vignetsL = np.zeros([nb_loops, 2*V+1, 2*V+1]) + np.nan
    vignetsNL = np.zeros([nb_noloops, 2*V+1, 2*V+1]) + np.nan
    vl = 0
    vnl = 0
    for mm in range(nb_maps):
        raw = raws[mm, :, :]
        # generate vignets WITH loops
        loop_ij = np.array(np.where(loops[mm, :, :])).T
        L = loop_ij.shape[0]
        for ll in range(L):
            ii, jj = loop_ij[ll, :]
            if not((ii > V) & (jj > V) & (ii+V < 289) & (jj+V < 289)):
                continue
            try:
                vignet_loop = raw[(ii-V): (ii+V+1), (jj-V): (jj+V+1)]
            except:
                print('WARNING cannot extract vignet around', ii, jj)
                continue
            # sanity check
            if not((vignet_loop.shape[0] == 2*V+1)and(vignet_loop.shape[1] == 2*V+1)):
                raise ValueError('vignet shape is wrong')
            vignetsL[vl, :, :] = vignet_loop
            vl += 1
        # generate vignets WITHOUT loops (generate ranomly)
        for nl in range(6*L):
            ii = random.randint(V, 289-V)
            try:
                jj = random.randint(min(ii+dist_min, 289-V), min(ii+dist_max, 289-V))
            except:
                continue
            # sanity check
            if not((ii > V) & (jj > V) & (ii+V < 289) & (jj+V < 289)):
                continue
            try:
                vignet_noloop = raw[(ii-V): (ii+V+1), (jj-V): (jj+V+1)]
            except:
                print('WARNING cannot extract vignet around', ii, jj)
                continue
            vignetsNL[vnl, :, :] = vignet_noloop
            vnl += 1

    vignetsL = vignetsL[:vl, :, :]
    vignetsNL = vignetsNL[:vnl, :, :]

    return vignetsL, vignetsNL


def train_logistic(vignetsL, vignetsNL):
    #build training matrices when there are Loops
    X_loop = np.zeros([vignetsL.shape[0], (2*V+1)**2])
    for ii in range(vignetsL.shape[0]):
        X_loop[ii, :] = np.ravel(vignetsL[ii, :, :])
    y_loop = np.ones(X_loop.shape[0], int)
    #build tarining matrices when there are NO Loops
    X_noloop = np.zeros([vignetsNL.shape[0], (2*V+1)**2])
    for ii in range(vignetsNL.shape[0]):
        X_noloop[ii, :] = np.ravel(vignetsNL[ii, :, :])
    y_noloop = np.zeros(X_noloop.shape[0], int)
    # concatenate training matrices
    X = np.vstack((X_loop, X_noloop))
    y = np.concatenate((y_loop, y_noloop))
    # train the model
    model = LogisticRegression(random_state=0, solver='lbfgs', verbose=0).fit(X, y)
    return model


def pattern_finder3(observe, dist_min, dist_max, medians, mads, model, smooth_target_res=0, output='ijs'):
    nb_coef = np.size(model.coef_)
    V = (int(np.sqrt(nb_coef))-1)//2
    # sanity check
    if not(len(observe.shape) == 2):
        raise ValueError('observe is not a 2d array')
    if not(output in ['proba', 'ijs']):
        raise ValueError("output must be 'proba' or 'ijs'")
    if not(type(smooth_target_res) is int):
        raise ValueError("smooth_target_res must be an integer")
    if not(smooth_target_res >= 0):
        raise ValueError("smooth_target_res must be a positive integer")
    observe_normalized = (observe-medians)/mads
    observe_normalized[~np.isfinite(observe_normalized)] = 0.0
    probas = np.zeros(observe.shape)
    
    for ii in range(V, observe.shape[0]-V):
        for jj in range(ii, observe.shape[1]-V):
            dist = jj-ii
            if (dist < dist_min) or (dist > dist_max):
                continue
            vignet = observe_normalized[(ii-V): (ii+V+1), (jj-V): (jj+V+1)]
            proba = model.predict_proba(np.reshape(vignet, [1, -1]))
            probas[ii, jj] = proba[0, 1]
            
    if (output == 'proba'):
        return probas
    if (output == 'ijs'):
        raw_ijs = np.array(np.where(probas > 0.90)).T
        if( len(raw_ijs[:, 0]) > 0) :
            I = max(raw_ijs[:, 0])
            J = max(raw_ijs[:, 1])
            candidate_p = np.zeros((I+1, J+1), bool)
            candidate_p[raw_ijs[:,0], raw_ijs[:,1]] = True
            labelled_mat, num_features = measurements.label(candidate_p)
            centered_ijs = np.zeros([num_features, 2], int)
            remove_p = np.zeros(num_features, bool)
            for ff in range(num_features):
                label_p = labelled_mat == ff
                # remove the label corresponding to non-candidates
                if (candidate_p[label_p].sum()==0):
                    remove_p[ff] = True
                    continue
                label_ijs = np.array(np.where(label_p)).T
                ijmax = np.argmax(probas[label_ijs[:,0], label_ijs[:,1]])
                centered_ijs[ff, 0] = label_ijs[ijmax,0]
                centered_ijs[ff, 1] = label_ijs[ijmax,1]
            centered_ijs = centered_ijs[~remove_p, :]
        else :
            centered_ijs = "NA"
        return centered_ijs

#------------------------------------------------------------------------------
if (__name__=='__main__'):
    training_set_dir = "/home/axel/Bureau/HACK/TRAINING_SET5"
    data_source = 'npz'
    nb_maps = 2000
    V = 5
    #############
    # access data
    #############
    raws,loops = access_data(nb_maps, training_set_dir, data_source=data_source)
    ########################
    # basic loops statistics
    ########################
    nb_loops = loops.sum()
    nb_loops_by_ij = loops.sum(0)
    nb_loops_by_dist = np.zeros(289)
    for dd in range(289) :
        nb_loops_by_dist[dd] = np.diag(nb_loops_by_ij, dd).sum(0)
    # domain of existence of loops
    dist_min = np.where(nb_loops_by_dist > 0)[0][0]
    dist_max = np.where(nb_loops_by_dist > 0)[0][-1]
    np.save('dist_min.npy', dist_min)
    np.save('dist_max.npy', dist_max)

    ################
    # normalize data
    ################
    medians = np.median(raws, 0)
    mads = np.median(abs(raws-medians), 0)
    raws_normalized = (raws-medians)/mads
    raws_normalized[~np.isfinite(raws_normalized)] = 0.0
    np.save('medians.npy', medians)
    np.save('mads.npy', mads)

    #########################################
    # extract vignets for logistic regression
    #########################################
    vignetsL, vignetsNL = extract_vignets(raws_normalized, loops, V)

    #########################
    # run logistic regression
    #########################
    model = train_logistic(vignetsL, vignetsNL)
    pickle.dump(model, open('model.pkl', 'wb'))

    #######
    # Plots
    #######
    smooth_target_res = 0

    # Plot 1) display the weights learnt by logistic regression
    plt.figure()
    plt.imshow(np.reshape(model.coef_, (2*V+1, 2*V+1)))
    plt.colorbar()
    plt.title('Learnt logistic regression kernel')

    # Plot 2) superpose
    # - loop probabilities obtained by logistic regression
    # - actual loop localizaations (small circles)
    raw_test = raws[-1, :, :]
    loop_test = np.array(np.where(loops[-1, :, :])).T
    # pattern_finder3(raw) normalizes raw internally;
    # hence do NOT provide normalised_raw as input !
    probas = pattern_finder3(raw_test, dist_min, dist_max, medians, mads, model, smooth_target_res, 'proba')
    plt.figure()
    plt.imshow(probas)
    plt.colorbar()
    for ll in range(loop_test.shape[0]):
        loop = loop_test[ll, :]
        plt.scatter(loop[1], loop[0], s=80, facecolors='none', edgecolors='yellow')
    plt.title('Logistic regression probability heat map')

    # Plot 3) superpose
    # - loop localizations  obtained by logistic regression (probability>90%)
    # - actual loop localizaations (small circles)
    ijs = pattern_finder3(raw_test, dist_min, dist_max, medians, mads, model, smooth_target_res, 'ijs')
    plt.figure()
    plt.plot(ijs[:, 1], ijs[:, 0], '+')
    for ll in range(loop_test.shape[0]):
        loop = loop_test[ll, :]
        plt.scatter(loop[1], loop[0], s=80, facecolors='none', edgecolors='green')
    plt.show()
