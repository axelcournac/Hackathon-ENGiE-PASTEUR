# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 17:28:02 2018

@author: GD5264
"""

import os
import numpy as np
import time
import itertools


def keep_txt(fnames):
    """
    given a list of filenames, return the sublist ending with '.txt'
    """

    fnames_txt = []
    for fname in fnames:
        if (len(fname.split('.')) == 2):
            if (fname.split('.')[1] == 'txt'):
                fnames_txt.append(fname)
    return sorted(fnames_txt)


def smooth(ijs, shape, res=0):
    """
    Given a sprase binary image such that
        image[i,j]=1 when (i,j) in ijs,
        image[i,j]=0 otherwise
    return the smoothed binary image, such that
        smooth_image[i,j] = 1 when (i +/- res ,j +/- res) in ijs,
        smooth_image[i,j] = 0 otherwise.

    INPUT:
    ijs: a numpy array of integers with 2 columns:
         ["i"=1st column, "j"=2nd column]
    shape: image boundaries (i in [0..shape[0]-1] and j in [0.. shape[1]-1])
    res: smoothing resolution (if res=0, then no-smoothing)

    OUTPUT:
    smooth_ijs: smoothed binary image, such that
        smooth_image[i,j] = 1 when (i,j) in smooth_ijs,
        smooth_image[i,j] = 0 otherwise.
    """

    # sanity check
    if (res < 0):
        raise ValueError('res must be positive')
    if not(type(res) is int):
        raise ValueError('res must be an integer')

    if (res == 0):
        return ijs
    kernel = [item for item in itertools.product(np.arange(-res, res+1), np.arange(-res, res+1))]
    kernel = np.array(kernel)
    smooth_ijs = np.zeros([0, 2], int)
    for ij in ijs:
        smooth_ij = ij + kernel
        smooth_ijs = np.vstack((smooth_ijs, smooth_ij))

    # clipping
    keep_p = (smooth_ijs[:, 0] >= 0) & (smooth_ijs[:, 0] < shape[0]) & (smooth_ijs[:, 1] >= 0) & (smooth_ijs[:, 1] < shape[1])
    smooth_ijs = smooth_ijs[keep_p, :]

    # avoid double counting of overlapping smoothed points
    smooth_ijs = np.unique(smooth_ijs, axis=0)
    return smooth_ijs


def confusion_matrix(target, predict, shape):
    """
    calculate the confusion matrix by comparing predict to target
    """

    target1 = target[:, 0] * shape[1] + target[:, 1]
    predict1 = predict[:, 0] * shape[1] + predict[:, 1]
    # sanity check
    target1_set = set(target1)
    if not(len(target1_set) == len(target1)):
        raise ValueError('target has duplicates')
    predict1_set = set(predict1)
    if not(len(predict1_set) == len(predict1)):
        raise ValueError('predict has duplicates')

    # True Positives
    inter1_set = target1_set.intersection(predict1_set)
    TP = len(inter1_set)
    # False Positives
    FP = len(predict1_set.difference(inter1_set))
    # False Negatives
    FN = len(target1_set) - TP
    # True Negatives
    TN = shape[0]*shape[1] - TP - FP - FN

    # sanity chek
    if (FN < 0):
        raise ValueError('False Negative counting mismatch')
    if (TN < 0):
        raise ValueError('True Negative counting mismatch')
    P = len(target1)  # Positives
    if not(FN + TP == P):
        raise ValueError('Positives counting mismatch')
    N = shape[0] * shape[1] - P  # Negatives
    if not(FP + TN == N):
        raise ValueError('Negatives counting mismatch')

    confusion = np.array([[TP, FP], [FN, TN]])

    return confusion


def confusion2F1(confusion):
    """
    calculate F1-score assigned to a confusion matrix
    """

    # unpack average confusion matrix
    TP, FP = confusion[0, :]
    FN, TN = confusion[1, :]
    # PRECision
    PREC = 0.0
    if ((TP + FP) > 0):
        PREC = TP/(TP+FP)
    # RECall
    REC = 0.0
    if ((TP + FN) > 0):
        REC = TP/(TP+FN)
    # F1 score
    F1 = 0.0
    if ((PREC + REC) > 0):
        F1 = 2*PREC*REC/(PREC+REC)
    return F1


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
    # Set your parameters:
    target_dir = '.\\Target'  # directory containing the target *.txt files
    predict_dir = '.\\Predict'  # directory containing the predict *.txt files
    shape = [289, 289]  # size of images to be compared
    smooth_target_res = 1  # smoothing parameter (no smoothing when set to 0)

    print('your parameters:')
    print('----------------')
    print('target_dir=', target_dir)
    print('predict_dir=', predict_dir)
    print('shape=', shape)
    print('smooth_target_res=', smooth_target_res, '\n')

    target_fnames = keep_txt(os.listdir(target_dir))
    predict_fnames = keep_txt(os.listdir(predict_dir))

    # sanity check
    if not(target_fnames == predict_fnames):
        raise ValueError('There is not a one-to-one mapping of files in target_dir and predict_dir')

    # generate confusion matrices for each target file vs. predict file
    tic = time.time()
    confusions = {}
    for fname in target_fnames:
        print('Parse', fname, flush=True)
        # read target data files
        target_fname = os.path.join(target_dir, fname)
        if (os.stat(target_fname).st_size == 0):
            target = np.zeros([0, 2], int)
        else:
            target = np.loadtxt(target_fname, int)
        # sanity check on target file format
        if not(len(target.shape) == 2):
            raise ValueError(target_fname + ' has corrupted format')
        # sanity check on values
        if np.any(target[:, 0] < 0):
            raise ValueError('Some targets have i<0')
        if np.any(target[:, 0] >= shape[0]):
            raise ValueError('Some targets have i>=shape[0]')
        if np.any(target[:, 1] < 0):
            raise ValueError('Some targets have j<0')
        if np.any(target[:, 1] >= shape[1]):
            raise ValueError('Some targets have j>=shape[1]')
        if (target.shape[0] > 0):
            if not(target.shape[0] == np.unique(target, axis=0).shape[0]):
                raise ValueError('There are duplicates in target')
        # smooth target
        smooth_target = smooth(target, shape, smooth_target_res)

        predict_fname = os.path.join(predict_dir, fname)
        if (os.stat(predict_fname).st_size == 0):
            predict = np.zeros([0, 2], int)
        else:
            predict = np.loadtxt(predict_fname, int)
        # sanity check on file format
        if not(len(predict.shape) == 2):
            raise ValueError(predict_fname + ' has corrupted format')
        # sanity check on values
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

        confusions[fname] = confusion_matrix(smooth_target, predict, shape)

    toc = time.time()
    print('Parsed', len(target_fnames), 'files in', toc-tic, 'sec')

    # calculate average confusion matrix
    confusion_avg = np.zeros((2, 2), float)
    nn = 0
    for key in confusions:
        nn += 1
        if (nn == 1):
            confusion_avg += confusions[key]
        else:
            confusion_avg = ((nn-1)/nn)*confusion_avg + (1/nn)*confusions[key]

    # calculate F1 score
    F1_avg = confusion2F1(confusion_avg)

    print('*******************************************')
    print('Your Score =', F1_avg, '(0.0 is bad; 1.0 is perfect)')
    print('*******************************************')
