#!/usr/bin/env python3

import numpy as np
import itertools
import hicstuff as hcs
import pathlib
from matplotlib import pyplot as plt
from scipy.ndimage import measurements
from log import logger
from vizmap import plot_matrix


def diag_zscores(M):
    """Compute a trend and std for each diagonal at equal distances,
    then returns the appropriate z-score for each pixel.
    """

    matrix = np.array(M, dtype=np.float64)
    n, m = matrix.shape
    assert n == m
    diagonal_trend = np.array(
        [np.mean(np.diagonal(matrix, j)) for j in range(n)]
    )

    diagonal_stds = np.array(
        [np.std(np.diagonal(matrix, j)) for j in range(n)]
    )

    # logger.info(diagonal_trend)
    # logger.info(diagonal_stds)

    D = np.zeros_like(matrix)
    for i, j in itertools.product(range(n), range(n)):
        d = np.int64(np.abs(i - j))
        std = diagonal_stds[d]
        mean = diagonal_trend[d]
        # logger.debug(f"{i}, {j}, {d}, {std}, {mean}")
        # D[i, j] = max((matrix[i, j] - mean) / std, 0)
        D[i, j] = (matrix[i, j] - mean) / std
    return D


def iterative_threshold(matrix, n_points):
    median = np.amax(matrix)
    for thresh in np.linspace(0, median, n_points):
        logger.info(f"thresh = {thresh}")
        thresholded = np.copy(matrix)
        thresholded[thresholded < thresh] = 0
        yield thresholded


def main():

    working_dir = pathlib.Path("/home/pepito/Hackathon-ENGiE-PASTEUR-master")
    training_set_dir = pathlib.Path("Training_Set/TRAINING_SET")
    dataset_name = pathlib.Path("MAT_RAW_realisation_1215.txt")
    dataset = working_dir / training_set_dir / dataset_name
    M = np.genfromtxt(dataset)
    N = hcs.normalize_dense(M)
    loops = diag_zscores(M)
    loops[np.isnan(loops)] = 0

    plot_matrix(loops, vmin=-3, filename="", cmap='seismic')

    N_POINTS = 20

    for matrix in iterative_threshold(loops, N_POINTS):
        
        features, _ = measurements.label(matrix)
        labels_diag = np.diag(features)
        filtered_features = np.copy(features)
        logger.info(filtered_features)
        for label in labels_diag:
            filtered_features[filtered_features == label] = 0

        filtered_features[filtered_features > 0] = 1
        filtered_matrix = matrix * filtered_features
        logger.info(f"filtered matrix : {filtered_matrix}")
        logger.info(f"nonfiltered matrix: {matrix}")
        plot_matrix(matrix, vmin=0)
        plot_matrix(filtered_matrix)



if __name__ == "__main__":
    main()
