#!/usr/bin/env python3

import numpy as np
import itertools
import hicstuff as hcs
import pathlib
from matplotlib import pyplot as plt
from scipy.ndimage import measurements
from scipy.ndimage import filters
import haar_filter
from log import logger
from vizmap import plot_matrix

N_POINTS = 30
REALIZATION_NUMBER = np.random.random_integers(low=1, high=2000)
WAVELET_PERCENTILE_CUTOFF = 96


def diag_zscores(M):
    """Compute a trend and std for each diagonal at equal distances,
    then returns the appropriate z-score for each pixel.
    
    Arguments:
        M {numpy.ndarray} -- Input matrix
    
    Returns:
        [numpy.ndarray] -- The matrix with z-scores
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
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    return D


def detrend(matrix):
    matrix = np.array(matrix, dtype=np.float64)
    n, m = matrix.shape
    assert n == m
    diagonal_trend = np.array(
        [np.mean(np.diagonal(matrix, j)) for j in range(n)]
    )

    # logger.info(diagonal_trend)
    # logger.info(diagonal_stds)

    D = np.zeros_like(matrix)
    for i, j in itertools.product(range(n), range(n)):
        d = np.int64(np.abs(i - j))

        mean = diagonal_trend[d]
        # logger.debug(f"{i}, {j}, {d}, {std}, {mean}")
        # D[i, j] = max((matrix[i, j] - mean) / std, 0)
        D[i, j] = matrix[i, j] / mean
    D[np.isnan(D)] = 0
    D[np.isinf(D)] = 0
    return D


def keep_biggest_connected_component(matrix):
    features, n_features = measurements.label(matrix)

    def feature_count(label):
        return features[features == label].sum()

    max_label = max(range(n_features), key=feature_count)
    logger.info(f"Maximum label: {feature_count(max_label)}")

    connected_matrix = np.copy(matrix)
    connected_matrix[features != max_label] = 0

    return connected_matrix


def iterative_threshold(matrix, n_points, max_value=None):
    """From a single matrix, generate matrices with a different threshold.
    The threshold increases linearly from zero to the maximum value of the
    matrix.
    
    Arguments:
        matrix {numpy.ndarray} -- The matrix to generate from
        n_points {numpy.ndarray} -- How many matrices to generate

    Returns:
        [numpy.ndarray] -- The matrix with sub-threshold elements set to zero.
    """

    if max_value is None:
        max_value = np.amax(matrix)

    for thresh in np.linspace(0, max_value, n_points):
        logger.info(f"thresh = {thresh}")
        thresholded = np.copy(matrix)
        thresholded[thresholded < thresh] = 0
        yield thresholded, thresh


def remove_diagonal(matrix):

    """Remove all diagonal-connected elements from the matrix.
    All components containing at least one diagonal pixel are set to zero.

    Arguments:
        matrix {numpy.ndarray} -- Input matrix to process

    Returns:
        [numpy.ndarray] -- Matrix with diagonal elements removed.
    """

    features, _ = measurements.label(matrix)
    labels_diag = np.diag(features)
    filtered_features = np.copy(features)

    for label in labels_diag:
        filtered_features[filtered_features == label] = 0

    filtered_features[filtered_features > 0] = 1
    filtered_matrix = matrix * filtered_features
    return filtered_matrix


def remove_singletons(matrix):
    """Remove all singleton pixels from a matrix.
    
    Arguments:
        matrix {numpy.ndarray} -- Input filtered matrix to process
    """

    features, n_features = measurements.label(matrix)
    filtered_features = np.copy(matrix)
    for label in range(n_features):
        mask = features == label
        if mask.sum() == 1:
            filtered_features[mask] = 0

    filtered_matrix = matrix * filtered_features
    return filtered_matrix


def plot_matrix_with_loops(
    matrix, loops, vmax=None, title=None, cmap="Reds", show=False
):

    x_loops, y_loops = loops

    if vmax is None:
        vmax = np.percentile(matrix, 95)
    plt.figure()
    plt.imshow(matrix, vmin=0, cmap=cmap, vmax=vmax)
    # plt.colorbar()
    plt.scatter(
        x=x_loops, y=y_loops, s=80, facecolors="none", edgecolors="blue"
    )
    if title is not None:
        plt.title(title)
    if show:
        plt.show()


def main():

    working_dir = pathlib.Path("/home/pepito/Hackathon-ENGiE-PASTEUR-master")
    training_set_dir = pathlib.Path("Training_Set/TRAINING_SET")
    dataset_name = pathlib.Path(
        f"MAT_RAW_realisation_{REALIZATION_NUMBER}.txt"
    )
    dataset_loops_name = pathlib.Path(
        f"Loops_realisation_{REALIZATION_NUMBER}.txt"
    )

    dataset = working_dir / training_set_dir / dataset_name
    dataset_loops = working_dir / training_set_dir / dataset_loops_name

    loops = np.genfromtxt(dataset_loops).T
    initial_matrix = np.array(np.genfromtxt(dataset))
    normalized_matrix = hcs.normalize_dense(initial_matrix)

    detrended_matrix = detrend(normalized_matrix)

    haar_filtered_matrix = haar_filter.haar_filter(
        detrended_matrix, thres_percentile=WAVELET_PERCENTILE_CUTOFF
    )
    #    connected_matrix = keep_biggest_connected_component(detrended_matrix)

    x_loops, y_loops = loops
    assert len(x_loops) == len(y_loops) > 2

    for M in (
        initial_matrix,
        normalized_matrix,
        detrended_matrix,
        haar_filtered_matrix,
    ):

        plot_matrix_with_loops(M, loops=loops, title="Initial matrix")

    plt.show()

    exit()

    plot_matrix_with_loops(
        diag_zscores(initial_matrix),
        loops=loops,
        title="Z-scored matrix",
        show=True,
    )

    gaussianized_matrix = filters.gaussian_filter(initial_matrix, sigma=.5)

    # plot_matrix(gaussianized_matrix)

    for matrix, thresh in iterative_threshold(haar_filtered_matrix, N_POINTS):
        break
        matrix_without_diagonal = remove_diagonal(matrix)
        z_scored_matrix = diag_zscores(matrix_without_diagonal)

        if np.sum(matrix_without_diagonal) == 0 and thresh > 0:
            logger.warning("Matrix is all zeros!")
            break

        title_original = f"Original matrix (thresh = {thresh})"
        title_threshold = (
            f"Thresholded matrix (thresh = {thresh})" " without diagonal"
        )
        title_zscore = (
            f"Z-scored matrix (thesh = {thresh})" " with diagonal removed"
        )

        assert np.isnan(matrix).sum() == 0
        assert np.isinf(matrix).sum() == 0
        assert np.isnan(matrix_without_diagonal).sum() == 0
        assert np.isinf(matrix_without_diagonal).sum() == 0
        assert np.isnan(z_scored_matrix).sum() == 0
        assert np.isinf(z_scored_matrix).sum() == 0

        plot_matrix_with_loops(
            matrix, loops=loops, title=title_original, vmax=1
        )
        plot_matrix_with_loops(
            matrix_without_diagonal, loops, title=title_threshold, vmax=1
        )
        print(matrix_without_diagonal)
        plot_matrix_with_loops(
            z_scored_matrix, loops=loops, title=title_zscore, vmax=3
        )

    plt.show()


if __name__ == "__main__":
    main()
