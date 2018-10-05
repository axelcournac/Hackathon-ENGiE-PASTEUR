#!/usr/bin/env python3

import numpy as np
import itertools
import hicstuff as hcs
import pathlib
from matplotlib import pyplot as plt
from scipy.ndimage import measurements
import haar_filter
from log import logger

N_POINTS = 30
REALIZATION_NUMBER = np.random.random_integers(low=1, high=2000)
WAVELET_PERCENTILE_CUTOFF = 96
ALIGNMENT_PERCENTAGE = .1
FEATURE_SIZE = 7
NEIGHBORHOOD_KERNEL = [[1, 1, 1], [1, 1, 1], [1, 1, 1]]

WORKING_DIR = pathlib.Path("/home/pepito/Hackathon-ENGiE-PASTEUR-master")
TRAINING_SET_DIR = pathlib.Path("Training_Set/TRAINING_SET")
REAL_SET_NAME = pathlib.Path(
    "data/MAT_RAW_chr1_AT147_Pds5-AID-noTir1-G1-cdc20-TS_2kb.txt"
)
DATASET_NAME = pathlib.Path(f"MAT_RAW_realisation_{REALIZATION_NUMBER}.txt")

DATASET_LOOPS_NAME = pathlib.Path(
    f"Loops_realisation_{REALIZATION_NUMBER}.txt"
)

REAL_DATASET = WORKING_DIR / REAL_SET_NAME

DATASET = WORKING_DIR / TRAINING_SET_DIR / DATASET_NAME
DATASET_LOOPS = WORKING_DIR / TRAINING_SET_DIR / DATASET_LOOPS_NAME


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
        D[i, j] = (matrix[i, j] - mean) / std ** 2
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


def remove_singletons(matrix, feature_size=1, kernel=None):
    """Remove all singleton pixels from a matrix.

    Arguments:
        matrix {numpy.ndarray} -- Input filtered matrix to process
    """

    features, n_features = measurements.label(matrix, structure=kernel)
    filtered_features = np.copy(matrix)
    for label in range(n_features):
        mask = features == label
        if mask.sum() <= feature_size:
            filtered_features[mask] = 0

    filtered_matrix = matrix * filtered_features
    return filtered_matrix


def removed_aligned_features(matrix, alignment_percentage):

    n, m = matrix.shape
    assert n == m

    filtered_matrix = np.copy(matrix)
    for i in range(n):
        diag = np.diagonal(matrix, i)
        if (diag > 0).sum() / len(diag) >= alignment_percentage:
            d = len(diag)
            filtered_matrix[np.arange(d), np.arange(d) + i] = 0

    triu_matrix = np.triu(filtered_matrix)
    final_matrix = (triu_matrix + triu_matrix.T) / 2

    return final_matrix


def clip_matrix(matrix, offset=None):
    if offset is None:
        offset = len(matrix) // 25
    clipped_matrix = np.zeros_like(matrix)
    clipped_matrix[offset:-offset, offset:-offset] = matrix[
        offset:-offset, offset:-offset
    ]
    return clipped_matrix


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


def matrix_to_score(matrix, filename=None):
    if filename is None:
        return zip(*np.nonzero(matrix))
    else:
        with open(filename, "w") as handle:
            for couple in zip(*np.nonzero(matrix)):
                x, y = couple
                handle.write(f"{x} {y}\n")


def do_everything(initial_matrix):
    normalized_matrix = hcs.normalize_dense(initial_matrix)

    detrended_matrix = detrend(normalized_matrix)

    # z_scored_matrix = diag_zscores(normalized_matrix)

    haar_filtered_matrix = haar_filter.haar_filter(
        detrended_matrix, thres_percentile=WAVELET_PERCENTILE_CUTOFF, level=1
    )

    alignment_filtered_matrix = removed_aligned_features(
        haar_filtered_matrix, alignment_percentage=ALIGNMENT_PERCENTAGE
    )

    singleton_filtered_matrix = remove_singletons(
        alignment_filtered_matrix, feature_size=FEATURE_SIZE
    )

    clipped_matrix = clip_matrix(singleton_filtered_matrix)

    return clipped_matrix


def plot_everything(initial_matrix, dataset_loops):
    normalized_matrix = hcs.normalize_dense(initial_matrix)

    detrended_matrix = detrend(normalized_matrix)

    haar_filtered_matrix = haar_filter.haar_filter(
        detrended_matrix, thres_percentile=WAVELET_PERCENTILE_CUTOFF, level=1
    )

    alignment_filtered_matrix = removed_aligned_features(
        haar_filtered_matrix, alignment_percentage=ALIGNMENT_PERCENTAGE
    )

    singleton_filtered_matrix = remove_singletons(
        alignment_filtered_matrix, feature_size=FEATURE_SIZE
    )

    clipped_matrix = clip_matrix(singleton_filtered_matrix)

    loops = np.genfromtxt(dataset_loops).T

    for title, M in zip(
        (
            "Initial matrix",
            "Normalized matrix",
            "Detrended matrix",
            "Haar filtered matrix",
            "Z scored haar filtered matrix",
            "Aligned elements removed matrix",
            "Singleton filtered matrix",
            "Clipped matrix",
        ),
        (
            initial_matrix,
            normalized_matrix,
            detrended_matrix,
            haar_filtered_matrix,
            diag_zscores(haar_filtered_matrix),
            alignment_filtered_matrix,
            singleton_filtered_matrix,
            clipped_matrix,
        ),
    ):

        plot_matrix_with_loops(
            np.tril(M), loops=loops, title=title, cmap="seismic", vmax=.1
        )

    plt.show()


def load_and_loop(dataset):
    matrix = np.loadtxt(dataset)
    final_matrix = do_everything(matrix)
    return np.array(matrix_to_score(final_matrix))


def main():

    # ijs_res = pattern_finder2.pattern_finder2(matrix, with_plots=True)

    #    connected_matrix = keep_biggest_connected_component(detrended_matrix)
    # plot_matrix(gaussianized_matrix)

    # for matrix, thresh in iterative_threshold(haar_filtered_matrix, N_POINTS):
    #     break
    #     matrix_without_diagonal = remove_diagonal(matrix)
    #     z_scored_matrix = diag_zscores(matrix_without_diagonal)

    #     if np.sum(matrix_without_diagonal) == 0 and thresh > 0:
    #         logger.warning("Matrix is all zeros!")
    #         break

    #     title_original = f"Original matrix (thresh = {thresh})"
    #     title_threshold = (
    #         f"Thresholded matrix (thresh = {thresh})" " without diagonal"
    #     )
    #     title_zscore = (
    #         f"Z-scored matrix (thesh = {thresh})" " with diagonal removed"
    #     )

    #     assert np.isnan(matrix).sum() == 0
    #     assert np.isinf(matrix).sum() == 0
    #     assert np.isnan(matrix_without_diagonal).sum() == 0
    #     assert np.isinf(matrix_without_diagonal).sum() == 0
    #     assert np.isnan(z_scored_matrix).sum() == 0
    #     assert np.isinf(z_scored_matrix).sum() == 0

    #     plot_matrix_with_loops(
    #         matrix, loops=loops, title=title_original, vmax=1
    #     )
    #     plot_matrix_with_loops(
    #         matrix_without_diagonal, loops, title=title_threshold, vmax=1
    #     )
    #     print(matrix_without_diagonal)
    #     plot_matrix_with_loops(
    #         z_scored_matrix, loops=loops, title=title_zscore, vmax=3
    #     )
    pass


if __name__ == "__main__":
    main()
