#!/usr/bin/env python3

"""Generate datasets from examples

Use a Poisson distribution from hicstuff to quickly generate many datasets
and write them.

"""


import numpy as np
import hicstuff as hcs
import itertools
import os
import pathlib


def base_filename(norm_state, chrom_number):
    return pathlib.Path(
        f"MAT_{norm_state}_chr{chrom_number}_control-G1-cdc20-TS_2kb.txt"
    )


bootstrap_path = pathlib.Path("bootstrapped")
example_path = pathlib.Path("data")

if not bootstrap_path.exists():
    os.mkdir(bootstrap_path)

ITERATIONS = 500

for i in range(ITERATIONS):
    for norm, chrom in itertools.product(("RAW", "SCN"), range(1, 16)):

        M = np.genfromtxt(example_path / base_filename(norm, chrom))
        N = hcs.noise(M)
        np.savetxt(
            bootstrap_path / base_filename(f"{norm}_bootstrapped_{i}", chrom),
            N,
        )
