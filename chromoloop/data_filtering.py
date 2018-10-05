
from pathlib import Path
from pylab import loadtxt, savetxt

from chromoloop.hicstuff import normalize_dense

def return_file(path_input, path_output):

    for file in path_input.iterdir():
        if 'MAT_RAW_realisation' not in file.stem:
            continue
        mat = loadtxt(file)
        norm = normalize_dense(mat)
        savetxt(path_output/file.name, norm)
        print(file)