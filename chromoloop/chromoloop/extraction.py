from pathlib import Path
from pylab import loadtxt
import numpy as np
import json



def extract_square(path_norm, path_loop, dimension=8):

    map_loop = {}
    for file in path_norm.iterdir():
        mat = loadtxt(file)
        idx = file.stem.split('_')[3]
        loop_name = 'Loops_realisation_' + idx + '.txt'
        file_loop = path_loop / loop_name
        all_loop = loadtxt(file_loop)
        map_loop[idx] = []
        for loop in all_loop:
            p1 = int(loop[0])
            p2 = int(loop[1])
            try:
                sub_mat = mat[np.ix_(range(p1 - dimension, p1 + dimension), range(p2-dimension, p2 + dimension ))]
                map_loop[idx].append(sub_mat.tolist())
            except:
                print('out of bounds')
                continue
        print(idx)

    return map_loop


def save_map_loop(map_loop, pathoutput):

    with open(pathoutput, 'w') as outfile:
        json.dump(map_loop, outfile)

def load_map_loop(pathoutput):

    with open(pathoutput) as f:
        map_list = json.load(f)

    map_loop = {}
    for idx, lmat in map_list.items():
        map_loop[idx] = [np.array(elt) for elt in lmat]

    return map_loop

if __name__=='__main__':

    path_data = Path(__file__).parent / 'data'
    path_loop = Path(__file__).parent / 'data'
    path_norm = path_data / 'normalized'

    map_loop = extract_square(path_norm, path_loop)
    save_map_loop(map_loop, path_data /'all_loops.json')

