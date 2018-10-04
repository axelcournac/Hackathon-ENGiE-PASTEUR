from pathlib import Path

from pylab import loadtxt, savetxtfrom chromoloop.hicstuff import normalize_densedef normalize_all(path_input, path_output):    for file in path_input.iterdir():

       if 'MAT_RAW_realisation' not in file.stem:

           continue

       mat = loadtxt(file)

       norm = normalize_dense(mat)

       savetxt(path_output/file.name, norm)

       print(file)    return Noneif __name__=='__main__':    path_input = Path(__file__).parent / 'data'

   path_output = path_input / 'normalized'    normalize_all(path_input, path_output)