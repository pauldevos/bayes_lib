import numpy as np

def save_chain(chain, fname):
    np.savetxt(fname, chain, delimiter = ",")
     
