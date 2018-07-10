import numpy as np
import pickle
from .model import Model

def save_chain(chain, fname):
    np.savetxt(fname, chain, delimiter = ",")

def save_model(fname, m = None):
    if m is None:
        m = Model.get_context()
    pickle.dump(m, open(fname, 'wb'))

def load_model(fname):
    return pickle.load(open(fname, 'rb'))

     
