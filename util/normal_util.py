import pickle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import networkx as nx
from scipy import sparse
import pylab as plt
import dateutil.parser as dparser
import re


def save_object(obj, filename):
    with open(filename, 'wb') as output:  # Overwrites any existing file.
        pickle.dump(obj, output, 2)

def load_object(filename):
    output = 0
    with open(filename, 'rb') as fp:
        output = pickle.load(fp)
    return output




