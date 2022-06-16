import numpy as np
import os
import pickle
path = os.path.dirname(os.path.abspath(__file__))

dir = 'test'
with open(path + '/' + dir + '/metrics.pick', 'rb') as f:
    mets = pickle.load(f)

print(mets.keys())
