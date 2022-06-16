import numpy as np
import os
import pickle
path = os.path.dirname(os.path.abspath(__file__))
import matplotlib.pyplot as plt
dir = 'test'

fns = ['metrics_u.pick', 'metrics_psi.pick', 'metrics_omega.pick']
errors = ['u_error', 'psi_error', 'omega_error']

# for error_name in errors:
error_name = errors[1]
for fn in fns:
    with open(path + '/' + dir + '/' + fn, 'rb') as f:
        error = pickle.load(f)[error_name]
        plt.scatter(range(len(error)), error, label=fn)

plt.legend()
plt.show()

