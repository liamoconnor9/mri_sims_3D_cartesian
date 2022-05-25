import pickle
import matplotlib.pyplot as plt
import os
path = os.path.dirname(os.path.abspath(__file__))
import numpy as np

with open(path + '/projections.pick', 'rb') as f:
    projections = pickle.load(f)

cutoff=1
x1 = np.array(projections[0])
x2 = np.array(projections[1])
print(len(x1))
plt.scatter(x1, x2)

# sign1 = np.sign(x1)
# sign2 = np.sign(x2)

# x1abslog = np.log(np.abs(x1))
# x2abslog = np.log(np.abs(x1))

# x1logsignd = x1abslog * sign1
# x2logsignd = x2abslog * sign2
# plt.scatter(x1logsignd, x2logsignd)

plt.show()