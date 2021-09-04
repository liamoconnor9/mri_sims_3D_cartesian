"""plot_spectrum.py [--asymptotic] <filename>

Usage:
    plot_spectrum.py [--asymptotic] <filename>

"""

import numpy as np
import sys
from pathlib import Path
import h5py
import matplotlib.pyplot as plt
from calc_asymptotic_approx import calc_asymptotic_growth

from docopt import docopt

plt.style.use('prl')

# parse arguments
args = docopt(__doc__)
filename = args['<filename>']
asymptotic = args['--asymptotic']

outbase = Path("plots")
filename = Path(filename)
data = h5py.File(filename, 'r')
spectrum = data['eigvals'][:]
R = data['eigvals'].attrs['R']
B = data['eigvals'].attrs['B']
q = data['eigvals'].attrs['q']
d = data['eigvals'].attrs['d']
ky = data['eigvals'].attrs['ky']
kz = data['eigvals'].attrs['kz']
# check asymptotic calculation
if asymptotic:
    omega = calc_asymptotic_growth(R, q, B, d, ky, kz)
    print("asymptotic omega = {}".format(omega))

thresh = 2


spec_thresh = spectrum[np.abs(spectrum) < thresh]
plt.scatter(spec_thresh[spec_thresh.real <= 0].real, spec_thresh[spec_thresh.real <= 0].imag)
plt.scatter(spec_thresh[spec_thresh.real > 0].real, spec_thresh[spec_thresh.real > 0].imag)
plt.gca().set_prop_cycle(None)
if asymptotic:
    plt.plot(-omega, 0., marker='^')
    plt.plot(omega, 0., marker='^')
plt.xlim(-0.25,0.25)
plt.ylim(-0.8,0.8)
plt.xlabel("real")
plt.ylabel("imag")

plot_file_name = Path(filename.stem + '_spectrum.png')
plt.tight_layout()
plt.savefig(outbase/plot_file_name, dpi=300)
