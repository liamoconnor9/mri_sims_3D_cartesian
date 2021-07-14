"""
This script plots both the largest growth rates of the MRI data
passed to the script. 

Usage:
    make_eigenvector_plots.py [--vmin=<vmin> --vmax=<vmax> --q=<q> --no-contours --label=<label>] <filename>

Options:
    --vmin=<vmin>    min for colorbar [default: 0]
    --vmax=<vmax>    max for colorbar [default: 0.701]
    --q=<q>          q = S/f [default: 0.75]
    --no-contours    turn off contours
    --label=<label>  run label
"""

import h5py
from docopt import docopt
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('prl')

# Parses filename passed to script
args = docopt(__doc__)

filename = Path(args['<filename>'])
vmin = float(args['--vmin'])
vmax = float(args['--vmax'])
q = float(args['--q'])
contours = not args['--no-contours']
outbase = Path("plots")
label = args['--label']

# Plot growth rates
datafile = h5py.File(filename,'r')
gamma_global = datafile['gamma'][:]
gamma_r = gamma_global.real/q # scale by S...gamma is already multiplied by f in script

#gamma_r[np.where(gamma_r<0)] = 0.0
#max_gamma = datafile['gamma'].attrs['max growth rate']
ky_global    = datafile['ky']
kz_global    = datafile['kz']
print(ky_global.shape)
print(gamma_r.shape)
try:
    max_ky = datafile['gamma'].attrs['max ky']
    max_kz = datafile['gamma'].attrs['max kz']
except KeyError:
    ky_index, kz_index = np.where(gamma_r == gamma_r.max())
    max_ky = ky_global[ky_index[0]]
    max_kz = kz_global[kz_index[0]]

print("max growth rate on grid = {}".format(gamma_r.max()))

contour_levels = np.linspace(0,gamma_r.max(),10)
PCM = plt.imshow(gamma_r, extent=[kz_global[:].min(),kz_global[:].max(),ky_global[:].min(),ky_global[:].max()], vmin=vmin,vmax=vmax, origin='lower')
if contours:
    plt.contour(kz_global,ky_global,gamma_r,levels=contour_levels,colors='k')
plt.contour(kz_global,ky_global,gamma_r,levels=[0.],colors='w',alpha=0.5)
plt.plot(max_kz, max_ky, 'ro')
plt.colorbar(PCM, label=r'$\gamma/|S|$')
plt.xlabel(r'$k_z$')
plt.ylabel(r'$k_y$')

if label:
    plt.text(0.05,0.9,r'${}$'.format(label), color='w',fontsize=20)
plt.tight_layout()

#plt.title(r'3D Keplerian MRI growth rates/f  $\left( S/S_{\mathrm{crit.}} = %.2f\right)$' %(R[i]))
plot_file_name = Path(filename.stem + '_growthrates.pdf')
plt.savefig(outbase/plot_file_name)#, dpi=300)
