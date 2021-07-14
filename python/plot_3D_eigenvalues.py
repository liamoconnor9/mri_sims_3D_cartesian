"""
constructs 3D eigenvector structures; plots them in z = 0 plane.

python3 path/to/datafile.h5
"""
import h5py
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

# Parses filename passed to script
parser = argparse.ArgumentParser(description='Passes filename')
parser.add_argument('filename', metavar='Rc', type=str, help='.h5 file to plot eigenvectors for maximum eigenvalue')
args = parser.parse_args()
filename = Path(vars(args)['filename'])
outbase = Path("plots")

eigenvector_vars = ['p','vx','vy','vz','ωy','ωz','bx','by','bz','jxx']

# Import .h5 file passed to script 
datafile = h5py.File(filename,'r')
eigvec = datafile['eigvec'][:,:,:,:]
x = datafile['x'][:]
gamma_global = datafile['gamma'][:,:]

# Find grid point with largest growth rate to plot corresponding eigenvectors
gamma_r = gamma_global.real
gamma_r[np.where(gamma_r<0)] = 0.0
max_point = np.amax(gamma_r)
z_max = np.where(gamma_r == max_point)[1][0]
y_max = np.where(gamma_r == max_point)[0][0]
# zv = kz_global[z_max] #max x value
# yv = ky_global[y_max] #max y value
# maxvalues[0,i] = zv
# maxvalues[1,i] = yv
# print(z_max)
# print(y_max)

ky = datafile['gamma'].attrs['max ky']
kz = datafile['gamma'].attrs['max kz']

Ly = 2*np.pi/ky
Lz = 2*np.pi/kz

n_yz = 128
y = np.linspace(0, Ly, n_yz, endpoint=False)
z = np.linspace(0, Lz, n_yz, endpoint=False)

zz,yy,xx = np.meshgrid(z,y,x,indexing='ij')
fig = plt.figure(figsize=(6,6))

yz_dep = np.exp(1j*(ky*yy +kz*zz))

p = (eigvec[y_max, z_max,0,:]*yz_dep).real
u = (eigvec[y_max, z_max,1,:]*yz_dep).real
v = (eigvec[y_max, z_max,2,:]*yz_dep).real
w = (eigvec[y_max, z_max,3,:]*yz_dep).real
Bx =(eigvec[y_max, z_max,6,:]*yz_dep).real
By =(eigvec[y_max, z_max,7,:]*yz_dep).real
Bz =(eigvec[y_max, z_max,8,:]*yz_dep).real

save_vtk = True
save_h5 = True
if save_vtk:
    vtkfile = filename.stem + '.vtk'
    from pyevtk.hl import gridToVTK 
    pointData = {'p':p.copy(), 'u':u.copy(), 'v':v.copy(), 'w':w.copy(),
                 'Bx':Bx.copy(), 'By':By.copy(), 'Bz':Bz.copy()}
    print("p flags", pointData['p'].flags)

    gridToVTK(vtkfile, xx, yy, zz, pointData=pointData)
if save_h5:
    h5file = filename.stem + '.h5'
    datafile = h5py.File(h5file, 'w')
    datafile.create_dataset('u', data=u)
    datafile.create_dataset('v', data=v)
    datafile.create_dataset('w', data=w)
    datafile.create_dataset('Bx', data=Bx)
    datafile.create_dataset('By', data=By)
    datafile.create_dataset('Bz', data=Bz)
    datafile.create_dataset('x_grid', data=xx)
    datafile.close()


grid = ImageGrid(fig, 111, nrows_ncols=(4,1),axes_pad=0.5, cbar_mode='each', cbar_location='top',cbar_size="10%", cbar_pad=0)
im = grid[0].imshow(u[0].T, extent=[0,Ly, -np.pi/2, np.pi/2])
cax = grid.cbar_axes[0]
cax.colorbar(im)
cax.axis[cax.orientation].label.set_text('u')
im = grid[1].imshow(v[0].T, extent=[0,Ly, -np.pi/2, np.pi/2])
cax = grid.cbar_axes[1]
cax.colorbar(im)
cax.axis[cax.orientation].label.set_text('v')
im = grid[2].imshow(w[0].T, extent=[0,Ly, -np.pi/2, np.pi/2])
cax = grid.cbar_axes[2]
cax.colorbar(im)
cax.axis[cax.orientation].label.set_text('w')
grid[3].imshow(p[0].T, extent=[0,Ly, -np.pi/2, np.pi/2])
cax = grid.cbar_axes[3]
cax.colorbar(im)
cax.axis[cax.orientation].label.set_text('p')


plot_file_name = Path(filename.stem + '_3D_v_eigenvector.png')
fig.savefig(outbase/plot_file_name, dpi=300)

fig.clear()
grid = ImageGrid(fig, 111, nrows_ncols=(3,1),axes_pad=0.5, cbar_mode='each', cbar_location='top',cbar_size="10%", cbar_pad=0)
im = grid[0].imshow(Bx[0].T, extent=[0,Ly, -np.pi/2, np.pi/2])
cax = grid.cbar_axes[0]
cax.colorbar(im)
cax.axis[cax.orientation].label.set_text('Bx')
im = grid[1].imshow(By[0].T, extent=[0,Ly, -np.pi/2, np.pi/2])
cax = grid.cbar_axes[1]
cax.colorbar(im)
cax.axis[cax.orientation].label.set_text('By')
im = grid[2].imshow(Bz[0].T, extent=[0,Ly, -np.pi/2, np.pi/2])
cax = grid.cbar_axes[2]
cax.colorbar(im)
cax.axis[cax.orientation].label.set_text('Bz')

plot_file_name = Path(filename.stem + '_3D_B_eigenvector.png')
fig.savefig(outbase/plot_file_name, dpi=300)
