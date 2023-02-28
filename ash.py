import numpy as np
import dxchange
import scipy.io
from scipy import ndimage
from solver_lam import SolverLam



theta = scipy.io.loadmat('ash/8id/angles.mat')['angles'].astype('float32').flatten()
cube = scipy.io.loadmat('ash/8id/cube.mat')['cube'][:,64:-64].astype('float32').swapaxes(0,2)[:,::-1,:]
data = scipy.io.loadmat('ash/8id/proj_data.mat')['proj_data'].astype('float32').swapaxes(0,1).swapaxes(1,2)
dxchange.write_tiff(cube,'data/cube',overwrite=True)
dxchange.write_tiff(data,'data/data',overwrite=True)


# I dont understand why the initial sample is tilted (I tilt it back)
cube_rot = ndimage.rotate(cube, axes=(1,0),angle=-6,reshape=False)
dxchange.write_tiff(cube_rot,'data/cube_rot',overwrite=True)

n = cube.shape[-1]
nz = cube.shape[0] # sample height
deth = data.shape[1] # detector height
ntheta = data.shape[0] # number of angles 
lamino_angle = 6 # tilt of the rotary stage

ctheta = ntheta # chunk size for angles (to fit GPU memory)

# define angles
theta=theta/np.pi*360

# read [nz,n,n] part of an object [256,256,256]
u = cube_rot+1j*cube_rot

with SolverLam(n, nz, deth, ntheta, ctheta, theta, lamino_angle) as slv:
    # generate data, forward Laminography transform (data = Lu)
    data = slv.fwd_lam(u)            
    dxchange.write_tiff(data.real, 'data/data_re', overwrite=True)    
    dxchange.write_tiff(data.imag, 'data/data_im', overwrite=True)    
    
    # ADMM solver
    u = np.zeros(u.shape,dtype='complex64')
    psi = np.zeros([3,*u.shape],dtype='complex64')
    h = np.zeros([3,*u.shape],dtype='complex64')    
    lamd = np.zeros([3,*u.shape],dtype='complex64')    
    niter = 32
    liter = 4
    alpha = 1e-8
    u = slv.admm(data, h, psi, lamd, u, alpha, liter, niter)
    dxchange.write_tiff(u.real, 'data/rec_re', overwrite=True)
    dxchange.write_tiff(u.imag, 'data/rec_im', overwrite=True)


ur_rot = ndimage.rotate(ur, axes=(1,0),angle=6)

dxchange.write_tiff(ur_rot.real,'data/rec_rot_re',overwrite=True)
dxchange.write_tiff(ur_rot.imag,'data/rec_rot_imag',overwrite=True)