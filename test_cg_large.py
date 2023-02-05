import numpy as np
import dxchange
from scipy import ndimage
from solver_lam import SolverLam

zoom = 4 # zoom initial object to increase data sizes

n = 256 # sample width
nz = 32 # sample height
deth = 128 # detector height
ntheta = 32 # number of angles 
lamino_angle = 20 # tilt of the rotary stage

ctheta = 4 # chunk size for angles (to fit GPU memory)

# define angles
theta = np.linspace(0, 360, ntheta, endpoint=True).astype('float32')

# read [nz,n,n] part of an object [256,256,256]
u = -dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2]

# zooming
n*=zoom
nz*=zoom
deth*=zoom
u = ndimage.zoom(u,zoom,order=1)

print(f'object size {nz}x{n}x{n}, data size {ntheta}x{deth}x{n}')

with SolverLam(n, nz, deth, ntheta, ctheta, theta, lamino_angle) as slv:
    # generate data, forward Laminography transform (data = Lu)
    data = slv.fwd_lam(u)            
    dxchange.write_tiff(data, 'data/data', overwrite=True)    
    # CG solver
    uinit = u*0
    niter = 32
    ur = slv.cg_lam(data,uinit,niter,dbg=True)            
    dxchange.write_tiff(ur, 'data/rec', overwrite=True)

