import numpy as np
import dxchange
from solver_lam import SolverLam


n = 256 # sample width
nz = 64 # sample height
deth = 64 # detector height
ntheta = 32 # number of angles 
lamino_angle = 20 # tilt of the rotary stage

ctheta = ntheta//4 # chunk size for angles (to fit GPU memory)

# define angles
theta = np.linspace(0, 360, ntheta, endpoint=True).astype('float32')

# read [nz,n,n] part of an object [256,256,256]
u = -dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2]+1j*(dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2])

with SolverLam(n, nz, deth, ntheta, ctheta, theta, lamino_angle) as slv:
    # generate data, forward Laminography transform (data = Lu)
    data = slv.fwd_lam(u)            
    dxchange.write_tiff(data.real, 'data/data_re', overwrite=True)    
    dxchange.write_tiff(data.imag, 'data/data_im', overwrite=True)    
    # CG solver
    uinit = u*0
    niter = 512
    ur = slv.cg_lam(data,uinit,niter,dbg=True)            
    dxchange.write_tiff(ur.real, 'data/rec_re', overwrite=True)
    dxchange.write_tiff(ur.imag, 'data/rec_im', overwrite=True)

