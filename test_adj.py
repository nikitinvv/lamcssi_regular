import numpy as np
import dxchange
from solver_lam import SolverLam
import time

nx = 64 # sample width
ny = 4096 # sample width
nz = 256 # sample height
deth = 1024 # detector height
detw = 512 # detector height
ntheta = 128 # number of angles 
ctheta = ntheta # chunk size for angles 
lamino_angle = 2

# define angles
theta = np.linspace(-20/180*np.pi,20/180*np.pi,ntheta,endpoint=True).astype('float32')
print(nz/np.sin(lamino_angle/180*np.pi))
# read [nz,n,n] part of an object [256,256,256]
u = np.random.random([nz,ny,nx]).astype('float32')#.-dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2,128-ny//2:128+ny//2,128-nx//2:128+nx//2]


with SolverLam(nx, ny, nz, detw, deth, ntheta, ctheta, theta, lamino_angle) as slv:
    # generate data, forward Laminography transform (data = Lu)
    t = time.time()
    data = slv.fwd_lam(u)            
    print(time.time()-t)
    # exit()
    print(np.linalg.norm(data))
    dxchange.write_tiff(data, 'data/data', overwrite=True)
    
    # adjoint Laminography transform (ur = L*data)
    ur = slv.adj_lam(data)            
    dxchange.write_tiff(ur, 'data/rec', overwrite=True)

print(np.sum(u*np.conj(ur)))
print(np.sum(data*np.conj(data)))