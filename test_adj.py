import numpy as np
import dxchange
from solver_lam import SolverLam
import time

<<<<<<< HEAD
nx = 64 # sample width
ny = 4096 # sample width
nz = 256 # sample height
deth = 1024 # detector height
detw = 512 # detector height
ntheta = 128 # number of angles 
ctheta = ntheta # chunk size for angles 
lamino_angle = 2
=======

n = 128 # sample width
nz = 64 # sample height
deth = 128 # detector height
ntheta = 64 # number of angles 
ctheta = ntheta//4 # chunk size for angles 
lamino_angle = 6
>>>>>>> e0130311d1a3964eb2df24143783678428c6d44a

# define angles
theta = np.linspace(-20/180*np.pi,20/180*np.pi,ntheta,endpoint=True).astype('float32')
print(nz/np.sin(lamino_angle/180*np.pi))
# read [nz,n,n] part of an object [256,256,256]
<<<<<<< HEAD
u = np.random.random([nz,ny,nx]).astype('float32')#.-dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2,128-ny//2:128+ny//2,128-nx//2:128+nx//2]
=======
u = -dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2,128-n//2:128+n//2,128-n//2:128+n//2]+1j*(dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2,128-n//2:128+n//2,128-n//2:128+n//2])
>>>>>>> e0130311d1a3964eb2df24143783678428c6d44a


with SolverLam(nx, ny, nz, detw, deth, ntheta, ctheta, theta, lamino_angle) as slv:
    # generate data, forward Laminography transform (data = Lu)
<<<<<<< HEAD
    t = time.time()
    data = slv.fwd_lam(u)            
    print(time.time()-t)
    # exit()
    print(np.linalg.norm(data))
    dxchange.write_tiff(data, 'data/data', overwrite=True)
=======
    data = slv.fwd_lam(u)                
    dxchange.write_tiff(data.real, 'data/data_re', overwrite=True)
    dxchange.write_tiff(data.imag, 'data/data_im', overwrite=True)
>>>>>>> e0130311d1a3964eb2df24143783678428c6d44a
    
    # adjoint Laminography transform (ur = L*data)
    ur = slv.adj_lam(data)            
    dxchange.write_tiff(ur, 'data/rec_re', overwrite=True)
    dxchange.write_tiff(ur, 'data/rec_im', overwrite=True)

    print(np.sum(u*np.conj(ur)))
    print(np.sum(data*np.conj(data)))
    ddata = slv.fwd_lam(ur)                
    
    print(np.sum(data*np.conj(ddata))/np.sum(ddata*np.conj(ddata)))    
    print(np.sum(u*np.conj(ur))/np.sum(ur*np.conj(ur)))    
    print('')
    
    data = slv.fwd_reg(u)                
    ur = slv.adj_reg(data)   
    ddata = slv.fwd_reg(ur)                    
    print(np.sum(u*np.conj(ur)))
    print(np.sum(data*np.conj(data)))
    
    
    
    print(np.sum(data*np.conj(ddata))/np.sum(ddata*np.conj(ddata)))    
    