import numpy as np
import dxchange
from solver_lam import SolverLam


n = 128 # sample width
nz = 64 # sample height
deth = 128 # detector height
ntheta = 64 # number of angles 
ctheta = ntheta//4 # chunk size for angles 
lamino_angle = 6

# define angles
theta = np.linspace(0, 360, ntheta, endpoint=True).astype('float32')

# read [nz,n,n] part of an object [256,256,256]
u = -dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2,128-n//2:128+n//2,128-n//2:128+n//2]+1j*(dxchange.read_tiff('data/delta-chip-256.tiff')[128-nz//2:128+nz//2,128-n//2:128+n//2,128-n//2:128+n//2])


with SolverLam(n, nz, deth, ntheta, ctheta, theta, lamino_angle) as slv:
    # generate data, forward Laminography transform (data = Lu)
    data = slv.fwd_lam(u)                
    dxchange.write_tiff(data.real, 'data/data_re', overwrite=True)
    dxchange.write_tiff(data.imag, 'data/data_im', overwrite=True)
    
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
    