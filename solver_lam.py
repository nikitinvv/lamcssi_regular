"""Module for laminography."""

import cupy as cp
import numpy as np
from kernels import fwd,adj
from cupyx.scipy.fft import rfft, irfft

class SolverLam():
    """Base class for laminography solvers using the direct line integration with linear interpolation on GPU.
    This class is a context manager which provides the basic operators required
    to implement a laminography solver. It also manages memory automatically,
    and provides correct cleanup for interruptions or terminations.
    Attribtues
    ----------
    nx : int
        Object size in x
    ny : int
        Object size in y
    nz : int
        Object size in z
    deth : int
        Detector height
    detw : int
        Detector width
    ntheta : int
        Number of projections
    ctheta : int
        Chunk size in angles for simultaneous processing with a GPU
    theta : float32
        Angles for laminography 
    lamino : float32
        Tilt angle for the rotary stage
    """
    def __init__(self, nx, ny, nz, detw, deth, ntheta, ctheta, theta, lamino_angle):
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.detw = detw
        self.deth = deth
        self.ntheta = ntheta
        self.ctheta = ctheta
        self.theta = theta/180*np.pi
        self.lamino_angle = (90-lamino_angle)/180*np.pi # NOTE: switching the angle to the 'laminography' formulation: [0,pi], rotation from the axis orhotgonal to the beam
        
    def __enter__(self):
        """Return self at start of a with-block."""
        return self

    def __exit__(self, type, value, traceback):
        """Free GPU memory due at interruptions or with-block exit."""
        pass

    def fwd_lam(self, u):
        """Forward laminography operator data = Lu"""
        
        # result
<<<<<<< HEAD
        data = np.zeros([self.ntheta, self.deth, self.detw], dtype='float32')    

        # GPU memory
        data_gpu = cp.zeros([self.ctheta, self.deth, self.detw], dtype='float32')
=======
        data = np.zeros([self.ntheta, self.deth, self.n], dtype='complex64')    

        # GPU memory
        data_gpu = cp.zeros([self.ctheta, self.deth, self.n], dtype='complex64')
>>>>>>> e0130311d1a3964eb2df24143783678428c6d44a
        theta_gpu = cp.zeros([self.ctheta], dtype='float32')                
        u_gpu = cp.asarray(u)
        # processing by chunks in angles
        for it in range(int(np.ceil(self.ntheta/self.ctheta))):
            st = it*self.ctheta
            end = min(self.ntheta,(it+1)*self.ctheta)
            
            # copy a data chunk to gpu                            
            data_gpu[:end-st] = cp.asarray(data[st:end])
            data_gpu[end-st:] = 0
            theta_gpu[:end-st] = cp.asarray(self.theta[st:end])        
            
            # generate data
            fwd(data_gpu,u_gpu,theta_gpu,self.lamino_angle)
            
            # copy result to CPU
            data[st:end] = data_gpu[:end-st].get()
        return data/ np.sqrt(self.ntheta*self.n)

    def adj_lam(self,data):
        """adjoint laminography operator u = L*data"""
        
        # GPU memory
<<<<<<< HEAD
        data_gpu = cp.zeros([self.ctheta, self.deth, self.detw], dtype='float32')
        u_gpu = cp.zeros([self.nz, self.ny, self.nx], dtype='float32')
=======
        data_gpu = cp.zeros([self.ctheta, self.deth, self.n], dtype='complex64')
        u_gpu = cp.zeros([self.nz, self.n, self.n], dtype='complex64')
>>>>>>> e0130311d1a3964eb2df24143783678428c6d44a
        theta_gpu = cp.zeros([self.ctheta], dtype='float32')                
        
        for it in range(int(np.ceil(self.ntheta/self.ctheta))):
            st = it*self.ctheta
            end = min(self.ntheta,(it+1)*self.ctheta)                        
            
            # copy a data chunk to gpu                            
            data_gpu[:end-st] = cp.asarray(data[st:end])
            data_gpu[end-st:] = 0
            theta_gpu[:end-st] = cp.asarray(self.theta[st:end])   
            # data_gpu = self.fbp_filter_center(data_gpu)
            # bakprojection            
            adj(u_gpu,data_gpu,theta_gpu,self.lamino_angle)   
        u =  u_gpu.get()
        
        return u/np.sqrt(self.ntheta*self.n)    
    
    def line_search(self, minf, gamma, Lu, Ld):
        """Line search for the step sizes gamma"""
        while(minf(Lu)-minf(Lu+gamma*Ld) < 0):
            gamma *= 0.5
        return gamma

    def line_search_ext(self, minf, gamma, Lu, Ld, gu, gd):
        """Line search for the step sizes gamma"""
        while(minf(Lu, gu)-minf(Lu+gamma*Ld, gu+gamma*gd) < 0):
            gamma *= 0.5
            if(gamma < 1e-8):
                gamma = 0
                break
        return gamma
    
    def cg_lam(self, data, u, titer, dbg=False):
        """CG solver for ||Lu-data||_2"""
        
        # minimization functional
        def minf(Lu):
            f = cp.linalg.norm(Lu-data)**2
            return f
        for i in range(titer):
            Lu = self.fwd_lam(u)
<<<<<<< HEAD
            grad = self.adj_lam(Lu-data) * 1 / \
                self.ntheta/self.nx/self.ny/self.nz
=======
            grad = self.adj_lam(Lu-data)
>>>>>>> e0130311d1a3964eb2df24143783678428c6d44a

            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (cp.sum(np.conj(d)*(grad-grad0))+1e-32)*d
            # line search
            Ld = self.fwd_lam(d)
            gamma = 0.5*self.line_search(minf, 8, Lu, Ld)
            grad0 = grad.copy()
            # update step
            u = u + gamma*d
            # check convergence
            if (dbg == True):
                print("%4d, gamma %.3e, fidelity %.7e" %
                      (i, gamma, minf(Lu)))
        return u
    
    
    def fwd_reg(self, u):
        """Forward operator for regularization"""
        res = np.zeros([3, *u.shape], dtype='complex64')
        res[0, :, :, :-1] = u[:, :, 1:]-u[:, :, :-1]
        res[1, :, :-1, :] = u[:, 1:, :]-u[:, :-1, :]
        res[2, :-1, :, :] = u[1:, :, :]-u[:-1, :, :]
        res*=1/np.sqrt(3)
        return res

    
    def adj_reg(self, gr):
        """Adjoint operator for regularization"""
        res = np.zeros(gr.shape[1:], dtype='complex64')
        res[:, :, 1:] = gr[0, :, :, 1:]-gr[0, :, :, :-1]
        res[:, :, 0] = gr[0, :, :, 0]
        res[:, 1:, :] += gr[1, :, 1:, :]-gr[1, :, :-1, :]
        res[:, 0, :] += gr[1, :, 0, :]
        res[1:, :, :] += gr[2, 1:, :, :]-gr[2, :-1, :, :]
        res[0, :, :] += gr[2, 0, :, :]
        res *= -1/np.sqrt(3)  # normalization
        return res
    
    
    def solve_reg(self, u, lamd, rho, alpha):
        """ Regularizer problem"""
        z = self.fwd_reg(u)+lamd/rho
        # Soft-thresholding
        za = np.sqrt(np.real(np.sum(z*np.conj(z), 0)))
        z[:, za <= alpha/rho] = 0
        z[:, za > alpha/rho] -= alpha/rho * \
            z[:, za > alpha/rho]/(za[za > alpha/rho])
        return z
    
    def update_penalty(self, psi, h, h0, rho):
        """Update rhofor a faster convergence"""
        # rho
        r = cp.linalg.norm(psi - h)**2
        s = cp.linalg.norm(rho*(h-h0))**2
        if (r > 10*s):
            rho *= 2
        elif (s > 10*r):
            rho *= 0.5
        return rho
    
    def cg_lam_ext(self, data, g, init, rho, titer, dbg=True):
        """extended CG solver for ||Lu-data||_2+rho||gu-g||_2"""
        # minimization functional
        def minf(Lu, gu):
            return np.linalg.norm(Lu-data)**2+rho*np.linalg.norm(gu-g)**2
        u = init.copy()
        
        for i in range(titer):
            Lu = self.fwd_lam(u)
            gu = self.fwd_reg(u)
            grad = self.adj_lam(Lu-data) + \
                rho*self.adj_reg(gu-g)
            # Dai-Yuan direction            
            if i == 0:
                d = -grad
            else:
                d = -grad+np.linalg.norm(grad)**2 / \
                    (np.sum(np.conj(d)*(grad-grad0))+1e-32)*d                
            grad0 = grad.copy()
            Ld = self.fwd_lam(d)
            gd = self.fwd_reg(d)
            # line search
            gamma = 0.5*self.line_search_ext(minf, 8, Lu, Ld,gu,gd)
            # update step
            u = u + gamma*d
            # check convergence
            if (dbg == True):
                print("%4d, gamma %.3e, fidelity %.7e" %
                      (i, gamma, minf(Lu,gu)))
        return u
    
    
    def admm(self, data, h, psi, lamd, u, alpha, titer, niter):
        """ ADMM for laminography problem with TV regularization"""
        rho = 0.5
        for m in range(niter):
            # keep previous iteration for penalty updates
            h0 = h.copy()
            # laminography problem
            u = self.cg_lam_ext(data, psi-lamd/rho, u, rho, titer)            
            # regularizer problem
            psi = self.solve_reg(u, lamd, rho, alpha)
            # h updates
            h = self.fwd_reg(u)
            # lambda update
            lamd = lamd + rho * (h-psi)
            # update rho for a faster convergence
            rho = self.update_penalty(psi, h, h0, rho)
            # Lagrangians difference between two iterations
            if (np.mod(m, 1) == 0):
                # print(m)
                lagr = self.take_lagr(
<<<<<<< HEAD
                    psi, data, h, lamd, alpha, rho)
=======
                    u, psi, data, h, lamd, alpha,rho)
>>>>>>> e0130311d1a3964eb2df24143783678428c6d44a
                print("%d/%d) rho=%.2e, Lagrangian terms:   %.2e %.2e %.2e %.2e, Sum: %.2e" %
                      (m, niter, rho, *lagr))
        return u

    
    def take_lagr(self, u, psi, data, h, lamd, alpha, rho):
        """ Lagrangian terms for monitoring convergence"""
        lagr = np.zeros(5, dtype="float32")
        Lu = self.fwd_lam(u)
        lagr[0] += np.linalg.norm(Lu-data)**2
        lagr[1] = alpha*np.sum(np.sqrt(np.real(np.sum(psi*np.conj(psi), 0))))        
        lagr[2] = np.sum(np.real(np.conj(lamd)*(h-psi)))        
        lagr[3] = rho*np.linalg.norm(h-psi)**2
        lagr[4] = np.sum(lagr[:4])
        return lagr