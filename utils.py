# utils functions

## just for plotting settings
import matplotlib.pyplot as plt
plt.style.use('dark_background')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['image.cmap'] = 'inferno'

import numpy as np
np.set_printoptions(precision=2)
from numpy.random import uniform

from scipy.io import loadmat, savemat

from scipy.interpolate import RegularGridInterpolator
# INTERP_METHOD = 'linear' # fast, but less accurate
INTERP_METHOD = 'quintic' # slowest, but most accurate
if INTERP_METHOD == 'linear': print('Warning: using linear interpolation, which is fast but less accurate')

# NGR = 28 # number of grid points in the x direction
# NGZ = 65 # number of grid points in the y direction
# NGR = NGZ = 64 # number of grid points 
NGR = NGZ = 24 # number of grid points 

USE_CURRENTS = True # usually True
USE_PROFILES = True # false -> more realistic
USE_MAGNETIC = True # usually True
NIN = int(USE_CURRENTS)*19 + int(USE_PROFILES)*38 + int(USE_MAGNETIC)*38 # input size

# load the vessel perimeter
VESS = loadmat('tcv_params/vess.mat')['vess']
VESS = np.vstack([VESS, VESS[0]])

# def sample_random_subgrid(rrG, zzG, nr=64, nz=64): # old, working
#     rm, rM, zm, zM = rrG.min(), rrG.max(), zzG.min(), zzG.max()
#     delta_r_min = .33*(rM-rm)
#     delta_r_max = .75*(rM-rm)
#     delta_z_min = .2*(zM-zm)
#     delta_z_max = .75*(zM-zm)
#     delta_r = uniform(delta_r_min, delta_r_max, 1)
#     r0 = uniform(rm, rm+delta_r_max-delta_r, 1)
#     delta_z = uniform(delta_z_min, delta_z_max, 1)
#     z0 = uniform(zm,zm+delta_z_max-delta_z, 1)
#     rr = np.linspace(r0, r0+delta_r, nr)
#     zz = np.linspace(z0, z0+delta_z, nz)
#     rrg, zzg = np.meshgrid(rr, zz)
#     return rrg, zzg

# def sample_random_subgrid(rrG, zzG, nr=64, nz=64): #general, working
#     rm, rM, zm, zM = rrG[0,0], rrG[-1,-1], zzG[0,0], zzG[-1,-1]
#     Δr, Δz = rM-rm, zM-zm 
#     nΔr, nΔz = Δr*uniform(0.4, 1.0), Δz*uniform(0.4, 1.0)
#     r0, z0 = uniform(rm, rM-nΔr), uniform(zm, zM-nΔz)
#     rr, zz = np.linspace(r0, r0+nΔr, nr), np.linspace(z0, z0+nΔz, nz)
#     rrg, zzg = np.meshgrid(rr, zz)
#     return rrg, zzg

def sample_random_subgrid(rrG, zzG, nr=64, nz=64): # tcv specific
    rm, rM, zm, zM = rrG[0,0], rrG[-1,-1], zzG[0,0], zzG[-1,-1]
    Δr, Δz = rM-rm, zM-zm 
    assert Δr < Δz, f'TCV grid is like this'
    nΔr = nΔz = Δr*uniform(0.5, 1.0) # force square grid
    r0, z0 = uniform(rm, rM-nΔr), uniform(zm, zM-nΔz)
    rr, zz = np.linspace(r0, r0+nΔr, nr), np.linspace(z0, z0+nΔz, nz)
    rrg, zzg = np.meshgrid(rr, zz)
    return rrg, zzg

def spans2grids(rs, zs):
    assert rs.shape[1:] == (NGR,), f"rs.shape = {rs.shape}, NGR = {NGR}"
    assert zs.shape[1:] == (NGZ,), f"zs.shape = {zs.shape}, NGZ = {NGZ}"
    assert rs.shape[0] == zs.shape[0], f"rs.shape = {rs.shape}, zs.shape = {zs.shape}"
    rrgs, zzgs = np.zeros((len(rs), NGR, NGZ)), np.zeros((len(zs), NGR, NGZ))
    for i in range(len(rs)):
        rrgs[i,:], zzgs[i,:] = np.meshgrid(rs[i], zs[i])
    return rrgs, zzgs

def grid2box(rrg, zzg):
    rm, rM, zm, zM = rrg.min(), rrg.max(), zzg.min(), zzg.max()
    return np.array([[rm,zm],[rM,zm],[rM,zM],[rm,zM],[rm,zm]])

def interp_fun(ψ, rrG, zzG, rrg, zzg, method=INTERP_METHOD):
    interp_func = RegularGridInterpolator((rrG[0,:], zzG[:,0]), ψ.T, method=method)
    pts = np.column_stack((rrg.reshape(-1), zzg.reshape(-1)))
    f_int = interp_func(pts).reshape(rrg.shape)
    return f_int

def resample_on_new_subgrid(fs:list, rrG, zzG, nr=64, nz=64):
    rrg, zzg = sample_random_subgrid(rrG, zzG, nr, nz)
    fs_int = [interp_fun(ψ, rrG, zzG, rrg, zzg) for ψ in fs]
    return fs_int, rrg, zzg

# kernels
def calc_laplace_df_dr_ker(hr, hz):
    α = -2*(hr**2 + hz**2)
    assert np.abs(α) > 1e-10, f"α = {α} <= 0, hr = {hr}, hz = {hz}"
    laplace_ker = np.array(([0, hr**2/α, 0], [hz**2/α, 1, hz**2/α], [0, hr**2/α, 0]))
    dr_ker = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr*α)*(hr**2*hz**2)
    return laplace_ker, dr_ker

#calculate the Grad-Shafranov operator pytorch
import torch
import torch.nn.functional as F

def laplace_ker(Δr, Δz, α, dev=torch.device("cpu")): # [[0, Δr**2/α, 0], [Δz**2/α, 1, Δz**2/α], [0, Δr**2/α, 0]]
    kr, kz = Δr**2/α, Δz**2/α
    ker = torch.zeros(len(Δr),1, 3, 3, dtype=torch.float32, device=dev)
    ker[:,0,0,1], ker[:,0,1,0], ker[:,0,1,2], ker[:,0,2,1], ker[:,0,1,1] = kr, kz, kz, kr, 1
    return ker
   
def dr_ker(Δr, Δz, α, dev=torch.device("cpu")): # [[0,0,0],[-1,0,+1],[0,0,0]] * (Δr**2 * Δz**2) / (2*Δr*α)
    ker = torch.zeros(len(Δr),1, 3, 3, dtype=torch.float32, device=dev)
    k = (Δr**2 * Δz**2) / (2*Δr*α)
    ker[:,0,1,0], ker[:,0,1,2] = -k, k
    return ker

def gauss_ker(dev=torch.device("cpu")):
    # ker = torch.tensor([[1,2,1],[2,4,2],[1,2,1]], dtype=torch.float32, device=dev).view(1,1,3,3) / 16
    ker = torch.tensor([[1,4,6,4,1],[4,16,24,16,4],[6,24,36,24,6],[4,16,24,16,4],[1,4,6,4,1]], dtype=torch.float32, device=dev).view(1,1,5,5) / 256
    return ker

# simple reshape block for convenience
class View(torch.nn.Module):
    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape
    def forward(self, x): 
        try:
            xshape0 = x.shape[0]
            nx = x.contiguous().view(self.shape)
            if x.ndim > 1: assert nx.shape[0] == xshape0, f"nx.shape[0] = {nx.shape[0]}, xshape0 = {xshape0}, self.shape = {self.shape}"
        except Exception as e:
            print(f"Error in View: {e}")
            print(f"x.shape = {x.shape}, self.shape = {self.shape}")
            raise e
        return nx

# einops TODO: look

def Ϛ(x, ker): # convolve x with ker
    assert x.ndim == 4 and x.shape[1] == 1, f"x.ndim = {x.ndim}, x.shape = {x.shape}"
    assert ker.ndim == 4 and ker.shape[1] == 1, f"ker.ndim = {ker.ndim}, ker.shape =  {ker.shape}"
    s2, s3 = x.shape[2], x.shape[3]
    if ker.shape[0] > 1: x = x.view(1,-1,s2,s3) # if the kernel is not the same for all samples
    p = ker.shape[2]//2 # padding size
    return F.pad(F.conv2d(x, ker, groups=ker.shape[0]), (p,p,p,p), mode='replicate').view(-1,1,s2,s3)

def calc_gso(ψ, r, z):
    sΨ2, sΨ3 = ψ.shape[0], ψ.shape[1]
    assert r.ndim == 1, f'r.ndim = {r.ndim}, r.shape = {r.shape}'
    assert z.ndim == 1, f'z.ndim = {z.ndim}, z.shape = {z.shape}'
    sr, sz = r.shape[0], z.shape[0]
    Ψ, r, z = torch.tensor(ψ).view(1,1,sΨ2,sΨ3), torch.tensor(r).view(1,sr), torch.tensor(z).view(1,sz)
    return calc_gso_batch(Ψ, r, z).numpy()[0,0]

def calc_gso_batch(Ψ, r, z, dev=torch.device('cpu')):
    assert r.ndim == 2, f'r.ndim = {r.ndim}, r.shape = {r.shape}'
    assert z.ndim == 2, f'z.ndim = {z.ndim}, z.shape = {z.shape}'
    assert r.shape[1] == NGR, f'r.shape[1] = {r.shape[1]}, NGR = {NGR}'
    assert z.shape[1] == NGZ, f'z.shape[1] = {z.shape[1]}, NGZ = {NGZ}'
    Δr, Δz = r[:,2]-r[:,1], z[:,2]-z[:,1]
    rr = r.repeat(1,NGZ).view(-1,1,NGR,NGZ).permute(0,1,3,2)  # repeat for batch
    α = (-2*(Δr**2 + Δz**2))
    β = ((Δr**2 * Δz**2) / α)
    ΔΨ = (1/β.view(-1,1,1,1)) * (Ϛ(Ψ, laplace_ker(Δr, Δz, α, dev)) - Ϛ(Ψ, dr_ker(Δr, Δz, α, dev))/rr) # grad-shafranov operator
    # ΔΨ = Ϛ(ΔΨ, gauss_ker(dev)) # apply gauss kernel
    return ΔΨ

## PLOTTING
