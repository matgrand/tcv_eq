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

# N_GRID_R = 28 # number of grid points in the x direction
# N_GRID_Z = 65 # number of grid points in the y direction
# N_GRID_R = N_GRID_Z = 64 # number of grid points 
N_GRID_R = N_GRID_Z = 24 # number of grid points 

# load the vessel perimeter
VESS = loadmat('tcv_params/vess.mat')['vess']
VESS = np.vstack([VESS, VESS[0]])

# def sample_random_subgrid(rrG, zzG, nr=64, nz=64):
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

def sample_random_subgrid(rrG, zzG, nr=64, nz=64):
    rm, rM, zm, zM = rrG[0,0], rrG[-1,-1], zzG[0,0], zzG[-1,-1]
    Δr, Δz = rM-rm, zM-zm 
    nΔr, nΔz = Δr*uniform(0.4, 1.0), Δz*uniform(0.4, 1.0)
    r0, z0 = uniform(rm, rM-nΔr), uniform(zm, zM-nΔz)
    rr, zz = np.linspace(r0, r0+nΔr, nr), np.linspace(z0, z0+nΔz, nz)
    rrg, zzg = np.meshgrid(rr, zz)
    return rrg, zzg

def get_box_from_grid(rrg, zzg):
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

def calc_gso(ψ, rr, zz):
    sΨ2, sΨ3 = ψ.shape[0], ψ.shape[1]
    srr2, srr3 = rr.shape[0], rr.shape[1]
    szz2, szz3 = zz.shape[0], zz.shape[1]
    Ψ, rr, zz = torch.tensor(ψ).view(1,1,sΨ2,sΨ3), torch.tensor(rr).view(1,1,srr2,srr3), torch.tensor(zz).view(1,1,szz2,szz3)
    return calc_gso_batch(Ψ, rr, zz).numpy()[0,0]

def Ϛ(x, ker): # convolve x with ker
    assert x.ndim == 4 and x.shape[1] == 1, f"x.ndim = {x.ndim}, x.shape = {x.shape}"
    assert ker.ndim == 4 and ker.shape[1] == 1, f"ker.ndim = {ker.ndim}, ker.shape = {ker.shape}"
    s2, s3 = x.shape[2], x.shape[3]
    if ker.shape[0] > 1: x = x.view(1,-1,s2,s3) # if the kernel is not the same for all samples
    p = ker.shape[2]//2 # padding size
    return F.pad(F.conv2d(x, ker, groups=ker.shape[0]), (p,p,p,p), mode='replicate').view(-1,1,s2,s3)

def calc_gso_batch(Ψ, rr, zz, dev=torch.device('cpu')):
    Δr, Δz = rr[:,0,1,2]-rr[:,0,1,1], zz[:,0,2,1]-zz[:,0,1,1] 
    α = (-2*(Δr**2 + Δz**2))
    β = ((Δr**2 * Δz**2) / α)
    ΔΨ = (1/β.view(-1,1,1,1)) * (Ϛ(Ψ, laplace_ker(Δr, Δz, α, dev)) - Ϛ(Ψ, dr_ker(Δr, Δz, α, dev))/rr) # grad-shafranov operator
    # ΔΨ = Ϛ(ΔΨ, gauss_ker(dev)) # apply gauss kernel
    return ΔΨ





## PLOTTING
