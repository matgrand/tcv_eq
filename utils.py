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

import torch
import torch.nn.functional as F
from torch.nn import Module, Linear, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Sequential, ConvTranspose2d
from torch.utils.data import Dataset

import builtins
import os

try: 
    JOBID = os.environ["SLURM_JOB_ID"] # get job id from slurm, when training on cluster
    DEV = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu") # nvidia
    LOCAL = False # for plotting or saving images
except:
    DEV = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu") # apple silicon / cpu
    JOBID = "local"
    LOCAL = True


from scipy.interpolate import RegularGridInterpolator
# INTERP_METHOD = 'linear' # fast, but less accurate
INTERP_METHOD = 'quintic' # slowest, but most accurate
if INTERP_METHOD == 'linear': print('Warning: using linear interpolation, which is fast but less accurate')

# DS_DIR = 'dss/ds' # where the final dataset will be stored
DS_DIR = 'dss' if LOCAL else '/nfsd/automatica/grandinmat/dss' 
os.makedirs(DS_DIR, exist_ok=True)
TRAIN_DS_PATH = f'{DS_DIR}/train_ds.npz'
EVAL_DS_PATH = f'{DS_DIR}/eval_ds.npz'

# paths to the best models
BEST_MODEL_TOT = 'best_tot.pth' 
BEST_MODEL_MSE = 'best_mse.pth'
BEST_MODEL_GSO = 'best_gso.pth'

CURR_EVAL_MODEL = 'data/2539240/best_mse.pth' # path to the 'best' model so far
# CURR_EVAL_MODEL = 'data/local/best_mse.pth' # path to the 'best' model so far
STRICT_LOAD = False # for loading the weights, should be true, but for testing varying architectures, set to false

TEST_DIR = 'test' if LOCAL else '/nfsd/automatica/grandinmat/test'
os.makedirs(TEST_DIR, exist_ok=True)

# NGR = 28 # number of grid points in the x direction
# NGZ = 65 # number of grid points in the y direction
# NGR = NGZ = 64 # number of grid points 
# NGR = NGZ = 24 # number of grid points <-
NGR = NGZ = 16 # number of grid points 

USE_CURRENTS = True # usually True
USE_PROFILES = True # false -> more realistic
USE_MAGNETIC = True # usually True
NIN = int(USE_CURRENTS)*19 + int(USE_PROFILES)*38 + int(USE_MAGNETIC)*38 # input size

NLCFS = 129 # number of LCFS points 

# read the original grid coordinates
d = loadmat('tcv_params/grid.mat')
rd, zd = d['r'].flatten(), d['z'].flatten() # original grid coordinates (DATA)
r,z = np.linspace(rd[0], rd[-1], NGR), np.linspace(zd[0], zd[-1], NGZ)  # grid coordinates
RRD, ZZD = np.meshgrid(rd, zd)  # meshgrid for the original grid coordinates (from the data)
del d, rd, zd, r, z

# load the vessel perimeter
m = loadmat('tcv_params/vess.mat')
vr, vz = m['vess_r'], m['vess_z']
vri, vzi = m['r_in'], m['z_in']
vro, vzo = m['r_out'], m['z_out']
v = np.hstack([vr, vz])
vi, vo = np.hstack([vri, vzi]), np.hstack([vro, vzo])
VESS = np.vstack([v, v[0]])[::-1]
VESSI = np.vstack([vi, vi[0]])
VESSO = np.vstack([vo, vo[0]])
del m, vr, vz, vri, vzi, vro, vzo, v, vi, vo

if not LOCAL: # Redefine the print function to always flush
    def print(*args, **kwargs): builtins.print(*args, **{**kwargs, 'flush': True})

####################################################################################################
## torch stuff
def to_tensor(x, device=torch.device("cpu")): return torch.tensor(x, dtype=torch.float32, device=device)

# custom trainable swish activation function
class ActF(Module): # swish
    def __init__(self): 
        super(ActF, self).__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
    def forward(self, x): return x*torch.sigmoid(self.beta*x)

# network architecture
class LiuqeNet(Module): # Paper net: branch + trunk conenction and everything
    def __init__(self, latent_size=32):
        super(LiuqeNet, self).__init__()
        assert latent_size % 2 == 0, "latent size should be even"
        # self.input_size, self.latent_size, self.grid_size = input_size, latent_size, grid_size
        # self.fgs = grid_size[0] * grid_size[1] # flat grid size
        self.ngr, self.ngz = NGR, NGZ # grid size
        #branch
        self.branch = Sequential(
            # View(-1, input_size),
            Linear(NIN, 64), ActF(),
            Linear(64, 32), ActF(),
            Linear(32, latent_size), ActF(),
        )
        #trunk
        def trunk_block(s): 
            return  Sequential(
                # View(-1, s),
                Linear(s, 32), ActF(),
                Linear(32, latent_size//2), ActF(),
            )
        self.trunk_r, self.trunk_z = trunk_block(self.ngr), trunk_block(self.ngz)
        # head
        self.head = Sequential(
            Linear(latent_size, 64), ActF(),
            Linear(64, self.ngr*self.ngz), ActF(),
            # View(-1, 1, *self.grid_size),
        )
    def forward(self, xb, r, z):
        xb = self.branch(xb)
        r, z = self.trunk_r(r), self.trunk_z(z) 
        xt = torch.cat((r, z), 1) # concatenate
        x = xt * xb # multiply trunk and branch
        x = self.head(x) # head net
        x = x.view(-1, 1, self.ngr, self.ngz) # reshape to grid
        return x
    
def test_network_io(verbose=True):
    if verbose: print('test_network_io')
    x, r, z = (torch.rand(1, NIN), torch.rand(1, NGR), torch.rand(1, NGZ))
    net = LiuqeNet()
    y = net(x, r, z)
    if verbose: print(f"single  -> in: {x.shape}, {r.shape}, {z.shape}, \nout: {y.shape}")
    n_sampl = 7
    nx, r, z = torch.rand(n_sampl, NIN), torch.rand(n_sampl, NGR), torch.rand(n_sampl, NGZ)
    ny = net(nx, r, z)
    if verbose: print(f"batched -> in: {nx.shape}, {r.shape}, {z.shape}, \nout: {ny.shape}")
    assert ny.shape == (n_sampl, 1, NGZ, NGR), f"Wrong output shape: {ny.shape}"

# function to load the dataset
def load_ds(ds_path):
    assert os.path.exists(ds_path), f"Dataset not found: {ds_path}"
    d = np.load(ds_path)
    # output: magnetic flux, transposed (matlab is column-major)
    X =  d["X"] # (n, NIN) # inputs: currents + measurements + profiles
    Y =  d["Y"] # (n, NGZ, NGZ) # outputs: magnetic flux
    r = d["r"] # (n, NGR) radial position of pixels 
    z = d["z"] # (n, NGZ) vertical position of pixels 
    return X, Y, r, z

####################################################################################################
class LiuqeDataset(Dataset):
    def __init__(self, ds_mat_path, verbose=True):
        self.X, self.Y, self.r, self.z = map(to_tensor, load_ds(ds_mat_path))
        self.Y = self.Y.view(-1,1,NGZ,NGR)
        if verbose: print(f"Dataset: N:{len(self)}, memory:{sum([x.element_size()*x.nelement() for x in [self.Y, self.X, self.r, self.z]])/1024**3:.2f}GB")
        # # move to DEV (doable bc the dataset is fairly small, check memory usage)
        # self.Y, self.X, self.r, self.z = self.Y.to(DEV), self.X.to(DEV), self.r.to(DEV), self.z.to(DEV)
    def __len__(self): return len(self.Y)
    def __getitem__(self, idx): return self.X[idx], self.Y[idx], self.r[idx], self.z[idx]

def _test_dataset():
    print("_test_dataset")
    ds = LiuqeDataset(EVAL_DS_PATH)
    print(f"Dataset length: {len(ds)}")
    print(f"Input shape: {ds[0][0].shape}")
    print(f"Output shape: {ds[0][1].shape}")
    n_plot = 10
    print(len(ds))
    idxs = np.random.randint(0, len(ds), n_plot)
    fig, axs = plt.subplots(1, n_plot, figsize=(3*n_plot, 5))
    for i, j in enumerate(idxs):
        Y, r, z = ds[j][1].cpu().numpy().squeeze(), ds[j][2].cpu().numpy().squeeze(), ds[j][3].cpu().numpy().squeeze()
        rr, zz = np.meshgrid(r, z)
        axs[i].contourf(rr, zz, Y, 100)
        plot_vessel(axs[i])
        # axs[i].contour(rr, zz, -Y, 20, colors="black", linestyles="dotted")
        fig.colorbar(axs[i].collections[0], ax=axs[i])
        axs[i].axis("off")
        axs[i].set_aspect("equal")
    plt.savefig(f"{TEST_DIR}/dataset_outputs.png")

    # now do the same fot the input:
    fig, axs = plt.subplots(1, n_plot, figsize=(3*n_plot, 5))
    for i, j in enumerate(idxs):
        inputs = ds[j][0].cpu().numpy().squeeze()
        if USE_CURRENTS: axs[i].plot(inputs[:19], label="currents")
        if USE_MAGNETIC: axs[i].plot(inputs[19:57], label="magnetic")
        if USE_PROFILES: axs[i].plot(inputs[57:], label="profiles")
        axs[i].legend()
        axs[i].set_title(f"Sample {j}")
        axs[i].set_xlabel("Input index")
    plt.savefig(f"{TEST_DIR}/dataset_inputs.png")


####################################################################################################
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

####################################################################################################
# kernels
def calc_laplace_df_dr_ker(hr, hz):
    α = -2*(hr**2 + hz**2)
    assert np.abs(α) > 1e-10, f"α = {α} <= 0, hr = {hr}, hz = {hz}"
    laplace_ker = np.array(([0, hr**2/α, 0], [hz**2/α, 1, hz**2/α], [0, hr**2/α, 0]))
    dr_ker = np.array(([0, 0, 0], [+1, 0, -1], [0, 0, 0]))/(2*hr*α)*(hr**2*hz**2)
    return laplace_ker, dr_ker

#calculate the Grad-Shafranov operator pytorch
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

####################################################################################################
## PLOTTING
from matplotlib.patches import PathPatch
from matplotlib.path import Path
def _fill_between_polygons(ax, inner_poly, outer_poly, **kwargs):
    """
    Fill the area between two polygons.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axes to draw to
    inner_poly : array-like, shape (n, 2)
        Vertices of the inner polygon as (x, y) pairs
    outer_poly : array-like, shape (m, 2)
        Vertices of the outer polygon as (x, y) pairs
    **kwargs : dict
        Additional arguments passed to matplotlib.patches.PathPatch
        
    Returns:
    --------
    patch : matplotlib.patches.PathPatch
        The patch representing the filled area
    """
    # Convert to numpy arrays if they aren't already
    inner_poly = np.asarray(inner_poly)
    outer_poly = np.asarray(outer_poly)
    # Ensure polygons are closed
    if not np.array_equal(inner_poly[0], inner_poly[-1]):
        inner_poly = np.vstack([inner_poly, inner_poly[0]])
    if not np.array_equal(outer_poly[0], outer_poly[-1]):
        outer_poly = np.vstack([outer_poly, outer_poly[0]])
    # Create a path with two polygons
    # The inner polygon needs to be in reverse order to create a hole
    vertices = np.vstack([outer_poly, inner_poly[::-1]])
    # Create path codes
    n_outer = len(outer_poly)
    n_inner = len(inner_poly)
    codes = np.ones(n_outer + n_inner, dtype=Path.code_type) * Path.LINETO
    codes[0] = Path.MOVETO  # Start of outer polygon
    codes[n_outer] = Path.MOVETO  # Start of inner polygon
    # Create the path and patch
    path = Path(vertices, codes)
    patch = PathPatch(path, **kwargs)
    # Add to the axes
    ax.add_patch(patch)
    return patch

def plot_vessel(ax=None, lw=1.5, alpha=1.0):
    if ax is None: ax = plt.gca()
    ax.plot(VESS[:,0], VESS[:,1], color='white', lw=lw, alpha=alpha) # most inner
    ax.plot(VESSI[:,0], VESSI[:,1], color='white', lw=lw, alpha=alpha) # inner
    ax.plot(VESSO[:,0], VESSO[:,1], color='white', lw=lw, alpha=alpha) # outer
    _fill_between_polygons(ax, VESSI, VESSO, color='gray', alpha=alpha*0.8, lw=0)
    _fill_between_polygons(ax, VESS, VESSI, color='gray', alpha=alpha*0.3, lw=0)
    return ax

def _test_plot_vessel():
    print("_test_plot_vessel")
    plt.figure()
    plot_vessel()
    plt.axis('equal')
    plt.title('Vessel 2')
    # plt.show()
    plt.savefig(f"{TEST_DIR}/vessel.png")
    # plt.close()

def plot_network_outputs(save_dir, ds, model:Module, title="test"):
    model.eval()
    os.makedirs(f"{save_dir}/imgs", exist_ok=True)
    for i in np.random.randint(0, len(ds), 2 if LOCAL else 50):  
        fig, axs = plt.subplots(2, 5, figsize=(15, 9))
        x, y, r, z = ds[i]
        x, y, r, z = x.to('cpu'), y.to('cpu'), r.to('cpu'), z.to('cpu')
        x, y, r, z = x.reshape(1,-1), y.reshape(1,1,NGZ,NGR), r.reshape(1,NGR), z.reshape(1,NGZ)
        yp = model(x, r, z)
        gso, gsop = calc_gso_batch(y, r, z), calc_gso_batch(yp, r, z)
        gso, gsop = gso.detach().numpy().reshape(NGZ,NGR), gsop.detach().numpy().reshape(NGZ,NGR)
        gso_min, gso_max = np.min([gso, gsop]), np.max([gso, gsop])
        gso_levels = np.linspace(gso_min, gso_max, 13, endpoint=True)
        # gsop = np.clip(gsop, gso_range[1], gso_range[0]) # clip to gso range
        
        yp = yp.detach().numpy().reshape(NGZ,NGR)
        y = y.detach().numpy().reshape(NGZ,NGR)
        rr, zz = np.meshgrid(r.detach().cpu().numpy(), z.detach().cpu().numpy())
        bmin, bmax = np.min([y, yp]), np.max([y, yp]) # min max Y
        blevels = np.linspace(bmin, bmax, 13, endpoint=True)
        # ψ_msex = (y - yp)**2
        # gso_msex = (gso - gsop)**2
        ψ_mae = np.abs(y - yp)
        gso_mae = np.abs(gso - gsop)
        lev0 = np.linspace(0, 10.0, 13, endpoint=True)
        lev1 = np.linspace(0, 1.0, 13, endpoint=True) 
        lev2 = np.linspace(0, 0.1, 13, endpoint=True)
        lev3 = np.linspace(0, 0.01, 13, endpoint=True)
        ε = 1e-12

        # im00 = axs[0,0].contourf(rr, zz, y, blevels)
        im00 = axs[0,0].scatter(rr, zz, c=y, s=4, vmin=bmin, vmax=bmax)
        axs[0,0].set_title("Actual")
        axs[0,0].set_aspect('equal')
        axs[0,0].set_ylabel("ψ")
        fig.colorbar(im00, ax=axs[0,0]) 
        # im01 = axs[0,1].contourf(rr, zz, yp, blevels)
        im01 = axs[0,1].scatter(rr, zz, c=yp, s=4, vmin=bmin, vmax=bmax)
        axs[0,1].set_title("Predicted")
        fig.colorbar(im01, ax=axs[0,1])
        im02 = axs[0,2].contour(rr, zz, y, blevels, linestyles='dashed')
        axs[0,2].contour(rr, zz, yp, blevels)
        axs[0,2].set_title("Contours")
        fig.colorbar(im02, ax=axs[0,2])
        # im03 = axs[0,3].contourf(rr, zz, np.clip(ψ_mae, lev2[0]+ε, lev2[-1]-ε), lev2)
        im03 = axs[0,3].scatter(rr, zz, c=ψ_mae, s=4, vmin=lev2[0], vmax=lev2[-1])
        axs[0,3].set_title(f"MAE {lev2[-1]}")
        fig.colorbar(im03, ax=axs[0,3])
        # im04 = axs[0,4].contourf(rr, zz, np.clip(ψ_mae, lev3[0]+ε, lev3[-1]-ε), lev3)
        im04 = axs[0,4].scatter(rr, zz, c=ψ_mae, s=4, vmin=lev3[0], vmax=lev3[-1])
        axs[0,4].set_title(f"MAE {lev3[-1]}")
        fig.colorbar(im04, ax=axs[0,4])

        # im10 = axs[1,0].contourf(rr, zz, gso, gso_levels)
        im10 = axs[1,0].scatter(rr, zz, c=gso, s=4, vmin=gso_min, vmax=gso_max)
        axs[1,0].set_ylabel("GSO")
        fig.colorbar(im10, ax=axs[1,0])
        # im11 = axs[1,1].contourf(rr, zz, gsop, gso_levels)
        im11 = axs[1,1].scatter(rr, zz, c=gsop, s=4, vmin=gso_min, vmax=gso_max)
        fig.colorbar(im11, ax=axs[1,1])
        im12 = axs[1,2].contour(rr, zz, gso, gso_levels, linestyles='dashed')
        axs[1,2].contour(rr, zz, gsop, gso_levels)
        fig.colorbar(im12, ax=axs[1,2])
        # im13 = axs[1,3].contourf(rr, zz, np.clip(gso_mae, lev0[0]+ε, lev0[-1]-ε), lev0)
        im13 = axs[1,3].scatter(rr, zz, c=gso_mae, s=4, vmin=lev0[0], vmax=lev0[-1])
        fig.colorbar(im13, ax=axs[1,3])
        # im14 = axs[1,4].contourf(rr, zz, np.clip(gso_mae, lev1[0]+ε, lev1[-1]-ε), lev1)
        im14 = axs[1,4].scatter(rr, zz, c=gso_mae, s=4, vmin=lev1[0], vmax=lev1[-1])
        fig.colorbar(im14, ax=axs[1,4])

        for ax in axs.flatten(): 
            ax.grid(False), ax.set_xticks([]), ax.set_yticks([]), ax.set_aspect("equal")
            plot_vessel(ax)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_visible(False)

        #suptitle
        plt.suptitle(f"[{JOBID}] LiuqeNet: {title} {i}")

        plt.tight_layout()
        plt.show() if LOCAL else plt.savefig(f"{save_dir}/imgs/net_example_{title}_{i}.png")
        
        plt.close()

if __name__ == '__main__':
    test_network_io()
    _test_dataset()
    _test_plot_vessel()
    if LOCAL: plt.show()

    