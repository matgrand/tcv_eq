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
from torch.nn import Module, Linear, Conv2d, MaxPool2d, BatchNorm2d, ReLU, Sequential, ConvTranspose2d, Tanh
from torch.utils.data import Dataset

import builtins
import os

CUDA = torch.device('cuda')
CPU = torch.device('cpu')
MPS = torch.device('mps')

DEV = CPU
try: 
    JOBID = os.environ["SLURM_JOB_ID"] # get job id from slurm, when training on cluster
    if torch.cuda.is_available():
        DEV = CUDA
        GPU_MEM, _ = torch.cuda.mem_get_info()
    LOCAL = False # for plotting or saving images
except:
    if torch.backends.mps.is_available():
        DEV = MPS
        GPU_MEM = 16 * 1024**3  # 16 GB in bytes
    JOBID = "local"
    LOCAL = True
print(f"Running JOBID: {JOBID}, on {DEV}, GPU_MEM: {GPU_MEM/1e9:.2f} GB" if DEV != CPU else f"Running JOBID: {JOBID}, on cpu")
SAVE_DIR = f"data/{JOBID}"

##################################################################################################################################
## Dataset inputs/outputs (see create_ds.m)
USE_REAL_INPUTS = True # if True, use real inputs, otherwise use random inputs
if not USE_REAL_INPUTS: print('Warning: using fitted inputs')
# input names
BM = 'Bm0' if USE_REAL_INPUTS else 'Bm1'  
FF = 'Ff0' if USE_REAL_INPUTS else 'Ff1'  
FT = 'Ft0' if USE_REAL_INPUTS else 'Ft1'  
IA = 'Ia0' if USE_REAL_INPUTS else 'Ia1'  
IP = 'Ip0' if USE_REAL_INPUTS else 'Ip1'  
IU = 'Iu0' if USE_REAL_INPUTS else 'Iu1'  
RB = 'rBt0' if USE_REAL_INPUTS else 'rBt1' 
# output names
FX = 'Fx' 
IY = 'Iy'
RQ = 'rq'
ZQ = 'zq'
INPUT_NAMES = [BM, FF, FT, IA, IP, IU, RB] # input names
OUTPUT_NAMES = [FX, IY, RQ, ZQ] # output names
DS_NAMES = INPUT_NAMES + OUTPUT_NAMES # dataset names
DS_SIZES = { BM:(38,), FF:(38,), FT:(1,), IA:(19,), IP:(1,), IU:(38,), RB:(1,),  # input sizes
             FX:(65,28), IY:(63,26), RQ:(129,), ZQ:(129,) } # output sizes

from scipy.interpolate import RegularGridInterpolator
INTERP_METHOD = 'linear' # fast, but less accurate
# INTERP_METHOD = 'quintic' # slowest, but most accurate
if INTERP_METHOD == 'linear': print('Warning: using linear interpolation, which is fast but less accurate')

# DS_DIR = 'dss/ds' # where the final dataset will be stored
DS_DIR = 'dss' if LOCAL else '/nfsd/automatica/grandinmat/dss' 
os.makedirs(f'{DS_DIR}/imgs', exist_ok=True)
TRAIN_DS_PATH = f'{DS_DIR}/train_ds.npz'
EVAL_DS_PATH = f'{DS_DIR}/eval_ds.npz'

# paths to the best models
LOSS_NAMES = ['l1', 'l2', 'l3', 'gso'] # loss names
def model_path(loss_name, save_dir=SAVE_DIR):
    assert loss_name in LOSS_NAMES, f"loss_name should be one of {LOSS_NAMES}, got {loss_name}"
    return f"{save_dir}/best_{loss_name}.pth"

BEST_MODEL_DIR = 'data/best/' # where the best models are saved
STRICT_LOAD = True # for loading the weights, should be true, but for testing varying architectures, set to false

TEST_DIR = 'test' if LOCAL else '/nfsd/automatica/grandinmat/test'
os.makedirs(TEST_DIR, exist_ok=True)

# NGR = NGZ = 24 # number of grid points 
NGR = NGZ = 16 # number of grid points <-

NIN = sum(DS_SIZES[name][0] for name in INPUT_NAMES) # 136

NLCFS = 129 # number of LCFS points 

assert NLCFS == DS_SIZES[RQ][0] == DS_SIZES[ZQ][0], f"NLCFS = {NLCFS}, DS_SIZES[RQ] = {DS_SIZES[RQ]}, DS_SIZES[ZQ] = {DS_SIZES[ZQ]}"

##################################################################################################################################


# read the original grid coordinates
d = loadmat('tcv_params/grid.mat')
rd, zd = d['r'].flatten(), d['z'].flatten() # original grid coordinates (DATA)
r,z = np.linspace(rd[0], rd[-1], NGR), np.linspace(zd[0], zd[-1], NGZ)  # grid coordinates
RRD, ZZD = np.meshgrid(rd, zd)  # meshgrid for the original grid coordinates (from the data)
RRD2, ZZD2 = np.meshgrid(rd[1:-1], zd[1:-1])  
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
def to_tensor(x, device=torch.device(CPU)): return torch.tensor(x, dtype=torch.float32, device=device)

# custom trainable swish activation function
class ActF(Module): # swish
    def __init__(self): 
        super(ActF, self).__init__()
        self.beta = torch.nn.Parameter(torch.tensor(1.0), requires_grad=True)
    def forward(self, x): return x*torch.sigmoid(self.beta*x)

# # network architecture
# class LiuqeNet(Module): 
#     def __init__(self, latent_size=32):
#         super(LiuqeNet, self).__init__()
#         assert latent_size % 2 == 0, "latent size should be even"
#         # self.input_size, self.latent_size, self.grid_size = input_size, latent_size, grid_size
#         # self.fgs = grid_size[0] * grid_size[1] # flat grid size
#         self.ngr, self.ngz = NGR, NGZ # grid size
#         #branch
#         self.branch = Sequential(
#             # View(-1, input_size),
#             Linear(NIN, 64), ActF(),
#             Linear(64, 32), ActF(),
#             Linear(32, latent_size), ActF(),
#         )
#         #trunk
#         def trunk_block(s): 
#             return  Sequential(
#                 # View(-1, s),
#                 Linear(s, 32), ActF(),
#                 Linear(32, latent_size//2), ActF(),
#             )
#         self.trunk_r, self.trunk_z = trunk_block(self.ngr), trunk_block(self.ngz)
#         # head
#         self.head = Sequential(
#             Linear(latent_size, 64), ActF(),
#             Linear(64, self.ngr*self.ngz), ActF(),
#             # View(-1, 1, *self.grid_size),
#         )
#     def forward(self, xb, r, z):
#         xb = self.branch(xb)
#         r, z = self.trunk_r(r), self.trunk_z(z) 
#         xt = torch.cat((r, z), 1) # concatenate
#         x = xt * xb # multiply trunk and branch
#         x = self.head(x) # head net
#         x = x.view(-1, 1, self.ngr, self.ngz) # reshape to grid
#         return x

PHYSICS_LS = 64 # physics latent size [ph]
GRID_LS = 32 # grid latent size [gr]
assert GRID_LS % 2 == 0, "grid latent size should be even"

class InputNet(Module): # input -> latent physics vector [x -> ph]
    def __init__(self, x_mean_std=torch.tensor([[0.0]*NIN, [1.0]*NIN], dtype=torch.float32)):
        super(InputNet, self).__init__()
        assert x_mean_std.shape == (2, NIN), f"x_mean_std.shape = {x_mean_std.shape}, NIN = {NIN}"
        self.x_mean_std = x_mean_std
        self.input_net = Sequential(
            Linear(NIN, 64), ActF(),
            Linear(64, 64), ActF(),
            Linear(64, PHYSICS_LS), Tanh(),
        )
    def forward(self, x): 
        assert x.shape[1] == NIN, f"x.shape[1] = {x.shape[1]}, NIN = {NIN}"
        x = (x - self.x_mean_std[0]) / self.x_mean_std[1] # normalize
        ph = self.input_net(x)
        return ph
    def to(self, device):
        super(InputNet, self).to(device)
        self.x_mean_std = self.x_mean_std.to(device)
        return self

class GridNet(Module): # grid -> latent grid vector [r,z,ph -> gr]
    def __init__(self):
        super(GridNet, self).__init__()
        def grid_block(s): 
            return  Sequential(
                Linear(s, 32), ActF(),
                Linear(32, GRID_LS//2), ActF(),
            )
        self.grid_r, self.grid_z = grid_block(NGR), grid_block(NGZ)
        # self.phys2grid = Sequential(
        #     Linear(PHYSICS_LS, 16), ActF(),
        #     Linear(16, GRID_LS), ActF(),
        # )
        self.phys2grid = Sequential(
            Linear(PHYSICS_LS, GRID_LS), ActF(),
        )
    def forward(self, r, z, ph):
        r, z = self.grid_r(r), self.grid_z(z) 
        gr1 = torch.cat((r, z), 1)
        gr2 = self.phys2grid(ph)
        assert gr1.shape == gr2.shape, f"gr1.shape = {gr1.shape}, gr2.shape = {gr2.shape}"
        gr = gr1 * gr2
        return gr
    
class FluxHead(Module): # grid -> flux [gr -> flux/curr_density]
    def __init__(self):
        super(FluxHead, self).__init__()
        self.head = Sequential(
            Linear(GRID_LS, 64), ActF(),
            Linear(64, NGR*NGZ), ActF(),
        )
    def forward(self, gr): 
        y = self.head(gr)
        y = y.view(-1, 1, NGR, NGZ) # reshape to grid
        return y
    
class LCFSHead(Module): # physics -> LCFS [ph -> LCFS]
    def __init__(self):
        super(LCFSHead, self).__init__()
        self.lcfs = Sequential(
            Linear(PHYSICS_LS, 32), ActF(),
            Linear(32, 32), ActF(),
            Linear(32, NLCFS*2), ActF(),
        )
    def forward(self, ph): return self.lcfs(ph)

class LiuqeNet(Module): # Liuqe net
    def __init__(self, input_net:InputNet, grid_net:GridNet, flux_head1:FluxHead, flux_head2:FluxHead, lcfs_head:LCFSHead):
        super(LiuqeNet, self).__init__()
        self.input_net = input_net
        self.grid_net = grid_net
        self.flux_head1 = flux_head1
        self.flux_head2 = flux_head2
        self.lcfs_head = lcfs_head
    def forward(self, x, r, z):
        ph = self.input_net(x)
        gr = self.grid_net(r, z, ph)
        y1 = self.flux_head1(gr)
        y2 = self.flux_head2(gr)
        y3 = self.lcfs_head(ph)
        return y1, y2, y3
    def to(self, device):
        super(LiuqeNet, self).to(device)
        self.input_net.to(device)
        return self
        
class LCFSNet(Module): # LCFS net
    def __init__(self, input_net:InputNet, lcfs_head:LCFSHead):
        super(LCFSNet, self).__init__()
        self.input_net = input_net
        self.lcfs_head = lcfs_head
    def forward(self, x):
        ph = self.input_net(x)
        lcfs = self.lcfs_head(ph)
        return lcfs
    def to(self, device):
        super(LCFSNet, self).to(device)
        self.input_net.to(device)
        return self

    
def test_network_io(verbose=True):
    v = verbose
    if v: print('test_network_io')
    # single sample
    x, r, z = (torch.rand(1, NIN), torch.rand(1, NGR), torch.rand(1, NGZ))
    input_net, grid_net, flux_head1, flux_head2, lcfs_head = InputNet(), GridNet(), FluxHead(), FluxHead(), LCFSHead()
    liuqenet = LiuqeNet(input_net, grid_net, flux_head1, flux_head2, lcfs_head)
    lcsfnet = LCFSNet(input_net, lcfs_head)
    y1, y2, y3 = liuqenet(x, r, z)
    assert y1.shape == (1, 1, NGZ, NGR), f"Wrong output shape: {y1.shape}"
    assert y2.shape == (1, 1, NGZ, NGR), f"Wrong output shape: {y2.shape}"
    assert y3.shape == (1, NLCFS*2), f"Wrong output shape: {y3.shape}"
    y = lcsfnet(x)
    assert y.shape == (1, NLCFS*2), f"Wrong output shape: {y.shape}"
    assert torch.allclose(y3, y), "y3 and y are not equal"
    if v: print(f"LiuqeNet -> in: {x.shape}, {r.shape}, {z.shape}, \n            out: {y1.shape}, {y2.shape}, {y3.shape}")
    if v: print(f"LCFSNet  -> in: {x.shape}, \n            out: {y.shape}")
    # batched
    n_sampl = 7
    nx, r, z = torch.rand(n_sampl, NIN), torch.rand(n_sampl, NGR), torch.rand(n_sampl, NGZ)
    ny1, ny2, ny3 = liuqenet(nx, r, z)
    assert ny1.shape == (n_sampl, 1, NGZ, NGR), f"Wrong output shape: {ny1.shape}"
    assert ny2.shape == (n_sampl, 1, NGZ, NGR), f"Wrong output shape: {ny2.shape}"
    assert ny3.shape == (n_sampl, NLCFS*2), f"Wrong output shape: {ny3.shape}"
    ny = lcsfnet(nx)
    assert ny.shape == (n_sampl, NLCFS*2), f"Wrong output shape: {ny.shape}"
    if v: print(f"LiuqeNet -> in: {nx.shape}, {r.shape}, {z.shape}, \n            out: {ny1.shape}, {ny2.shape}, {ny3.shape}")
    if v: print(f"LCFSNet  -> in: {nx.shape}, \n            out: {ny.shape}")

# function to load the dataset
def load_ds(ds_path):
    assert os.path.exists(ds_path), f"Dataset not found: {ds_path}"
    d = np.load(ds_path)
    # output: magnetic flux, transposed (matlab is column-major)
    X =  d["X"] # (n, NIN) # inputs: currents + measurements + profiles
    r = d["r"] # (n, NGR) radial position of pixels 
    z = d["z"] # (n, NGZ) vertical position of pixels 
    Y1 =  d["Y1"] # (n, NGZ, NGZ) # outputs: magnetic flux
    Y2 =  d["Y2"] # (n, NGZ, NGZ) # outputs: curr density
    Y3 =  d["Y3"] # (n, NLCFS*2) # outputs: last closed flux surface (LCFS)
    x_mean_std = d["x_mean_std"] # (2, NIN) # mean and std of the inputs
    return X, r, z, Y1, Y2, Y3, x_mean_std

####################################################################################################
class LiuqeDataset(Dataset):
    def __init__(self, ds_mat_path, verbose=True):
        X, r, z, Y1, Y2, Y3, x_mean_std = map(to_tensor, load_ds(ds_mat_path))
        Y1 = Y1.view(-1,1,NGZ,NGR)
        Y2 = Y2.view(-1,1,NGZ,NGR)
        self.data = [X, r, z, Y1, Y2, Y3]
        # move to DEV (doable bc the dataset is fairly small, check memory usage)
        tot_memory_ds = sum([x.element_size()*x.nelement() for x in self.data])
        gpu_free_mem = torch.cuda.mem_get_info()[0] if DEV == CUDA else np.inf
        self.on_dev = DEV != CPU and tot_memory_ds < gpu_free_mem
        if self.on_dev: self.data = [x.to(DEV) for x in self.data]
        self.x_mean_std = x_mean_std.to(DEV) if self.on_dev else x_mean_std
        if verbose: print(f"Dataset: N:{len(self)}, memory:{tot_memory_ds/1e6}MB, on_dev:{self.on_dev}")
    def __len__(self): return len(self.data[0])
    def __getitem__(self, idx): return [x[idx] for x in self.data]

def test_dataset(ds:LiuqeDataset, verbose=True):
    if verbose:
        print("test_dataset")
        print(f"Dataset length: {len(ds)}")
        print(f"Inputs: X -> {ds[0][0].shape}, r -> {ds[0][1].shape}, z -> {ds[0][2].shape}")
        print(f"Outputs: Y1 -> {ds[0][3].shape}, Y2 -> {ds[0][4].shape}, Y3 -> {ds[0][5].shape}")
    n_plot = 10
    print(len(ds))
    idxs = np.random.randint(0, len(ds), n_plot)
    fig, axs = plt.subplots(2, n_plot, figsize=(3*n_plot, 5))
    for i, j in enumerate(idxs):
        X,r,z,Y1,Y2,Y3 = map(lambda x: x.cpu().numpy(), ds[j])
        Y1, Y2 = Y1.reshape(NGZ, NGR), Y2.reshape(NGZ, NGR)
        rr, zz = np.meshgrid(r, z)
        # Y1
        axs[0,i].contourf(rr, zz, Y1, 100)
        plot_vessel(axs[0,i])
        axs[0,i].plot(Y3[:NLCFS], Y3[NLCFS:], color='gray', lw=1.5)
        fig.colorbar(axs[0,i].collections[0], ax=axs[0,i])
        axs[0,i].axis("off")
        axs[0,i].set_aspect("equal")
        # Y2
        axs[1,i].contourf(rr, zz, Y2, 100)
        plot_vessel(axs[1,i])
        axs[1,i].plot(Y3[:NLCFS], Y3[NLCFS:], color='gray', lw=1.5)
        fig.colorbar(axs[1,i].collections[0], ax=axs[1,i])
        axs[1,i].axis("off")
        axs[1,i].set_aspect("equal")

    plt.savefig(f"{TEST_DIR}/dataset_outputs.png")

    # now do the same fot the input:
    fig, axs = plt.subplots(1, n_plot, figsize=(3*n_plot, 5))
    for i, j in enumerate(idxs):
        inputs = ds[j][0].cpu()
        inputs = ((ds[j][0] - ds.x_mean_std[0]) / ds.x_mean_std[1]).cpu().numpy().squeeze() # normalize
        axs[i].plot(inputs, label='inputs')
        axs[i].legend()
        axs[i].set_title(f"Sample {j}")
        axs[i].set_xlabel("Input index")
    plt.savefig(f"{TEST_DIR}/dataset_inputs.png")
    return


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
def laplace_ker(Δr, Δz, α, dev=torch.device(CPU)): # [[0, Δr**2/α, 0], [Δz**2/α, 1, Δz**2/α], [0, Δr**2/α, 0]]
    kr, kz = Δr**2/α, Δz**2/α
    ker = torch.zeros(len(Δr),1, 3, 3, dtype=torch.float32, device=dev)
    ker[:,0,0,1], ker[:,0,1,0], ker[:,0,1,2], ker[:,0,2,1], ker[:,0,1,1] = kr, kz, kz, kr, 1
    return ker
   
def dr_ker(Δr, Δz, α, dev=torch.device(CPU)): # [[0,0,0],[-1,0,+1],[0,0,0]] * (Δr**2 * Δz**2) / (2*Δr*α)
    ker = torch.zeros(len(Δr),1, 3, 3, dtype=torch.float32, device=dev)
    k = (Δr**2 * Δz**2) / (2*Δr*α)
    ker[:,0,1,0], ker[:,0,1,2] = -k, k
    return ker

def gauss_ker(dev=torch.device(CPU)):
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

def calc_gso_batch(Ψ, r, z, dev=torch.device(CPU)):
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

def test_plot_vessel():
    print("test_plot_vessel")
    plt.figure()
    plot_vessel()
    plt.axis('equal')
    plt.title('Vessel 2')
    # plt.show()
    plt.savefig(f"{TEST_DIR}/vessel.png")
    # plt.close()s

def plot_network_outputs(ds:LiuqeDataset, model:LiuqeNet, title="test"):
    model.eval()
    model.to(CPU)
    os.makedirs(f"{SAVE_DIR}/imgs", exist_ok=True)
    for i in np.random.randint(0, len(ds), 2 if LOCAL else 50):  
        fig, axs = plt.subplots(2, 5, figsize=(16, 9))
        x, r, z, y1, y2, y3 = ds[i]
        x, r, z, y1, y2, y3 = map(lambda x: x.to(CPU), [x,r,z,y1,y2,y3])
        x, r, z, y1, y2, y3 = x.reshape(1,-1), r.reshape(1,NGR), z.reshape(1,NGZ), y1.reshape(1,1,NGZ,NGR), y2.reshape(1,1,NGZ,NGR), y3.reshape(1,2*NLCFS)
        yp1, yp2, yp3 = model(x, r, z)
        gso, gsop = calc_gso_batch(y1, r, z), calc_gso_batch(yp1, r, z)
        gso, gsop = gso.detach().numpy().reshape(NGZ,NGR), gsop.detach().numpy().reshape(NGZ,NGR)        
        rr, zz = np.meshgrid(r.detach().cpu().numpy(), z.detach().cpu().numpy())
        yp1 = yp1.detach().numpy().reshape(NGZ,NGR)
        y1 = y1.detach().numpy().reshape(NGZ,NGR)
        yp2 = yp2.detach().numpy().reshape(NGZ,NGR)
        y2 = y2.detach().numpy().reshape(NGZ,NGR)
        yp3 = yp3.detach().numpy().reshape(2*NLCFS)
        y3 = y3.detach().numpy().reshape(2*NLCFS)
        min1, max1 = np.min([y1, yp1]), np.max([y1, yp1]) # min max Y1
        min2, max2 = np.min([y2, yp2]), np.max([y2, yp2]) # min max Y2
        levels1 = np.linspace(min1, max1, 13, endpoint=True)
        levels2 = np.linspace(min2, max2, 13, endpoint=True)
        y1_mae = np.abs(y1 - yp1)
        y2_mae = np.abs(y2 - yp2)
        lev0 = np.linspace(0, 1.0, 13, endpoint=True)
        lev1 = np.linspace(0, 0.1, 13, endpoint=True) 
        lev2 = np.linspace(0, 1.0, 13, endpoint=True)
        lev3 = np.linspace(0, 0.1, 13, endpoint=True)

        lw3, col3 = 1.5, 'gray'
        im00 = axs[0,0].scatter(rr, zz, c=y1, s=4, vmin=min1, vmax=max1)
        axs[0,0].plot(y3[:NLCFS], y3[NLCFS:], col3, lw=lw3)
        axs[0,0].set_title("Actual")
        axs[0,0].set_aspect('equal')
        axs[0,0].set_ylabel("Y1")
        fig.colorbar(im00, ax=axs[0,0]) 
        im01 = axs[0,1].scatter(rr, zz, c=yp1, s=4, vmin=min1, vmax=max1)
        axs[0,1].plot(yp3[:NLCFS], yp3[NLCFS:], col3, lw=lw3)
        axs[0,1].set_title("Predicted")
        fig.colorbar(im01, ax=axs[0,1])
        im02 = axs[0,2].contour(rr, zz, y1, levels1, linestyles='dashed')
        axs[0,2].contour(rr, zz, yp1, levels1)
        axs[0,2].set_title("Contours")
        fig.colorbar(im02, ax=axs[0,2])
        im03 = axs[0,3].scatter(rr, zz, c=y1_mae, s=4, vmin=lev2[0], vmax=lev2[-1])
        axs[0,3].plot(y3[:NLCFS], y3[NLCFS:], col3, lw=lw3, linestyle='dashed')
        axs[0,3].plot(yp3[:NLCFS], yp3[NLCFS:], col3, lw=lw3)
        axs[0,3].set_title(f"MAE {lev2[-1]}")
        fig.colorbar(im03, ax=axs[0,3])
        im04 = axs[0,4].scatter(rr, zz, c=y1_mae, s=4, vmin=lev3[0], vmax=lev3[-1])
        axs[0,4].plot(y3[:NLCFS], y3[NLCFS:], col3, lw=lw3, linestyle='dashed')
        axs[0,4].plot(yp3[:NLCFS], yp3[NLCFS:], col3, lw=lw3)
        axs[0,4].set_title(f"MAE {lev3[-1]}")
        fig.colorbar(im04, ax=axs[0,4])

        im10 = axs[1,0].scatter(rr, zz, c=y2, s=4, vmin=min2, vmax=max2)
        axs[1,0].plot(y3[:NLCFS], y3[NLCFS:], col3, lw=lw3)
        axs[1,0].set_ylabel("Y2")
        fig.colorbar(im10, ax=axs[1,0])
        im11 = axs[1,1].scatter(rr, zz, c=yp2, s=4, vmin=min2, vmax=max2)
        axs[1,1].plot(yp3[:NLCFS], yp3[NLCFS:], col3, lw=lw3)
        fig.colorbar(im11, ax=axs[1,1])
        im12 = axs[1,2].contour(rr, zz, y2, levels2, linestyles='dashed')
        axs[1,2].contour(rr, zz, yp2, levels2)
        fig.colorbar(im12, ax=axs[1,2])
        im13 = axs[1,3].scatter(rr, zz, c=y2_mae, s=4, vmin=lev0[0], vmax=lev0[-1])
        axs[1,3].plot(y3[:NLCFS], y3[NLCFS:], col3, lw=lw3, linestyle='dashed')
        axs[1,3].plot(yp3[:NLCFS], yp3[NLCFS:], col3, lw=lw3)
        fig.colorbar(im13, ax=axs[1,3])
        im14 = axs[1,4].scatter(rr, zz, c=y2_mae, s=4, vmin=lev1[0], vmax=lev1[-1])
        axs[1,4].plot(y3[:NLCFS], y3[NLCFS:], col3, lw=lw3, linestyle='dashed')
        axs[1,4].plot(yp3[:NLCFS], yp3[NLCFS:], col3, lw=lw3)
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
        plt.show() if LOCAL else plt.savefig(f"{SAVE_DIR}/imgs/net_example_{title}_{i}.png")
        plt.close()
    return
    
def plot_lcfs_net_out(ds:LiuqeDataset, model:LCFSNet, title='test'):
    model.eval()
    model.to(CPU)
    os.makedirs(f"{SAVE_DIR}/imgs", exist_ok=True)
    lw3 = 1.5
    for i in np.random.randint(0, len(ds), 5 if LOCAL else 50):  
        plt.figure(figsize=(16, 9))
        x, y3 = ds[i][0].to(CPU), ds[i][5].to(CPU)
        x = x.reshape(1, -1)
        yp3 = model(x)
        yp3 = yp3.detach().numpy().reshape(2 * NLCFS)
        y3 = y3.detach().numpy().reshape(2 * NLCFS)
        # convert to 2d points
        y3 = np.array([y3[:NLCFS], y3[NLCFS:]]).T
        yp3 = np.array([yp3[:NLCFS], yp3[NLCFS:]]).T
        err = np.linalg.norm(y3 - yp3, axis=-1)
        norm_err = err / np.max(err)
        err_colors = np.array([plt.cm.viridis(norm_err[j]) for j in range(NLCFS)])

        plt.subplot(1, 3, 1)
        plt.plot(y3[:,0], y3[:,1], lw=lw3, label='actual')
        plt.plot(yp3[:,0], yp3[:,1], lw=lw3, label='predicted')
        plot_vessel()
        plt.title("LCFS")
        plt.axis('equal')
        plt.xlabel("R")
        plt.ylabel("Z")
        plt.legend(loc='upper right')

        plt.subplot(1, 3, 2)
        α = 0.8
        plt.plot(y3[:,0], y3[:,1], lw=1, color='white', alpha=α)
        plt.plot(yp3[:,0], yp3[:,1], lw=1, color='white', alpha=α)
        sc = plt.scatter(yp3[:,0], yp3[:,1], c=err, s=20, cmap='viridis')
        plt.colorbar(sc)
        plt.title("LCFS MAE")
        plt.axis('equal')
        plt.xlabel("R")
        plt.ylabel("Z")

        plt.subplot(1, 3, 3)
        plt.plot(y3[:,0], y3[:,1], lw=1, color='white', alpha=α)
        plt.plot(yp3[:,0], yp3[:,1], lw=1, color='white', alpha=α)
        for j in range(NLCFS):
            plt.plot([y3[j,0], yp3[j,0]], [y3[j,1], yp3[j,1]], color=err_colors[j], lw=2)
        plt.title("LCFS MAE")
        plt.axis('equal')
        plt.xlabel("R")
        plt.ylabel("Z")

        plt.suptitle(f"[{JOBID}] LCFSNet: {title} {i}")
        plt.tight_layout()
        plt.show() if LOCAL else plt.savefig(f"{SAVE_DIR}/imgs/lcfs_example_{title}_{i}.png")
        plt.close()
    return



if __name__ == '__main__':
    test_network_io()
    test_plot_vessel()
    if LOCAL: plt.show()

    