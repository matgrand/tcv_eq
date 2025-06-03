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
BM = f'Bm{"0" if USE_REAL_INPUTS else "1"}'  
FF = f'Ff{"0" if USE_REAL_INPUTS else "1"}'  
FT = f'Ft{"0" if USE_REAL_INPUTS else "1"}'  
IA = f'Ia{"0" if USE_REAL_INPUTS else "1"}'  
IP = f'Ip{"0" if USE_REAL_INPUTS else "1"}'  
IU = f'Iu{"0" if USE_REAL_INPUTS else "1"}'  
RB = f'rBt{"0" if USE_REAL_INPUTS else "1"}' 
# output names
FX = 'Fx' 
IY = 'Iy'
BR = 'Br' # recalculated in python, not in matlab
BZ = 'Bz' # recalculated in python, not in matlab
RQ = 'rq'
ZQ = 'zq'
#more names
SEP = 'sep' # this is the LCFS, last closed flux surface, or separatrix
PHYS = 'phys' # physics inputs
PTS = 'pts' # points 

INPUT_NAMES = [BM, FF, FT, IA, IP, IU, RB] # input names
OUTPUT_NAMES = [FX, IY, BR, BZ, RQ, ZQ] # output names
DS_NAMES = INPUT_NAMES + OUTPUT_NAMES # dataset names
DS_SIZES = { 
    BM:(38,), FF:(38,), FT:(1,), IA:(19,), IP:(1,), IU:(38,), RB:(1,),  # input sizes
    FX:(65,28), IY:(63,26), BR:(65,28), BZ:(65,28), RQ:(129,), ZQ:(129,),  # output sizes
    SEP:(2*129,), # LCFS/SEP size, 2*NLCFS points (r,z) for the last closed flux surface
}
DTYPE = 'float32'

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

PHYSICS_LS = 64 # physics latent size [ph]

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

class FHead(Module): # [pt, ph] -> [1] function (flux/Br/Bz/curr density) 
    def __init__(self):
        super(FHead, self).__init__()
        self.head = Sequential(
            Linear(PHYSICS_LS+2, 64), ActF(),
            Linear(64, 1), ActF(),
        )
    def forward(self, gr): return self.head(gr)
    
class LCFSHead(Module): # physics -> LCFS [ph -> LCFS]
    def __init__(self):
        super(LCFSHead, self).__init__()
        self.lcfs = Sequential(
            Linear(PHYSICS_LS, 32), ActF(),
            Linear(32, 32), ActF(),
            Linear(32, NLCFS*2), ActF(),
        )
    def forward(self, ph): return self.lcfs(ph)

def concat_pts_ph(pts, ph):
    # pts: (BS, NP, 2)
    # ph: (BS, PHYSICS_LS)
    assert pts.dim() == 3, f'Expected pts to be of shape (BS, NPTS, 2), got {pts.shape}'
    bs, npts = pts.shape[:2]
    # Reshape ph to (BS, 1, PHYSICS_LS) and expand to (BS, NPTS, PHYSICS_LS)
    assert ph.dim() == 2, f'Expected ph to be of shape (BS, PX), got {ph.shape}'
    x_expanded = ph.view(bs, 1, PHYSICS_LS).expand(bs, npts, PHYSICS_LS)
    # Concatenate along the last dimension
    out = torch.cat([pts, x_expanded], dim=-1)  # (BS, NP, PHYSICS_LS+2)
    return out

class FullNet(Module): # [pt, ph] -> [1] function (flux/Br/Bz/curr density)
    def __init__(self, input_net:InputNet, fx_head:FHead, iy_head:FHead, br_head:FHead, bz_head:FHead, lcfs_head:LCFSHead):
        super(FullNet, self).__init__()
        self.input_net = input_net
        self.fx_head = fx_head
        self.iy_head = iy_head
        self.br_head = br_head
        self.bz_head = bz_head
        self.lcfs_head = lcfs_head
    def to(self, device):
        super(FullNet, self).to(device)
        self.input_net.to(device)
        return self
    def forward(self, x, pts):
        assert pts.shape[-1] == 2, f"pts.shape[-1] = {pts.shape[-1]}, should be 2 (r,z)"
        assert pts.dim() == 3, f"pts.dim() = {pts.dim()}, should be 3 (batch, n_points, 2)"
        assert x.dim() == 2, f"x.dim() = {x.dim()}, should be 2 (batch, NIN)"
        assert x.shape[-1] == NIN, f"x.shape[1] = {x.shape[1]}, NIN = {NIN}"
        assert pts.shape[0] == x.shape[0], f"pts.shape[0] = {pts.shape[0]}, x.shape[0] = {x.shape[0]}, should be equal (batch size)"
        ph = self.input_net(x) # get physics vector
        assert ph.dim() == 2 and ph.shape[1] == PHYSICS_LS, f"ph.dim() = {ph.dim()}, ph.shape[1] = {ph.shape[1]}, PHYSICS_LS = {PHYSICS_LS}"
        pts_ph = concat_pts_ph(pts, ph) # concatenate pts and ph
        assert pts_ph.dim() == 3 and pts_ph.shape[2] == 2 + PHYSICS_LS, f"pts_ph.dim() = {pts_ph.dim()}, pts_ph.shape[2] = {pts_ph.shape[2]}, should be 2 + PHYSICS_LS = {2 + PHYSICS_LS}"
        fx = self.fx_head(pts_ph) # get flux
        iy = self.iy_head(pts_ph) # get current density
        br = self.br_head(pts_ph) # get Br
        bz = self.bz_head(pts_ph) # get Bz
        assert fx.dim() == 3 and fx.shape[-1] == 1, f"fx.dim() = {fx.dim()}, fx.shape[-1] = {fx.shape[-1]}, should be 2 (batch, 1)"
        assert iy.dim() == 3 and iy.shape[-1] == 1, f"iy.dim() = {iy.dim()}, iy.shape[-1] = {iy.shape[-1]}, should be 2 (batch, 1)"
        assert br.dim() == 3 and br.shape[-1] == 1, f"br.dim() = {br.dim()}, br.shape[-1] = {br.shape[-1]}, should be 2 (batch, 1)"
        assert bz.dim() == 3 and bz.shape[-1] == 1, f"bz.dim() = {bz.dim()}, bz.shape[-1] = {bz.shape[-1]}, should be 2 (batch, 1)"
        lcfs = self.lcfs_head(ph) # get LCFS
        assert lcfs.dim() == 2 and lcfs.shape[1] == NLCFS*2, f"lcfs.dim() = {lcfs.dim()}, lcfs.shape[1] = {lcfs.shape[1]}, should be 2*NLCFS = {NLCFS*2}"
        return fx, iy, br, bz, lcfs # return flux, current density, Br, Bz, LCFS

        
class LiuqeRTNet(Module): # LCFS net
    def __init__(self, input_net:InputNet, fx_head:FHead, br_head:FHead, bz_head:FHead):
        super(LiuqeRTNet, self).__init__()
        self.input_net = input_net
        self.fx_head = fx_head
        self.br_head = br_head
        self.bz_head = bz_head
    def to(self, device):
        super(LiuqeRTNet, self).to(device)
        self.input_net.to(device)
        return self
    def forward(self, x, pts):
        assert pts.shape[-1] == 2, f"pts.shape[-1] = {pts.shape[-1]}, should be 2 (r,z)"
        assert pts.dim() == 3, f"pts.dim() = {pts.dim()}, should be 3 (batch, n_points, 2)"
        assert x.dim() == 2, f"x.dim() = {x.dim()}, should be 2 (batch, NIN)"
        assert x.shape[-1] == NIN, f"x.shape[1] = {x.shape[1]}, NIN = {NIN}"
        assert pts.shape[0] == x.shape[0], f"pts.shape[0] = {pts.shape[0]}, x.shape[0] = {x.shape[0]}, should be equal (batch size)"
        ph = self.input_net(x) # get physics vector
        assert ph.dim() == 2 and ph.shape[1] == PHYSICS_LS, f"ph.dim() = {ph.dim()}, ph.shape[1] = {ph.shape[1]}, PHYSICS_LS = {PHYSICS_LS}"
        pts_ph = concat_pts_ph(pts, ph) # concatenate pts and ph
        assert pts_ph.dim() == 3 and pts_ph.shape[2] == 2 + PHYSICS_LS, f"pts_ph.dim() = {pts_ph.dim()}, pts_ph.shape[2] = {pts_ph.shape[2]}, should be 2 + PHYSICS_LS = {2 + PHYSICS_LS}"
        fx = self.fx_head(pts_ph) # get flux
        br = self.br_head(pts_ph) # get Br
        bz = self.bz_head(pts_ph) # get Bz
        return fx, br, bz

    
def test_network_io(verbose=True):
    v = verbose
    if v: print('test_network_io')
    n_points = 5 # number of points to sample
    # single sample
    x, pts = (torch.rand(1, NIN), torch.rand(1, n_points, 2)) # x: (1, NIN), pts: (1, n_points, 2)
    full_net = FullNet(input_net=InputNet(),
                      fx_head=FHead(),
                      iy_head=FHead(),
                      br_head=FHead(),
                      bz_head=FHead(),
                      lcfs_head=LCFSHead())
    rt_net = LiuqeRTNet(full_net.input_net,
                        full_net.fx_head,
                        full_net.br_head,
                        full_net.bz_head)
    fx, iy, br, bz, lcfs = full_net(x, pts)
    fx2, br2, bz2 = rt_net(x, pts)
    assert fx.shape == (1, n_points, 1), f"fx.shape = {fx.shape}, should be (1, {n_points}, 1)"
    assert fx.shape == iy.shape == br.shape == bz.shape, "fx, iy, br, bz should have the same shape"
    assert fx.shape == fx2.shape == br2.shape == bz2.shape, "fx, fx2, br2, bz2 should have the same shape" 
    assert lcfs.shape == (1, NLCFS*2), f"lcfs.shape = {lcfs.shape}, should be (1, {NLCFS*2})"
    assert torch.allclose(fx, fx2), f"fx and fx2 should be close, but got {torch.max(torch.abs(fx-fx2))}"
    assert torch.allclose(br, br2), f"br and br2 should be close, but got {torch.max(torch.abs(br-br2))}"
    assert torch.allclose(bz, bz2), f"bz and bz2 should be close, but got {torch.max(torch.abs(bz-bz2))}"
    if v: print(f"FullNet -> in: [{x.shape}, {pts.shape}], \n            out: [{fx.shape}, {iy.shape}, {br.shape}, {bz.shape}, {lcfs.shape}]")
    if v: print(f"LiuqeRTNet  -> in: [{x.shape}, {pts.shape}], \n            out: [{fx2.shape}, {br2.shape}, {bz2.shape}]") 
    # batched
    bs = 7
    x, pts = torch.rand(bs, NIN), torch.rand(bs, n_points, 2) # x: (bs, NIN), pts: (bs, n_points, 2)
    fx, iy, br, bz, lcfs = full_net(x, pts)
    fx2, br2, bz2 = rt_net(x, pts)
    assert fx.shape == (bs, n_points, 1), f"fx.shape = {fx.shape}, should be ({bs}, {n_points}, 1)"
    assert fx.shape == iy.shape == br.shape == bz.shape, "fx, iy, br, bz should have the same shape"
    assert fx.shape == fx2.shape == br2.shape == bz2.shape, "fx, fx2, br2, bz2 should have the same shape"
    assert lcfs.shape == (bs, NLCFS*2), f"lcfs.shape = {lcfs.shape}, should be ({bs}, {NLCFS*2})"
    assert torch.allclose(fx, fx2), f"fx and fx2 should be close, but got {torch.max(torch.abs(fx-fx2))}"
    assert torch.allclose(br, br2), f"br and br2 should be close, but got {torch.max(torch.abs(br-br2))}"
    assert torch.allclose(bz, bz2), f"bz and bz2 should be close, but got {torch.max(torch.abs(bz-bz2))}"
    if v: print(f"FullNet -> in: [{x.shape}, {pts.shape}], \n            out: [{fx.shape}, {iy.shape}, {br.shape}, {bz.shape}, {lcfs.shape}]")
    if v: print(f"LiuqeRTNet  -> in: [{x.shape}, {pts.shape}], \n            out: [{fx2.shape}, {br2.shape}, {bz2.shape}]")

# function to load the dataset
def load_ds(ds_path):
    assert os.path.exists(ds_path), f"Dataset not found: {ds_path}"
    d = np.load(ds_path)
    r = {
        PHYS:d[PHYS], 
        PTS:d[PTS], 
        FX:d[FX], 
        IY:d[IY], 
        BR:d[BR], 
        BZ:d[BZ], 
        SEP:d[SEP], 
        "x_mean_std":d["x_mean_std"]
    }
    return r

####################################################################################################
class LiuqeDataset(Dataset):
    def __init__(self, ds_path, verbose=True):
        d = load_ds(ds_path)
        x_mean_std = to_tensor(d.pop("x_mean_std")) # mean and std for inputs
        print(f"Dataset loaded from {ds_path}, keys: {d.keys()}")
        self.data = {k:to_tensor(v) for k,v in d.items()} # convert to tensors
        # move to DEV (doable bc the dataset is fairly small, check memory usage)
        tot_memory_ds = sum([x.element_size()*x.nelement() for x in self.data.values()]) # total memory in bytes
        gpu_free_mem = torch.cuda.mem_get_info()[0] if DEV == CUDA else np.inf
        self.on_dev = DEV != CPU and tot_memory_ds < gpu_free_mem
        if self.on_dev: self.data = [x.to(DEV) for x in self.data.values()]
        self.x_mean_std = x_mean_std.to(DEV) if self.on_dev else x_mean_std
        if verbose: print(f"Dataset: N:{len(self)}, memory:{tot_memory_ds/1e6}MB, on_dev:{self.on_dev}")
    def __len__(self): return len(self.data[PHYS]) # number of samples, all data should have the same length
    def __getitem__(self, idx): return [x[idx] for x in self.data.values()] # return all data as a tuple

def test_dataset(ds:LiuqeDataset, verbose=True):
    if verbose:
        print("test_dataset")
        print(f"Dataset length: {len(ds)}")
        print(f"Inputs: X -> {ds[0][0].shape}, r -> {ds[0][1].shape}, z -> {ds[0][2].shape}")
        print(f"Outputs: Y1 -> {ds[0][3].shape}, Y2 -> {ds[0][4].shape}, Y3 -> {ds[0][5].shape}")
    n_plot = 10
    print(len(ds))
    idxs = np.random.randint(0, len(ds), n_plot)
    plt.figure(figsize=(15, 3*n_plot))
    for i, j in enumerate(idxs):
        x, pts, fx, iy, br, bz, sep = map(lambda x: x.cpu().numpy(), ds[j])
        μ, σ = ds.x_mean_std.cpu().numpy()
        x = (x-μ) / σ  # normalize inputs
        r,z = pts[:,0], pts[:,1]
        ms = 1
        plt.subplot(n_plot, 5, i*5+1)
        plt.scatter(r, z, c=fx, s=ms), plt.title('FX')
        plt.plot(sep[:NLCFS], sep[NLCFS:], 'gray', lw=2)
        plot_vessel(), plt.axis('equal'), plt.axis('off'), plt.colorbar()
        plt.subplot(n_plot, 5, i*5+2)
        plt.scatter(r, z, c=iy, s=ms), plt.title('IY')
        plt.plot(sep[:NLCFS], sep[NLCFS:], 'gray', lw=2)
        plot_vessel(), plt.axis('equal'), plt.axis('off'), plt.colorbar()
        plt.subplot(n_plot, 5, i*5+3)
        plt.scatter(r, z, c=br, s=ms), plt.title('Br')
        plt.plot(sep[:NLCFS], sep[NLCFS:], 'gray', lw=2)
        plot_vessel(), plt.axis('equal'), plt.axis('off'), plt.colorbar()
        plt.subplot(n_plot, 5, i*5+4)
        plt.scatter(r, z, c=bz, s=ms), plt.title('Bz')
        plt.plot(sep[:NLCFS], sep[NLCFS:], 'gray', lw=2)
        plot_vessel(), plt.axis('equal'), plt.axis('off'), plt.colorbar()
        plt.subplot(n_plot, 5, i*5+5)
        plt.bar(np.arange(len(x)), x), plt.title('Physical Inputs')
        plt.tight_layout()
    plt.savefig(f"{TEST_DIR}/dataset_examples.png")

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

def sample_random_points(n):
    pts = np.zeros((n, 2), dtype=DTYPE)
    pts[:,0] = uniform(RRD[0,0], RRD[0,-1], n) # r
    pts[:,1] = uniform(ZZD[0,0], ZZD[-1,0], n) # z
    return pts

def interp_pts(f:np.array, pts:np.array, gr=RRD[0,:], gz=ZZD[:,0], method=INTERP_METHOD):
    """
    Interpolate the function f at the points pts using the grid gr, gz.
    f should be a 2D array with shape (len(gr), len(gz)).
    pts should be a 2D array with shape (n, 2), where n is the number of points.
    """
    assert f.ndim == 2, f"f.ndim = {f.ndim}, f.shape = {f.shape}"
    assert pts.ndim == 2 and pts.shape[1] == 2, f"pts.ndim = {pts.ndim}, pts.shape = {pts.shape}"
    interp_func = RegularGridInterpolator((gr, gz), f.T, method=method)
    return interp_func(pts).reshape(-1)

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

# calculate Br, Bz from the flux map (Fx)
def meqBrBz(f:np.ndarray):
    if f.ndim == 2: f = f.reshape((1, *f.shape)) 
    assert f.ndim == 3, f'Input array must be 2D or 3D, got {f.ndim}D'
    assert f.shape[1:] == (65, 28), f'Input array must have shape (N, 28, 65), got {f.shape}'

    dr, dz, r = RRD[0,1]-RRD[0,0], ZZD[1,0]-ZZD[0,0], RRD[0]
    i4pirdr, i4pirdz = 1/(4*np.pi*dr*r), 1/(4*np.pi*dz*r)
    Br, Bz = np.zeros_like(f, dtype=DTYPE), np.zeros_like(f, dtype=DTYPE)

    # Br = -1/(2*pi*R)* dF/dz
    # Central differences for interior points: F/dz[i] =  F[i-1] - F[i+1]/(2*dz)
    Br[:,1:-1,:]    = -i4pirdz *  (f[:,2:,:] - f[:,:-2,:])
    # At grid boundary i, use: dF/dz[i] = (-F(i+2) + 4*F(i+1) - 3*F(i))/(2*dz)
    Br[:,-1,:]      = -i4pirdz * (+f[:,-3,:] - 4*f[:,-2,:] + 3*f[:,-1,:])
    Br[:,0,:]       = -i4pirdz * (-f[:,2,:]  + 4*f[:,1,:]  - 3*f[:,0,:])

    # Bz = 1/(2*pi*R)* dF/dr
    # Central differences dF/dz[i] =  F[i-1] - F[i+1]/(2*dz)
    Bz[:,:,1:-1]    = i4pirdr[1:-1] * (f[:,:,2:] - f[:,:,:-2])
    # At grid boundary i, use: dF/dz[i] = (-F(i+2) + 4*F(i+1) - 3*F(i))/(2*dz)
    Bz[:,:,-1]      = i4pirdr[-1] * (f[:,:,-3] - 4*f[:,:,-2] + 3*f[:,:,-1])
    Bz[:,:,0]       = i4pirdr[0] *  (-f[:,:,2] + 4*f[:,:,1]  - 3*f[:,:,0])
    return Br, Bz

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

def plot_network_outputs(ds:LiuqeDataset, model:FullNet, title="test"):
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
        plt.suptitle(f"[{JOBID}] FullNet: {title} {i}")
        plt.tight_layout()
        plt.show() if LOCAL else plt.savefig(f"{SAVE_DIR}/imgs/net_example_{title}_{i}.png")
        plt.close()
    return
    
def plot_lcfs_net_out(ds:LiuqeDataset, model:LiuqeRTNet, title='test'):
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

        plt.suptitle(f"[{JOBID}] LiuqeRTNet: {title} {i}")
        plt.tight_layout()
        plt.show() if LOCAL else plt.savefig(f"{SAVE_DIR}/imgs/lcfs_example_{title}_{i}.png")
        plt.close()
    return



if __name__ == '__main__':
    test_network_io()
    test_plot_vessel()
    if LOCAL: plt.show()

    