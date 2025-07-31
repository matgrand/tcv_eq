from utils import *
np.set_printoptions(precision=4)

OUT_DIR = 'test_shots'; #% more space available

TIME_INTERV = [0.4, 0.9]; #% time interval
DEC = 1; #% decimation factor

# ONNX_NET_PATH = 'onnx_net_forward/net.onnx' #bad opset 15
# ONNX_NET_PATH = 'data/best_old/net.onnx'  
# ONNX_NET_PATH = 'data/best/net.onnx'
# ONNX_NET_PATH = 'data/2988088/net.onnx'
# ONNX_NET_PATH = 'data/2777016/net.onnx'
# ONNX_NET_PATH = 'data/3009496/net.onnx' 
ONNX_NET_PATH = 'data/3009431/net.onnx' 

# ONNX_NET_PATH1 = 'data/local/net_dyn.onnx'
# ONNX_NET_PATH2 = 'data/local/net_dyn_ops17.onnx'
# ONNX_NET_PATH1 = 'data/3009496/net.onnx'
# ONNX_NET_PATH2 = 'data/3009496/net_dyn_ops17.onnx'



shots = [
    # 79742, # single null
    # 86310, # double null
    # 78893, # negative triangularity
    # 83848, # ?
    78071  # standard, test ctrl pts (t=0.571) (warn: theta is wrong)
]

## functions
def load_shot(shot, shots_dir=OUT_DIR):
    file_path = f'{shots_dir}/{shot}_cache.mat'
    assert os.path.exists(file_path), f"Shot file {file_path} does not exist."
    d = loadmat(file_path)
    return d['t'], d['Fx'], d['Br'], d['Bz'], d['Bm'], d['Ff'], d['Ft'], d['Ia'], d['Ip'], d['Iu'], d['rBt']


## net stuff
assert os.path.exists(ONNX_NET_PATH), f"ONNX net file {ONNX_NET_PATH} does not exist."
import onnx
onnx_model = onnx.load(ONNX_NET_PATH)
onnx.checker.check_model(onnx_model)

import onnxruntime as ort
ort_session = ort.InferenceSession(ONNX_NET_PATH)
# input names: phys, r, z, output names: rt

def net_forward(phys, r, z):
    phys = phys.astype(np.float32)
    r = r.astype(np.float32)
    z = z.astype(np.float32)
    inputs = {
        'phys': phys,
        'r': r,
        'z': z
    }
    outputs = ort_session.run(None, inputs)[0]
    Fx, Br, Bz = outputs[:, 0], outputs[:, 1], outputs[:, 2]
    return Fx, Br, Bz

# ## Pytorch version
# PT_NET_DIR = 'data/3009431/'
# x_mean_std = to_tensor(np.load(f'{PT_NET_DIR}/x_mean_std.npz')['x_mean_std'])
# m = FullNet(InputNet(x_mean_std), PtsEncoder(), FHead(3), FHead(1), LCFSHead())
# m = m.to(CPU) # move to CPU
# m.load_state_dict(torch.load(model_path(FX, save_dir=PT_NET_DIR), weights_only=True, map_location=torch.device(CPU))) # load pretrained model
# net = LiuqeRTNet(m.input_net, m.pts_enc, m.rt_head) # create LiuqeRTNet
# def net_forward(phys, r, z):
#     phys = torch.tensor(phys, dtype=torch.float32)
#     r = torch.tensor(r, dtype=torch.float32)
#     z = torch.tensor(z, dtype=torch.float32)
#     with torch.no_grad():
#         outputs = net(phys, r, z)
#     Fx, Br, Bz = outputs[:, 0].numpy(), outputs[:, 1].numpy(), outputs[:, 2].numpy()
#     return Fx, Br, Bz

# dummy control points

nq = 5  # number of control points
thetaq = np.linspace(0, 2 * np.pi, nq + 1)[:-1]
rq = 0.88 + 0.15 * np.cos(thetaq)
zq = 0.20 + 0.45 * np.sin(thetaq)
print(f'Control points: {rq}, \n{zq}')

# load tcv grid
tcv_grid = loadmat('tcv_params/grid.mat')
gr = tcv_grid['r'].squeeze()
gz = tcv_grid['z'].squeeze()
rg, zg = np.meshgrid(gr, gz)
rg, zg = rg.flatten(), zg.flatten()

print(f'Shots: {shots}')
print('Starting tests...\n')

for shot in shots:
    print(f'\n\nShot {shot}')
    # Load shot data
    t, Fx, Br, Bz, Bm, Ff, Ft, Ia, Ip, Iu, rBt = load_shot(shot)

    tidxs = np.where((t >= TIME_INTERV[0]) & (t <= TIME_INTERV[1]))[0]
    tidxs = tidxs[::DEC]
    nt = len(tidxs)
    assert nt >= 1, 'No time samples in the specified interval'
    print(f'Time samples: {nt}')

    # LIUQE/true values
    t = t[tidxs].reshape(-1)
    FxLg = Fx[:, :, tidxs]

    # Preallocate
    FxNg = np.zeros((65 * 28, nt))
    FxLq = np.zeros((nq, nt))
    FxNq = np.zeros((nq, nt))

    # Net inputs
    phys = np.vstack([Bm, Ff, Ft, Ia, Ip, 0.0*Iu, rBt])
    phys = phys[:, tidxs]

    for i in range(nt):
        FxNg[:, i], _, _ = net_forward(phys[:, i], rg, zg)
        FxNq[:, i], _, _ = net_forward(phys[:, i], rq, zq)
        FxLq[:, i] = interp_pts(FxLg[:, :, i],  np.vstack([rq, zq]).T)

    # Stats
    print(f'Stats for shot {shot}:')
    FxLg_reshaped = FxLg.reshape(-1, nt)
    Fxg_abs_err = np.abs(FxLg_reshaped - FxNg)
    Fxq_abs_err = np.abs(FxLq - FxNq)
    Fx_range = np.vstack([FxLg_reshaped.min(axis=0), FxLg_reshaped.max(axis=0)])
    assert Fx_range.shape == (2, nt), 'Fx_range has wrong size'
    Fxg_perc_err = 100 * Fxg_abs_err / (Fx_range[1, :] - Fx_range[0, :])
    Fxq_perc_err = 100 * Fxq_abs_err / (Fx_range[1, :] - Fx_range[0, :])

    print(f'Avg range value: {Fx_range[1, :].mean():.4f}')
    print(
        f'Fx Error on grid: \n abs:  avg {Fxg_abs_err.mean():.4f}, std {Fxg_abs_err.std():.4f}, max {Fxg_abs_err.max():.4f} \n'
        f' perc: avg {Fxg_perc_err.mean():.2f}%, std {Fxg_perc_err.std():.2f}%, max {Fxg_perc_err.max():.2f}%'
    )
    print(
        f'Fx Error on control points: \n abs:  avg {Fxq_abs_err.mean():.4f}, std {Fxq_abs_err.std():.4f}, max {Fxq_abs_err.max():.4f} \n'
        f' perc: avg {Fxq_perc_err.mean():.2f}%, std {Fxq_perc_err.std():.2f}%, max {Fxq_perc_err.max():.2f}%'
    )

    # Plot (grid)
    import matplotlib.pyplot as plt
    grid_t_idx = 0
    plt.figure(figsize=(18, 10), num=f'Shot {shot} Fx Comparison')
    for row in range(2):
        for col in range(4):
            plt.subplot(2, 4, row * 4 + col + 1)
            if col == 0:
                data = FxLg_reshaped[:, grid_t_idx]
                title_str = 'FxLg (True)'
            elif col == 1:
                data = FxNg[:, grid_t_idx]
                title_str = 'FxNg (Net)'
            elif col == 2:
                data = Fxg_abs_err[:, grid_t_idx]
                title_str = 'Abs Error'
            elif col == 3:
                data = Fxg_perc_err[:, grid_t_idx]
                title_str = 'Perc Error (%)'
            if row == 0:
                plt.scatter(rg, zg, 30, data, cmap='viridis')
                plt.colorbar()
            else:
                plt.contourf(rg.reshape(65, 28), zg.reshape(65, 28), data.reshape(65, 28), cmap='viridis')
                plt.colorbar()
            plt.axis('equal')
            plt.title(title_str)
            plt.xlabel('R [m]')
            plt.ylabel('Z [m]')
    print(t.shape)
    plt.suptitle(f'Shot {shot}, t={t[0]:.3f} s\n{ONNX_NET_PATH}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot (control points)
    plt.figure(figsize=(18, 10), num=f'Shot {shot} Fx Control Points')
    for k in range(nq):
        plt.subplot(nq, 3, 3 * k + 1)
        plt.plot(t, FxLq[k, :], label='LIUQE')
        plt.plot(t, FxNq[k, :], label='Net')
        plt.legend()
        plt.title(f'Ctrl Pt {k+1}: Fx')
        plt.xlabel('Time [s]')
        plt.ylabel('Fx [Wb]')
        plt.grid(True)

        plt.subplot(nq, 3, 3 * k + 2)
        plt.plot(t, Fxq_abs_err[k, :])
        plt.title(f'Ctrl Pt {k+1}: Abs Error')
        plt.xlabel('Time [s]')
        plt.ylabel('Error [Wb]')
        plt.grid(True)

        plt.subplot(nq, 3, 3 * k + 3)
        plt.plot(t, Fxq_perc_err[k, :])
        plt.title(f'Ctrl Pt {k+1}: Error (%)')
        plt.xlabel('Time [s]')
        plt.ylabel('Error [%]')
        plt.grid(True)
    plt.suptitle(f'Shot {shot} Fx at Control Points\n{ONNX_NET_PATH}')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

plt.show()

