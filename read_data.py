import numpy as np
from scipy.io import loadmat
from definitions import *

from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
from tqdm import tqdm

# read the shots
# shots = [int(shot.strip()) for shot in open(SHOTS_FILE).read().split(',') if shot.strip()]
# get all the files in in the directory
shots = [int(f.split('.')[0]) for f in os.listdir(DS_DIR) if f.endswith('.mat')]

# read the grid coordinates
d = loadmat('tcv_params/grid.mat')
r, z = d['r'].flatten(), d['z'].flatten()  # grid coordinates
rr, zz = np.meshgrid(r, z)  # meshgrid for plotting
rr1, zz1 = np.meshgrid(r[1:-1], z[1:-1])  # meshgrid for plotting

# keep only the first 3 shots
# shots = shots[:2]

print(f'Shots: {shots}')

for shot in shots:
    print(f'Animating shot: {shot}')

    d = loadmat(f'{DS_DIR}/{shot}.mat')

    t, Ip = d['t'].flatten(), d['Ip'].flatten()  # time and plasma current
    sip = np.sign(np.mean(Ip)) # sign of the plasma current
    Fx = d['Fx']  # flux map
    Iy = d['Iy']  # current density map
    Ia = d['Ia']  # coil currents
    Bm = d['Bm']  # magnetic probe measurements
    Uf = d['Uf']  # flux loop poloidal flux

    # Create figure and subplots
    fig = plt.figure(figsize=(8, 10))
    gs = gridspec.GridSpec(3, 2, figure=fig)

    # Static plot for plasma current
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(t, Ip*sip)
    ax1.set_title(f'{"Neg" if sip<0 else "Pos"} Plasma Current vs Time')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Ip')
    time_line = ax1.axvline(x=t[0], color='r')

    # Other subplots for animation
    # Coil currents
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title('Coil Currents (Ia)[A]')
    ax2.set_xlabel('Coil Index')
    ax2.set_ylabel('Current')
    ax2.set_ylim(np.nanmax(Ia), np.nanmin(Ia))
    bars_Ia = ax2.bar(range(Ia.shape[0]), Ia[:, 0])

    # Heatmaps for flux map and current density 
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_title(f'{"Neg" if sip<0 else "Pos"} Flux Map (Fx)[Wb]')
    extent = [r[0], r[-1], z[0], z[-1]]
    heatmap_Fx = ax3.imshow(Fx[:, :, 0]*sip, origin='lower', extent=extent)
    heatmap_Fx.set_clim(np.nanmin(Fx*sip), np.nanmax(Fx*sip))
    ax3.set_aspect('equal')
    plt.colorbar(heatmap_Fx, ax=ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_title(f'{"Neg" if sip<0 else "Pos"} Curr Dens (Iy)[A/m^2]')
    extent = [r[1], r[-2], z[1], z[-2]]  # Adjust extent for Iy
    heatmap_Iy = ax4.imshow(Iy[:, :, 0]*sip, origin='lower', extent=extent)
    ax4.set_aspect('equal')
    heatmap_Iy.set_clim(np.nanmin(Iy*sip), np.nanmax(Iy*sip))
    plt.colorbar(heatmap_Iy, ax=ax4)

    ax5 = fig.add_subplot(gs[2, 0])
    ax5.set_title('Magnetic Probes (Bm)[T]')
    ax5.set_ylabel('Measurement')
    ax5.set_ylim(np.nanmax(Bm), np.nanmin(Bm))
    bars_Bm = ax5.bar(range(Bm.shape[0]), Bm[:, 0])

    ax6 = fig.add_subplot(gs[2, 1])
    ax6.set_title('Flux Loop Voltage (Uf)[V]')
    ax6.set_ylabel('Flux')
    ax6.set_ylim(np.nanmax(Uf), np.nanmin(Uf))
    bars_Ff = ax6.bar(range(Uf.shape[0]), Uf[:, 0])

    plt.tight_layout()

    # Create animation
    frames = min(200, len(t))  # Use up to 200 frames or the available time points
    progress_bar = tqdm(total=frames, desc=f'Animating {shot}')

    # Animation update function
    def update(frame):
        # Update vertical time line
        time_line.set_xdata([t[frame], t[frame]])
        # Update bar plots
        for i, bar in enumerate(bars_Ia): bar.set_height(Ia[i, frame])
        for i, bar in enumerate(bars_Bm): bar.set_height(Bm[i, frame])
        for i, bar in enumerate(bars_Ff): bar.set_height(Uf[i, frame])
        # Update heatmaps
        heatmap_Fx.set_array(Fx[:, :, frame]*sip)
        heatmap_Iy.set_array(Iy[:, :, frame]*sip)
        heatmap_Fx.autoscale()
        heatmap_Iy.autoscale()

        # Update titles with time information
        ax1.set_title(f'Plasma Current vs Time (t={t[frame]:.4f}s)')
        # Update progress bar
        progress_bar.update(1)
        if frame == frames - 1: progress_bar.close()
        return [time_line, heatmap_Fx, heatmap_Iy] + list(bars_Ia) + list(bars_Bm) + list(bars_Ff)

    ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

    # Save animation (optional)
    ani.save(f'figs/anim_{shot}.gif', fps=10)

    # plt.show()
    plt.close(fig)  # Close the figure to avoid displaying it in non-interactive environments