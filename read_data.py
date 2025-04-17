
import numpy as np
from scipy.io import loadmat
from definitions import *


# read the shots
shots = [int(shot.strip()) for shot in open(SHOTS_FILE).read().split(',') if shot.strip()]

# keep only the first 3 shots
shots = shots[:100]

print(f"Shots: {shots}")

# read the shots
for shot in shots:
    print(f"Shot: {shot}")
    d = loadmat(f'data/{shot}.mat') # load the data
    ss, ts = {}, {}

    for sn, tn in zip(SIGNALS, TIME_VECTORS):
        # get the signal and time vector
        s = d[sn].flatten()
        t = d[tn].flatten()
        print(f'{sn}: {s.shape}, t:{t.shape}')
        ss[sn], ts[sn] = s, t

    # plot the signals
    # plasma currents
    plt.figure()
    plt.plot(ts[IP], ss[IP], label=IP)
    plt.plot(ts[IPLIUQE], ss[IPLIUQE], label=IPLIUQE)
    plt.plot(ts[IPREF], ss[IPREF], label=IPREF)
    plt.xlabel('Time [s]')
    plt.ylabel('Current [A]')
    plt.legend()
    plt.title(f'Shot {shot} - Plasma Currents')
    plt.savefig(f'figs/IP_{shot}.svg')

