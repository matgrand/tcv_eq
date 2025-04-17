# common definitions for the project

## plotting
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# Set default figure size and grid properties
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

## Shots and signals
SHOTS_FILE = 'data/good_shots.txt'
# signal names
IP='IP'
IPLIUQE='IPLIUQE'
IPREF='IPREF'
BT='BT'
SPLASMA='SPLASMA'
ZMAG='ZMAG'
# define the signals and their time vectors
SIGNALS = [IP,IPLIUQE,IPREF,BT,SPLASMA,ZMAG]
TIME_VECTORS = [f't_{s}' for s in SIGNALS]



