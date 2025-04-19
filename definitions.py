# common definitions for the project

## plotting
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# Set default figure size and grid properties
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'monospace'

import os
if not os.path.exists('figs'): os.makedirs('figs')

## Shots and signals
SHOTS_FILE = 'good_shots.txt'
DS_DIR = 'ds'
# DS_DIR = '/NoTivoli/grandin/ds'

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



