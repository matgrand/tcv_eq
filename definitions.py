# common definitions for the project

## plotting
import matplotlib.pyplot as plt
plt.style.use('dark_background')
# Set default figure size and grid properties
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['image.cmap'] = 'inferno'

import os
if not os.path.exists('figs'): os.makedirs('figs')

## Shots and signals
SHOTS_FILE = 'good_shots.txt'
DS_DIR = 'ds'
# DS_DIR = '/NoTivoli/grandin/ds'

# signal names
T  = 't'  # Time vector | `(t)` | `[s]` |
IP = 'Ip' # Plasma current | `(t)` | `[A]` |
FX = 'Fx' # Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
IY = 'Iy' # Plasma current density map | `(ry,zy,t)` | `[A/m^2]` |
IA = 'Ia' # Fitted poloidal field coil currents | `(*,t)` | `[A]` |
BM = 'Bm' # Simulated magnetic probe measurements | `(*,t)` | `[T]` |
UF = 'Uf' # Simulated flux loop voltage measurements (=d/dt(Ff)) | `(*,t)` | `[V]` |

# TCV parameters


