# MEQT Helper Function Output Documentation

Code-specific output quantities are documented in code-specific files: see `FBTT`, `LIUT`, `FGET`, `FGST`.

---

## Dimensions

| Symbol | Description                                                         |
|:--------|:----------------------------------------------------------------------|
| `Q`    | Normalized sqrt poloidal flux including axis (`L.pQ`)               |
| `q`    | Normalized sqrt poloidal flux excluding axis (`L.pq`)               |
| `rx,zx`| Computational grid `L.rx, L.zx`                                     |
| `ry,zy`| Inner computational grid excluding boundary: `L.ry, L.zy`           |
| `o`    | Poloidal angle theta grid (`L.oq`)                                  |
| `g`    | Basis functions                                                     |
| `D`    | Plasma domain (>1 for e.g. doublets)                                |
| `W`    | Wall gaps                                                           |
| `nFW`  | Number of flux values being tracked                                 |
| `R`    | Rational 1/q surfaces                                               |
| `nR`   | Number of 1/q surfaces being tracked                                |
| `S`    | Flux surfaces with requested outboard r/a values                    |
| `nS`   | Number of tracked values                                            |
| `t`    | Time                                                                |

---

## Main Equilibrium and Integral Quantities

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.shot`  | Shot number | — | — |
| `.t`     | Time base | `(t)` | `[s]` |
| `.ag`    | Basis function coefficients | `(g,t)` | `[-]` |
| `.Bm`    | Simulated magnetic probe measurements | `(*,t)` | `[T]` |
| `.Um`    | Simulated magnetic field time derivative measurements | `(*,t)` | `[T/s]` |
| `.bp`    | Beta poloidal | `(t)` | `[-]` |
| `.bpli2` | Beta poloidal + li/2 | `(t)` | `[-]` |
| `.bt`    | Beta toroidal | `(t)` | `[-]` |
| `.Uf`    | Simulated flux loop poloidal flux | `(*,t)` | `[Wb]` |
| `.Uf`    | Simulated flux loop voltage measurements (=d/dt(Uf)) | `(*,t)` | `[V]` |
| `.FR`    | Ratio of normalized X-point/boundary flux | `(D,t)` | `[-]` |
| `.Ft0`   | Vacuum contribution to toroidal flux | `(t)` | `[Wb]` |
| `.Ft`    | Plasma contribution to toroidal flux | `(t)` | `[Wb]` |
| `.Fx`    | Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
| `.Ia`    | Fitted poloidal field coil currents | `(*,t)` | `[A]` |
| `.Ip`    | Total plasma current | `(t)` | `[A]` |
| `.Is`    | Vessel segment currents | `(*,t)` | `[A]` |
| `.Iu`    | Vessel currents in generalized description | `(*,t)` | `[A]` |
| `.Iv`    | Vessel filament currents | `(*,t)` | `[A]` |
| `.Iy`    | Plasma current distribution | `(ry,zy,t)` | `[A]` |
| `.Opy`   | Plasma domain index | `(ry,zy,t)` | `[-]` |
| `.li`    | Normalized internal inductance | `(t)` | `[-]` |
| `.mu`    | Normalized diamagnetism | `(t)` | `[-]` |
| `.nA`    | Number of magnetic axes found | `(t)` | `[-]` |
| `.PA`    | Pressure on axis | `(D,t)` | `[Pa]` |

---

## Domain Integrals

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.Al`    | Parallel inductance | `(t)` | `[H]` |
| `.Fpi`   | Parallel inductance contribution | `(t)` | `[H]` |
| `.Lpq`   | Flux loop contribution to inductance | `(t)` | `[H]` |
| `.Li`    | Total inductance (Lpq+Fpi+Al) | `(t)` | `[H]` |

---

## Flux Surface Integrals

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.Fq`    | Plasma flux on flux surfaces | `(rx,zx,t)` | `[Wb]` |
| `.Fr`    | Plasma flux on rational surfaces | `(rx,zx,t)` | `[Wb]` |
| `.Fb`    | Boundary flux (vacuum) | `(rx,zx,t)` | `[Wb]` |
| `.Ft`    | Total flux | `(rx,zx,t)` | `[Wb]` |
| `.FB`    | Flux surface | `(rx,zx,t)` | `[Wb]` |
| `.Fp`    | Flux surface on the outer layer | `(rx,zx,t)` | `[Wb]` |

---

## Shape Information

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.Rcx`   | Central radius of plasma shape | `(t)` | `[m]` |
| `.Zcx`   | Central Z-coordinate of plasma shape | `(t)` | `[m]` |
| `.Rpm`   | Plasma major radius | `(t)` | `[m]` |
| `.Rvm`   | Plasma vertical minor radius | `(t)` | `[m]` |
| `.Rbh`   | Plasma boundary height | `(t)` | `[m]` |
| `.Rp`    | Plasma shape radius | `(t)` | `[m]` |

---

## Magnetic Field Integrals

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.B`     | Magnetic field strength | `(rx,zx,t)` | `[T]` |
| `.Bz`    | Vertical magnetic field | `(rx,zx,t)` | `[T]` |
| `.Br`    | Radial magnetic field | `(rx,zx,t)` | `[T]` |
| `.Bt`    | Toroidal magnetic field | `(rx,zx,t)` | `[T]` |

---

## Coil Configuration

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.Aco`   | Coil area | `(rx,zx,t)` | `[m^2]` |
| `.Lco`   | Coil inductance | `(t)` | `[H]` |
| `.Rco`   | Coil radius | `(rx,zx,t)` | `[m]` |
| `.Ic`    | Coil current | `(t)` | `[A]` |

---

## Plasma Geometry

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.Rpf`   | Plasma profile function | `(rx,zx,t)` | `[m]` |
| `.Zpf`   | Plasma Z-profile function | `(rx,zx,t)` | `[m]` |
| `.Rpb`   | Plasma boundary position | `(rx,zx,t)` | `[m]` |
| `.Zpb`   | Plasma boundary Z-position | `(rx,zx,t)` | `[m]` |

---
good_ip
## Plasma Shape Computations

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.Cp`    | Plasma current profile | `(t)` | `[A]` |
| `.Co`    | Coil current profile | `(t)` | `[A]` |
| `.Xpp`   | Plasma shape position | `(rx,zx,t)` | `[m]` |
| `.Xpr`   | Plasma shape radial distance | `(rx,zx,t)` | `[m]` |

---

## Additional Quantities

| Variable | Description | Dimensions | Units |
|:------------|:--------------------------|:------------|:-------|
| `.Opx`   | O-point position | `(rx,zx,t)` | `[m]` |
| `.Vpl`   | Plasma velocity | `(rx,zx,t)` | `[m/s]` |
| `.Fpl`   | Plasma flux | `(rx,zx,t)` | `[Wb]` |
| `.Fbp`   | Boundary flux | `(rx,zx,t)` | `[Wb]` |

