# Random lessons for working on TCV equlibria

More info in `docs/Getting_started.m`, `docs/Introduction.m`, `docs/Eleven_FBT_equilibria.m`

Codes:
- FBTE (`fbt`) - Fit desired equilibrium description
- LIUQE (`liu`) - Equilibrium reconstruction from measurements
- FGS (`fgs`) - Forward Grad-Shafranov Static
- FGE (`fge`) - Forward Grad-Shafranov Evolutive

Main data structures:
- `L`: General parameters structure
- `L.P`: User-set Parameters
- `L.G`: Geometry parameters
- `L.*`: Parameters computed from `L.P`, `L.G` by `[code]c.m`
- `LX`: Input data structure
- `LY`: Output data structure


