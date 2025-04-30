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



# Notes

- to possibly generate the dataset faster consider this section of `help meqt`:
    ```
    % Quantities on interpolation points (when using 'infct')
    .Fn    Flux at interpolation points L.P.rn, L.P.zn                 (*,t)     [Wb]
    .Un    Loop voltage at interpolation points                        (*,t)     [V]
    .Brn   Br at interpolation points                                  (*,t)     [T]
    .Bzn   Bz at interpolation points                                  (*,t)     [T]
    .Brrn  dBr/dr at interpolation points                              (*,t)     [T/m]
    .Brzn  dBr/dz at interpolation points                              (*,t)     [T/m]
    .Bzzn  dBz/dz at interpolation points                              (*,t)     [T/m]
    .Bzrn  dBz/dr at interpolation points                              (*,t)     [T/m]
    ```

- to access data using mdsplus do:
    
    ```
    ip = mdsdata('tcv_eq("I_PL", "LIUQE.M", "NOEVAL")');
    ```

    to find the names of the variables, use jtraverser -> tcv_shot -> shot -> EQUIL_1 (liuqe.m) -> RESULTS 