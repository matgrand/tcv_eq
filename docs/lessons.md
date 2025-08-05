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

    to find the names of the variables, use jtraverser -> tcv_shot -> shot -> results -> EQUIL_1
    (liuqe.m) -> RESULTS 
    



## Integration into current shape controller

- to understand the points required by the shape controller:
![alt text](image.png)

- testing shots: "_puoi iniziare a guardare un po' di configurazioni diverse. Posso farti una ricerca nel logbook, ma devo essere in laboratorio perché per qualche motivo da fuori le query SQL non funzionano: Ad esempio potresti prendere un limiter (magari uno yo-yo, provo a cercarlo nel logbook), un sn (magari uno standard, ad esempio 79742), un double-null (ad esempio 86310), un NT (78893 a memoria dovrebbe andare), snowflakes, XPT..._"
  


## Hybrid
- to run shape controller (`shapectrl` for where the net is actually used), first `main_shape_ctrl`, then `test_closed_shape`

