# Ideas

- 2 different optimizers for indepdendent nets, maybe 3 actually [ACTUALLY USELESS]
- train only one branch at a time to prove it works [GOOD]
- add pre and post resclaing layers [ONLY PRE is fine]
- somehow estimate xpoints when there are any 
- bring it further, estimate everything (every fuking output (normalized)) with a support network
  (single layer) [KIND OF DONE, working]
- to make batched position inference, just assemble a network with multiple copies of the trained
  one [DONE, maybe more elegant with convolutional layers]
- build a truly no-latency/no-overhead test of the inference speed, in C++ probably [WORKS]
- extend to prediction ahead in time -> need to recalculate a lot of equils
- single point prediction alongside grid prediction -> keep the physics section constant [DONE,
  works well]
- use real data (in contrast to fitted data) [DONE, slight performance decrease, not a big deal]
- multiple estimation from (real) thomson + interf + sxr 
- estimate not only the Flux map but also something else [DONE, Br, Bz, Iy]
- instead of multiplication in the latent space, use 1d convolution (this is probably equivalent to
  current batched version) or other operation (like exp, cos, sin) or mix of these, or try fourier
  encoding
- in dataset preparation, add a cloud of points around the LCFS
- create an animation on a single shot
- get full sequences, understand liuqe params
- test training in sequence + seq optimized net arch like RNN, LSTM, Transf, Mamba. (also useful for
  project with Rigoni)
- test with [%] error loss 
- train only Fx, Br, Bz just to be sure