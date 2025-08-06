# Ideas

- 2 different optimizers for indepdendent nets, maybe 3 actually [ACTUALLY USELESS]
- train only one branch at a time to prove it works [GOOD]
- add pre and post resclaing layers [ONLY PRE is fine]
- somehow estimate xpoints when there are any 
- bring it further, estimate everything (every output (normalized)) with a support network
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
- test with [%] error loss [DONE, working, similar performance wrt MSE]
- train only Fx, Br, Bz just to be sure


## discussion with Adriano & Francesco ideas 
- use matlab r2019a
- estimate the gaps, more for the paper
- grow rate vertical stability (ask about the routine)
- map the (flux maps?) with PCA to get a uniform distribution in the training
- training in pre-shot? o finetuning
- similarity search degli spari con gli embeddings?
  
### Notes on tests with Cosmas
- Test with Cosmas controller
- 9 modes, and 5 modes, 5 better
- shot 87044
- 1000ms starting from 1 -> 1.1 [s]
- Net is vertically stable, nice, goes into a slight limiter configuration (with 9)
- one mode is not going fully to 0
- the response of the netowork to the controller might not be correct, it's not reacting to
  controller changes
- worse than lih
- improve lih with a better current distributions estimate
- with 5 modes is better -> broad shape is "fine", some mistakes in the finer details. -> I think
  it's a problem of accuracy
- estimate the response of the plasma to a variaiton in the coils current, even as a loss.
- see influence of slight position variations (expect linear behaviour) 
- [ ] investigate the my_pc/lac8 value difference. 
- BIG - since the control points dont change during a shot, u dont need to recalculate them

## More ideas
- [ ] test a train with only the most important physics inputs
- [ ] Analyze interpolation errors, specifically close to the grid edges (the padding?) (use finer grid?)
- [ ] (related to previous one) split the netowork in 2? 1 for inner zone, and 1 for the one closer to the grid boundary (or closer to the sensors?)
- Do I need a cleaner dataset to reach those precisions/accuracies?
- [v] Test a training with absolute error (not MSE) [WORKS]
- [ ] Test SoftPlus activation


### Dataset mods
- [ ] make the r,z borders wider (limit of the grid)
- [ ] Br, Bz calculated from finer grid