# Ideas

- build a truly no-latency/no-overhead test of the inference speed, in C++ probably
- extend to prediction ahead in time
- single point prediction alongside grid prediction -> keep the physics section constant
- use real data (in contrast to fitted data)
- multiple estimation from (real) thomson + interf + sxr
- estimate not only the Flux map but also something else
- instead of multiplication in the latent space, use 1d convolution (this is probably equivalent to
  current batched version) or other operation (like exp, cos, sin) or mix of these.