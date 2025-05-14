#include "mex.h"
#include <dlfcn.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    dlopen("/home/mg/libtorch/lib/libtorch_cpu.so", RTLD_NOW | RTLD_GLOBAL);
    dlopen("/home/mg/libtorch/lib/libtorch.so", RTLD_NOW | RTLD_GLOBAL);
    dlopen("/home/mg/libtorch/lib/libc10.so", RTLD_NOW | RTLD_GLOBAL);
    mexPrintf("LibTorch libraries loaded\n");
}
