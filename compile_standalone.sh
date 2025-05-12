g++ -std=c++17 -O2 -fPIC \
    -I/home/mg/libtorch/include \
    -I/home/mg/libtorch/include/torch/csrc/api/include \
    -L/home/mg/libtorch/lib \
    -o standalone_net_test standalone_net_test.cpp \
    -Wl,-rpath,/home/mg/libtorch/lib \
    -ltorch -ltorch_cpu -lc10

./standalone_net_test
