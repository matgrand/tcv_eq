clc; clear all; close all;

% NOTE: for now only way is do this before running matlab:
% export LD_PRELOAD="/home/mg/libtorch/lib/libtorch_cpu.so:/home/mg/libtorch/lib/libtorch.so:/home/mg/libtorch/lib/libc10.so"
% export LD_PRELOAD="$(pwd)/libtorch/lib/libtorch_cpu.so:$(pwd)/libtorch/lib/libtorch.so:$(pwd)/libtorch/lib/libc10.so"

%% Compile the C++ MEX function
delete('net_forward.mex*');
try
    libtorch_path = [pwd '/libtorch']; 
    mex('', ...
        'CXXFLAGS=$CXXFLAGS -std=c++17 -fPIC -O2', ...
        ['-I' libtorch_path '/include'], ...
        ['-I' libtorch_path '/include/torch/csrc/api/include'], ...
        ['-L' libtorch_path '/lib'], ...
        ['LDFLAGS=$LDFLAGS -Wl,-rpath,' libtorch_path '/lib'], ...
        '-ltorch', ...
        '-ltorch_cpu', ...
        '-lc10', ...
        'net_forward.cpp');
catch
    % If the compilation fails, print the error message
    fprintf('Compilation failed: %s\n', lasterr);
end

%% test with simplified inputs
x = [3.0, 5.0];
y = net_forward(x);

% print x and y
fprintf('x -> [ %s ]\n', num2str(x, '%+.4f '));
fprintf('y -> [ %s ]\n', num2str(y, '%+.4f '));