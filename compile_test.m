clc; clear all; close all;
for i = 1:1000 fprintf('\n'); end

%% Compile the C++ MEX function
delete('net_forward.mex*');
try
    libtorch_path = '/home/mg/libtorch';
    mex('', ...
        'CXXFLAGS=$CXXFLAGS -std=c++17 -fPIC -O2', ...
        ['-I' libtorch_path '/include'], ...
        ['-I' libtorch_path '/include/torch/csrc/api/include'], ...
        ['-L' libtorch_path '/lib'], ...
        'LDFLAGS=$LDFLAGS -Wl,-rpath,/home/mg/libtorch/lib', ...
        '-ltorch', ...
        '-ltorch_cpu', ...
        '-lc10', ...
        'net_forward.cpp');
catch ME
    % If compilation fails, display the error message
    fprintf('Compilation failed: %s\n', ME.message);
end

%% test with simplified inputsa
x = [1.0, 2.0];
y = net_forward(x);

% print x and y
fprintf('x -> [ %s ]\n', num2str(x, '%+.4f '));
fprintf('y -> [ %s ]\n', num2str(y, '%+.4f '));

% NOTE: do this before:
% export LD_PRELOAD="/home/mg/libtorch/lib/libtorch_cpu.so:/home/mg/libtorch/lib/libtorch.so:/home/mg/libtorch/lib/libc10.so"