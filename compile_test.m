clc; clear all; close all;
for i = 1:1000 fprintf('\n'); end

%% Compile the C++ MEX function
delete('net_forward.mex*');
try
    mex('', ... % -v for verbose output
        'CXXFLAGS="\$CXXFLAGS -std=c++17 -fPIC -O2"', ... % Added -O2 optimization flag
        '-I/home/mg/libtorch/include', ...
        '-I/home/mg/libtorch/include/torch/csrc/api/include', ...
        '-L/home/mg/libtorch/lib', ...
        'LDFLAGS="\$LDFLAGS -Wl,-rpath,/hccome/mg/libtorch/lib"', ... % Rpath ensures finding libs at runtime
        '-ltorch', ...          % Link against torch library
        '-ltorch_cpu', ...      % Link against torch_cpu library
        '-lc10', ...           % Link against c10 library (core tensor library)
        'net_forward.cpp')
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
