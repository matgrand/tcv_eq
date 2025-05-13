clc; clear all; close all;

%% Compile the C++ MEX function
%delete all files that start with 'net_forward.mex' (like rm net_forward.mex*)
delete('net_forward.mex*');
try
    mex('', ... % -v for verbose output
        'CXXFLAGS="\$CXXFLAGS -std=c++17 -fPIC -O2"', ... % Added -O2 optimization flag
        '-I/home/mg/libtorch/include', ...
        '-I/home/mg/libtorch/include/torch/csrc/api/include', ...
        '-L/home/mg/libtorch/lib', ...
        'LDFLAGS="\$LDFLAGS -Wl,-rpath,/home/mg/libtorch/lib"', ... % Rpath ensures finding libs at runtime
        '-ltorch', ...          % Link against torch library
        '-ltorch_cpu', ...      % Link against torch_cpu library
        '-lc10', ...           % Link against c10 library (core tensor library)
        'net_forward.cpp')
catch ME
    % If compilation fails, display the error message
    fprintf('Compilation failed: %s\n', ME.message);
end

%% test with real inputs, NOTE: they are float, not double 
load("test/test_inference.mat") % xs, ys, rs, zs
fprintf('Shapes: xs:[%s], ys:[%s], rs:[%s], zs:[%s]\n', ...
    num2str(size(xs)), num2str(size(ys)), num2str(size(rs)), num2str(size(zs)));
x = double(xs(1,:)); 
r = double(rs(1,:));
z = double(zs(1,:));
y = net_forward(x, r, z);

fprintf('Shapes: x:[%s], r:[%s], z:[%s], y:[%s]\n', ...
    num2str(size(x)), num2str(size(r)), num2str(size(z)), num2str(size(y)));

% print first n elements of each vector
n = 5;
fprintf('x -> [ %s ]\n', num2str(x(1:n), '%+.4f '));
fprintf('y -> [ %s ]\n', num2str(y(1:n), '%+.4f '));
fprintf('r -> [ %s ]\n', num2str(r(1:n), '%+.4f '));
fprintf('z -> [ %s ]\n', num2str(z(1:n), '%+.4f '));


