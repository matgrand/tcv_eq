clc; clear all; close all;

% Compile the C++ MEX function

%delete all files that start with 'net_forward.mex' (like rm net_forward.mex*)
delete('net_forward.mex*');

try
    %% Compile the C++ MEX function
    % mex('-v', ... % Verbose output, useful for debugging
    %     'CXXFLAGS="$CXXFLAGS -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=1 -pthread"', ... %abi 1 -> newer
    %     'LDFLAGS="$LDFLAGS -Wl,-rpath=/home/mg/libtorch/lib -pthread"', ...  % multi-threading
    %     ['-I', '/home/mg/libtorch/include'], ...
    %     ['-I', '/home/mg/libtorch/include/torch/csrc/api/include'], ...
    %     ['-L', '/home/mg/libtorch/lib'], ...
    %     '-ltorch', ...
    %     '-ltorch_cpu', ...
    %     '-lc10', ...
    %     'net_forward.cpp')
    % mex(... %'-v', ... % Verbose output, useful for debugging
    % 'CXXFLAGS="$CXXFLAGS -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0"', ... %abi 0 -> older
    % 'LDFLAGS="$LDFLAGS -Wl,-rpath=/home/mg/libtorch/lib"', ... % no multi-threading
    % ['-I', '/home/mg/libtorch/include'], ...
    % ['-I', '/home/mg/libtorch/include/torch/csrc/api/include'], ...
    % ['-L', '/home/mg/libtorch/lib'], ...
    % '-ltorch', ...
    % '-ltorch_cpu', ...
    % '-lc10', ...
    % 'net_forward.cpp')
    % mex('-v', ...
    %     'CXXFLAGS="\$CXXFLAGS -std=c++17 -fPIC"', ... 
    %     '-I/home/mg/libtorch/include', ...
    %     '-I/home/mg/libtorch/include/torch/csrc/api/include', ...
    %     '-L/home/mg/libtorch/lib', ...
    %     'LDFLAGS="\$LDFLAGS -Wl,-rpath,/home/mg/libtorch/lib -ltorch -ltorch_cpu -lc10"', ...
    %     'net_forward.cpp')
    mex('-v', ...
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

% input1 = rand(1, 95);
% input2 = rand(1, 16);
% input3 = rand(1, 16);
input1 = linspace(0, 10, 10);
input2 = linspace(0, 1, 10);
input3 = linspace(-1, 0, 10);
output = net_forward(input1, input2, input3);

% Display the inputs
disp('Input 1:');
disp(size(input1));
disp(input1);
disp('Input 2:');
disp(size(input2));
disp(input2);
disp('Input 3:');
disp(size(input3));
disp(input3);

% Display the output (first 4 rows and 4 columns)
disp('Output:');
disp(size(output));
disp(output(1:4, 1:4));