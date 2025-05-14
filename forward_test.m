
% set LD_PRELOAD for libtorch and to execute the mex function
libpath = fullfile(pwd, 'libtorch', 'lib');
preload = strjoin({ ...
    fullfile(libpath, 'libtorch_cpu.so'), ...
    fullfile(libpath, 'libtorch.so'), ...
    fullfile(libpath, 'libc10.so') ...
}, ':');
setenv('LD_PRELOAD', preload);




%% test with simplified inputs
x = [3.0, 5.0];

tic
y = net_forward(x);
toc

% print x and y
fprintf('x -> [ %s ]\n', num2str(x, '%+.4f '));
fprintf('y -> [ %s ]\n', num2str(y, '%+.4f '));
