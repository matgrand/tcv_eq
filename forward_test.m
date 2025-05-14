%% test with simplified inputsa
x = [1.0, 2.0];

tic
y = net_forward(x);
toc

% print x and y
fprintf('x -> [ %s ]\n', num2str(x, '%+.4f '));
fprintf('y -> [ %s ]\n', num2str(y, '%+.4f '));
