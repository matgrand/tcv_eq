
try
    addpath([pwd '/onnx_net_forward']);

    d = load('dss/demo.mat'); % load demo data (should be in DS_DIR/demo.mat , DS_DIR defined in utils.py)

    NIN = 95;
    NLCFS = 129;

    %% test with simplified inputs
    % x = [3.0, 5.0];
    % x = 1:95;

    rand_i = randi([1, size(d.X, 1)], 1, 1);

    x = d.X(rand_i, :); % 1st row of demo data

    % net_forward_mex([pwd 'onnx_net_forward/net.onnx']); % to load the model
    y = net_forward_mex(single(x));

    y_true = d.Y(rand_i, :); % 1st row of demo data

    % plot y and y_true on a 2d plot
    figure;
    plot(y(1:NLCFS), y(NLCFS+1:end), 'r', 'LineWidth', 2);
    hold on;
    plot(y_true(1:NLCFS), y_true(NLCFS+1:end), 'b', 'LineWidth', 2);
    title('y and y_true');
    xlabel('x');
    ylabel('y');
    legend('y', 'y\_true');
    grid on;
    axis equal;
    % save figure
    saveas(gcf, 'test/matlab_inference.png');

    % print first 5 elements of x and y
    fprintf('x      -> [ %s ]\n', num2str(x(1:min(5,end)), '%+.4f '));
    fprintf('y      -> [ %s ]\n', num2str(y(1:min(5,end)), '%+.4f '));
    fprintf('y_true -> [ %s ]\n', num2str(y_true(1:min(5,end)), '%+.4f '));

    N = 100000;
    fprintf('Testing inference time for %d iterations...\n', N);
    % evaluate inference time in 10k iterations
    % warmup + plotting
    for i = 1:10
        x = d.X(i, :); 
        y_true = d.Y(i, :); 
        y = net_forward_mex(single(x));

        % plot y and y_true on a 2d plot
        figure;
        plot(y(1:NLCFS), y(NLCFS+1:end), 'r', 'LineWidth', 2);
        hold on;
        plot(y_true(1:NLCFS), y_true(NLCFS+1:end), 'b', 'LineWidth', 2);
        title('y and y_true');
        xlabel('x');
        ylabel('y');
        legend('y', 'y\_true');
        grid on;
        axis equal;
        % save figure
        saveas(gcf, sprintf('test/matlab_inference_%d.png', i));

    end
    times = zeros(1, N);
    ys = zeros(N, 2*NLCFS);
    ttot = tic;
    for i = 1:N
        x = d.X(randi([1, size(d.X, 1)], 1, 1), :); % randomize input
        t1 = tic;
        y = net_forward_mex(single(x));
        times(i) = toc(t1);
        ys(i, :) = y;
    end
    ttot = toc(ttot);
    % print y
    fprintf('y -> [ %s ]\n', num2str(y(1:min(5,end)), '%+.4f '));
    % print times
    fprintf('Inference time -> %.1f ± %.1f [μs] | max %.1f [μs]\n', ...
        mean(times) * 1e6, std(times) * 1e6, max(times) * 1e6);
    fprintf('Total time -> %.1f [s]\n', ttot);
    fprintf('Frequency -> %.1f [μs] | %.1f [Hz]\n', 1e6*ttot/N, N/ttot);

    % plot histogram
    figure;
    histogram(times * 1e6, 100);
    title('Inference time histogram');
    xlabel('Time [μs]');
    ylabel('Count');
    ylim([0 100]);
    grid on;
    % save figure
    saveas(gcf, 'test/matlab_inference_time.png');


catch ME
    fprintf('Error: %s\n', ME.message);
end