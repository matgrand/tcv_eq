
try
    addpath([pwd '/onnx_net_forward']);

    NIN = 95;
    NLCFS = 129;

    %% test with simplified inputs
    % x = [3.0, 5.0];
    x = 1:NIN;

    % net_forward_mex([pwd 'onnx_net_forward/net.onnx']); % to load the model
    y = net_forward_mex(single(x));
    % print first 5 elements of x and y
    fprintf('x  -> [ %s ]\n', num2str(x(1:min(5,end)), '%+.4f '));
    fprintf('y  -> [ %s ]\n', num2str(y(1:min(5,end)), '%+.4f '));

    N = 100000;
    fprintf('Testing inference time for %d iterations...\n', N);
    % evaluate inference time in 10k iterations
    % warmup
    for i = 1:10
        x = rand(1, NIN);
        y = net_forward_mex(single(x));
    end
    times = zeros(1, N);
    ys = zeros(N, 2*NLCFS);
    ttot = tic;
    for i = 1:N
        x = rand(1, NIN);
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