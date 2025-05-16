
try
    addpath([pwd '/onnx_net_forward']);

    %% test with simplified inputs
    x = [3.0, 5.0];

    y = net_forward_mex(x);

    % print x and y
    fprintf('x -> [ %s ]\n', num2str(x, '%+.4f '));
    fprintf('y -> [ %s ]\n', num2str(y, '%+.4f '));

    N = 100000;
    fprintf('Testing inference time for %d iterations...\n', N);
    % evaluate inference time in 10k iterations
    % warmup
    for i = 1:10
        x = rand(1, 2);
        y = net_forward_mex(x);
    end
    times = zeros(1, N);
    for i = 1:N
        x = rand(1, 2);
        tic
        y = net_forward_mex(x);
        times(i) = toc;
    end
    fprintf('Inference time -> %.1f ± %.1f [μs] | max %.1f [μs]\n', ...
        mean(times) * 1e6, std(times) * 1e6, max(times) * 1e6);

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