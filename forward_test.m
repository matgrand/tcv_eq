
try
    clc; clear; close all;
    PLOT = false;
    n_ctrl_pts = 24; % number of control points


    addpath([pwd '/onnx_net_forward']);
    model_path = [pwd '/onnx_net_forward/net.onnx'];

    % first call to load the model
    net_forward_mex(model_path);


    d = load('dss/demo.mat'); % load demo data (should be in DS_DIR/demo.mat , DS_DIR defined in utils.py)

    n_examples = size(d.phys, 1);
    max_n_pts = size(d.pts, 2);

    fprintf('Loaded demo data with %d examples and %d points.\n', n_examples, max_n_pts);

    NIN = 136;

    %% test with simplified inputs
    rand_i = randi([1, size(d.phys, 1)], 1, 1);
    phys = d.phys(rand_i, :); 
    r = d.pts(rand_i, :, 1);
    z = d.pts(rand_i, :, 2);

    fprintf('phys size: %d, r size: %d, z size: %d\n', size(phys, 2), size(r, 2), size(z, 2));

    % net_forward_mex([pwd 'onnx_net_forward/net.onnx']); % to load the model
    [Fx, Br, Bz] = net_forward_mex(single(phys), single(r), single(z));
    % print first 5 elements of results
    fprintf('fx_pred -> [ %s ]\n', num2str(Fx(1:min(5,end)), '%+.4f '));
    fprintf('fx_true -> [ %s ]\n', num2str(d.Fx(rand_i, 1:min(5,end)), '%+.4f '));
    fprintf('br_pred -> [ %s ]\n', num2str(Br(1:min(5,end)), '%+.4f '));
    fprintf('br_true -> [ %s ]\n', num2str(d.Br(rand_i, 1:min(5,end)), '%+.4f '));
    fprintf('bz_pred -> [ %s ]\n', num2str(Bz(1:min(5,end)), '%+.4f '));
    fprintf('bz_true -> [ %s ]\n', num2str(d.Bz(rand_i, 1:min(5,end)), '%+.4f '));

    % use timeit to measure inference time
    for n = [1, 2, 3, 4, 8, 16, 32, 64, 128, 256]
        phys = d.phys(rand_i, :); 
        r = d.pts(rand_i, 1:n, 1);
        z = d.pts(rand_i, 1:n, 2);
        t = timeit(@() net_forward_mex(single(phys), single(r), single(z)));
        fprintf('Inference time for %d control points: %.1f μs\n', n, t * 1e6);
    end


    % assert(false, 'This is a test, please remove this line to continue.');

    n_plots = 10;
    rand_idxs =  randi([1, size(d.phys, 1)], 1, n_plots);

    if PLOT
        fprintf('Plotting %d random examples...', n_plots);
        for i = 1:n_plots
            rand_i = rand_idxs(i);
            phys = d.phys(rand_i, :); 
            r = d.pts(rand_i, :, 1);
            z = d.pts(rand_i, :, 2);

            % net_forward_mex([pwd 'onnx_net_forward/net.onnx']); % to load the model
            [Fx, Br, Bz] = net_forward_mex(single(phys), single(r), single(z));

            preds = {Fx, Br, Bz};
            trues = {d.Fx(rand_i, :), d.Br(rand_i, :), d.Bz(rand_i, :)};
            labels = {'Fx', 'Br', 'Bz'};

            for k = 1:3
                ftrue = trues{k};
                fpred = preds{k};
                fmax = max([ftrue(:); fpred(:)]);
                fmin = min([ftrue(:); fpred(:)]);
                diffmap = 100 * abs(ftrue - fpred) ./ (fmax - fmin);
                sz = 20;
                fig = figure('Visible', 'off', 'Position', [100, 100, 1600, 800]);
                subplot(1,3,1);
                scatter(r, z, sz, ftrue, 'filled');
                title([labels{k} ' true']);
                xlabel('r'); ylabel('z');
                axis equal; colorbar;
                caxis([fmin fmax]);
                subplot(1,3,2);
                scatter(r, z, sz, fpred, 'filled');
                title([labels{k} ' pred']);
                xlabel('r'); ylabel('z');
                axis equal; colorbar;
                caxis([fmin fmax]);
                subplot(1,3,3);
                scatter(r, z, sz, diffmap, 'filled');
                title([labels{k} ' rel. error [%]']);
                xlabel('r'); ylabel('z');
                axis equal; colorbar;
                set(gcf, 'Name', labels{k});
                % save figure
                saveas(gcf, sprintf('test/matlab_%s_%s.png', lower(labels{k}), num2str(i)));
                % close figure
                close(gcf);
            end
        end
        fprintf(' Done.\n');
    end

    N = 30000;
    % n_ctrl_pts = 24; % number of control points
    fprintf('Testing inference time for %d iterations...', N);
    rand_idxs = randi([1, size(n_examples, 1)], N, 1);
    times = zeros(1, N);
    outs = zeros(N, n_ctrl_pts, 3);
    ttot = tic;
    for i = 1:N
        ri = rand_idxs(i);
        phys = d.phys(ri, :);
        r = d.pts(ri, 1:n_ctrl_pts, 1);
        z = d.pts(ri, 1:n_ctrl_pts, 2);
        t1 = tic;
        [fx, br, bz] = net_forward_mex(single(phys), single(r), single(z));
        times(i) = toc(t1);
        ys(i, :) = [fx, br, bz];
    end
    ttot = toc(ttot);
    fprintf(' Done.\n');
    % print times
    fprintf('Inference time -> %.1f ± %.1f [μs] | max %.1f [μs]\n', ...
        mean(times) * 1e6, std(times) * 1e6, max(times) * 1e6);
    fprintf('Total time -> %.1f [s]\n', ttot);
    fprintf('Frequency -> %.1f [μs] | %.1f [Hz]\n', 1e6*ttot/N, N/ttot);

    if PLOT
        % plot histogram
        figure('Visible', 'on', 'Position', [100, 100, 800, 600]);
        histogram(times * 1e6, 100);
        title('Inference time histogram');
        xlabel('Time [μs]');
        ylabel('Count');
        ylim([0 100]);
        grid on;
        % save figure
        saveas(gcf, 'test/matlab_inference_time.png');
        close(gcf);
    end

catch ME
    fprintf('Error: %s\n', ME.message);
    fprintf('Stack trace:\n');
    for i = 1:length(ME.stack)
        fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
end