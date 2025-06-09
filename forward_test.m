
try

    %% current best speeds:
    % 1k -> 365 μs (fx, br, bz, iy grouped, dynamic axes)
    % 1k -> 247 μs (fx, br, bz, iy grouped, static axes)
    % 64 -> 44 μs (fx, br, bz, iy grouped, static axes)
    % 64 -> 45 μs (fx, br, bz, iy grouped, dynamic axes)
    % new architecture: (fx, br, bz grouped)
    % 25 -> 23 / 49+-22 / tot 63 dynamic 
    % 25 -> 20 / 20+-10 / tot 27 static [much faster]
    clc; clear; close all;
    PLOT = false;
    % PLOT = true; % set to true to plot results
    n_ctrl_pts = 25; % number of control points
    N = 10000; % number of iterations for inference time test


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
    % rand_i = randi([1, size(d.phys, 1)], 1, 1);
    rand_i = 1;
    phys = d.phys(rand_i, :); 
    r = d.pts(rand_i, 1:n_ctrl_pts, 1);
    z = d.pts(rand_i, 1:n_ctrl_pts, 2);

    fprintf('phys size: %d, r size: %d, z size: %d\n', size(phys, 2), size(r, 2), size(z, 2));

    [Fx, Br, Bz] = net_forward_mex(single(phys), single(r), single(z));
    % print first 5 elements of results
    fprintf('fx_pred -> [ %s ]\n', num2str(Fx(1:min(8,end)), '%+.4f '));
    fprintf('fx_true -> [ %s ]\n', num2str(d.Fx(rand_i, 1:min(8,end)), '%+.4f '));
    fprintf('br_pred -> [ %s ]\n', num2str(Br(1:min(8,end)), '%+.4f '));
    fprintf('br_true -> [ %s ]\n', num2str(d.Br(rand_i, 1:min(8,end)), '%+.4f '));
    fprintf('bz_pred -> [ %s ]\n', num2str(Bz(1:min(8,end)), '%+.4f '));
    fprintf('bz_true -> [ %s ]\n', num2str(d.Bz(rand_i, 1:min(8,end)), '%+.4f '));

    % use timeit to measure inference time
    fprintf('Measuring inference time...\n');
    for n = [1, 10, 20, 25, 30, 100, 128, 256, 300, 400, 500, 1000]
        if n >= 100
            n_iter = round(N/30);
        else
            n_iter = N;
        end

        phys = d.phys(rand_i, :); 
        r = d.pts(rand_i, 1:n, 1);
        z = d.pts(rand_i, 1:n, 2);
        try
            t1 = timeit(@() net_forward_mex(single(phys), single(r), single(z)));
        catch ME
            t1 = NaN; % in case of error, set time to NaN
        end
        % fprintf('1 sample,      %d control points: %.1f μs\n', n, t * 1e6);
        rand_idxs = randi([1, size(d.phys, 1)], n_iter, 1);
        vphys = d.phys(rand_idxs, :);
        vr = d.pts(rand_idxs, 1:n, 1);
        vz = d.pts(rand_idxs, 1:n, 2);
        try 
            t2 = timeit(@() test_speed(vphys, vr, vz, n));
        catch ME
            t2 = NaN; % in case of error, set time to NaN
        end
        % fprintf('%d samples, %d control points: %.1f μs per sample\n', N, n, t * 1e6 / N);
        fprintf('%d pts -> [%.1f μs (1)] [%.1f μs (%d)]\n', ...
            n, t1 * 1e6, t2 * 1e6 / n_iter, n_iter);
    end


    % assert(false, 'This is a test, please remove this line to continue.');

    n_plots = 20;
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

    % n_ctrl_pts = 24; % number of control points
    fprintf('Testing inference time for %d iterations...', N);
    rand_idxs = randi([1, size(n_examples, 1)], N, 1);
    times = zeros(1, N);
    outs = zeros(N, n_ctrl_pts, 3);
    ttot = tic;
    for i = 1:N
        ri = rand_idxs(i);
        t1 = tic;
        [fx, br, bz] = net_forward_mex(single(d.phys(ri, :)), single(d.pts(ri, 1:n_ctrl_pts, 1)), single(d.pts(ri, 1:n_ctrl_pts, 2)));
        times(i) = toc(t1);
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

function t = test_speed(vphys, vr, vz, ncp)
    start = tic;
    n = size(vphys, 1);
    assert(size(vr,1) == n && size(vz,1) == n, ...
        'Input vectors must have the same number of rows.');
    for i = 1:n
        [fx, br, bz] = net_forward_mex(single(vphys(i,:)), single(vr(i,1:ncp)), single(vz(i,1:ncp)));
    end
    t = toc(start);
end