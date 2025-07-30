function run_net_on_shot(shot_number, save_dir)
    fprintf('Running network on shot %d, saving in %s\n', shot_number, save_dir);
    try
        n_ctrl_pts = 25; % number of control points

        addpath([pwd '/onnx_net_forward']);
        model_path = [pwd '/onnx_net_forward/net.onnx'];

        % first call to load the model
        net_forward_mex(model_path);


        % load shot
        shot_file_path = fullfile('test_shots', sprintf('%d.mat', shot_number));

        d = load(shot_file_path);
        fprintf('Loaded shot data from: %s\n', shot_file_path);

        % phys = pack_inputs(d.Bm, d.Ff, d.Ft, d.Ia, d.Ip, d.Iu, d.rBt);
        phys = pack_inputs(d.Bm, d.Ff, 0*d.Ft, d.Ia, d.Ip, 0*d.Iu, 0*d.rBt); % set Iu to 0 for testing
        fprintf('Packed inputs size: %s\n', mat2str(size(phys)));

        % points of the LCFS 
        rq = single(d.rq);
        zq = single(d.zq);
        nq = size(rq, 1); % number of points on the LCFS
        % fprintf('LCFS points: %d\n', nq);

        nt = size(phys,2); % number of time points

        %load tcv grid
        g = load('tcv_params/grid.mat');
        gr = g.r;
        gz = g.z;

        % create a meshgrid from r and z
        [r, z] = meshgrid(gr, gz);
        %flatten r and z
        r = single(r(:));
        z = single(z(:));

        Fx = zeros(65*28, nt); % preallocate Fx
        Br = zeros(65*28, nt); % preallocate Br
        Bz = zeros(65*28, nt); % preallocate Bz
        Fxq = zeros(nq, nt); % preallocate Fxq
        Brq = zeros(nq, nt); % preallocate Brq
        Bzq = zeros(nq, nt); % preallocate Bzq

        start = tic; % start timer
        for i = 1:nt % loop over time points
            [Fx(:,i), Br(:,i), Bz(:,i)] = net_forward_mex(phys(:, i), r, z);
            [Fxq(:,i), Brq(:,i), Bzq(:,i)] = net_forward_mex(phys(:, i), rq(:, i), zq(:, i));
            % avg_norm_Fxq = mean(vecnorm(Fxq(:,i)));
            % avg_norm_Brq = mean(vecnorm(Brq(:,i)));
            % avg_norm_Bzq = mean(vecnorm(Bzq(:,i)));
            % fprintf('Time %d: avg(norm(Fxq)) = %.4f, avg(norm(Brq)) = %.4f, avg(norm(Bzq)) = %.4f\n', i, avg_norm_Fxq, avg_norm_Brq, avg_norm_Bzq);
        end
        elapsed_time = toc(start); % measure elapsed time
        fprintf('Elapsed time for shot %d: %.2f seconds\n', shot_number, elapsed_time);

        % reshape results to match the grid size
        Fx = reshape(Fx, 65, 28, nt);
        Br = reshape(Br, 65, 28, nt);
        Bz = reshape(Bz, 65, 28, nt);

        assert(all(size(Fx) == size(d.Fx)), 'Fx has wrong size');
        assert(all(size(Br) == size(d.Br)), 'Br has wrong size');
        assert(all(size(Bz) == size(d.Bz)), 'Bz has wrong size');
        

        % save results in a .mat file
        save_file = fullfile(save_dir, sprintf('%d_net.mat', shot_number));
        save(save_file, 'Fx', 'Br', 'Bz', 'Fxq', 'Brq', 'Bzq');
        fprintf('Saved results to: %s\n', save_file);

    catch ME
        fprintf('Error: %s\n', ME.message);
        fprintf('Stack trace:\n');
        for i = 1:length(ME.stack)
            fprintf('  %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
        end
    end

    function inputs = pack_inputs(Bm, Ff, Ft, Ia, Ip, Iu, rBt)
        % Pack inputs into a single array 
        % NOTE: the order is very important here, it should match the order in which the network was trained, see utils.py -> INPUT_NAMES
        inputs = [Bm; Ff; Ft; Ia; Ip; Iu; rBt];
        % Ensure inputs are in single precision
        inputs = single(inputs);
    end
end
