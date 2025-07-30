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

        phys = pack_inputs(d.Bm, d.Ff, d.Ft, d.Ia, d.Ip, d.Iu, d.rBt);
        % phys = pack_inputs(d.Bm, d.Ff, 0*d.Ft, d.Ia, d.Ip, 0*d.Iu, 0*d.rBt); % set Iu to 0 for testing
        fprintf('Packed inputs size: %s\n', mat2str(size(phys)));

        % control points
        rc = [0.7038, 0.6722, 0.6516, 0.7108, 0.9376, 1.0843, 1.0931, 0.9414, 0.8023, 0.6240];
        zc = [-0.1195, 0.1285, 0.3775, 0.6159, 0.6127, 0.4150, 0.1691, -0.0246, -0.7500, -0.1229];
        rq = single(rc);
        zq = single(zc);
        nq = length(rq); % number of control points
        fprintf('Control points: %d\n', nq);

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

        Fxg = zeros(65*28, nt); % preallocate Fx
        Brg = zeros(65*28, nt); % preallocate Br
        Bzg = zeros(65*28, nt); % preallocate Bz
        Fxc = zeros(nq, nt); % preallocate Fxc
        Brc = zeros(nq, nt); % preallocate Brc
        Bzc = zeros(nq, nt); % preallocate Bzc

        start = tic; % start timer
        for i = 1:nt % loop over time points -> network inferece
            [Fxg(:,i), Brg(:,i), Bzg(:,i)] = net_forward_mex(phys(:, i), r, z);
            [Fxc(:,i), Brc(:,i), Bzc(:,i)] = net_forward_mex(phys(:, i), rq, zq);
        end
        elapsed_time = toc(start); % measure elapsed time
        fprintf('Elapsed time for shot %d: %.2f seconds\n', shot_number, elapsed_time);

        % reshape results to match the grid size
        Fx = reshape(Fx, 65, 28, nt);
        Br = reshape(Br, 65, 28, nt);
        Bz = reshape(Bz, 65, 28, nt);

        assert(all(size(Fxg) == size(d.Fx)), 'Fx has wrong size');
        assert(all(size(Brg) == size(d.Br)), 'Br has wrong size');
        assert(all(size(Bzg) == size(d.Bz)), 'Bz has wrong size');

        % save results in a .mat file
        save_file = fullfile(save_dir, sprintf('%d_net.mat', shot_number));
        save(save_file, 'Fxg', 'Brg', 'Bzg', 'Fxc', 'Brc', 'Bzc');
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
