function run_net_on_shot(shot_number, save_dir)
    fprintf('Running network on shot %d, saving in %s\n', shot_number, save_dir);
    try
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
        
        nt = size(phys,2); % number of time points
        
        % points of the LCFS 
%         rq = single(d.rq);
%         zq = single(d.zq);
%         nq = size(rq, 1); % number of points on the LCFS

        % dummy interpolation points
        nq = 5; % number of control points
        thetaq = linspace(0,2*pi,nq+1); thetaq = thetaq(1:end-1)';
        rq = single(0.88 + 0.15*cos(thetaq));
        zq = single(0.20 + 0.45*sin(thetaq));

%         % fixed control points
%         rc = [0.7038, 0.6722, 0.6516, 0.7108, 0.9376, 1.0843, 1.0931, 0.9414, 0.8023, 0.6240];
%         zc = [-0.1195, 0.1285, 0.3775, 0.6159, 0.6127, 0.4150, 0.1691, -0.0246, -0.7500, -0.1229];
%         rq = single(rc);
%         zq = single(zc);
%         nq = length(rq); % number of control points
        fprintf('Control points: %d\n', nq);

        %load tcv grid
        g = load('tcv_params/grid.mat');
        gr = g.r;
        gz = g.z;

        % create a meshgrid from r and z
        [r, z] = meshgrid(gr, gz);
        %flatten r and z
        r = single(r(:));
        z = single(z(:));

        FxgN = zeros(65*28, nt); % preallocate Fx
        BrgN = zeros(65*28, nt); % preallocate Br
        BzgN = zeros(65*28, nt); % preallocate Bz
        FxqN = zeros(nq, nt); % preallocate Fxq
        BrqN = zeros(nq, nt); % preallocate Brq
        BzqN = zeros(nq, nt); % preallocate Bzq        
        FxqL = zeros(nq, nt); % preallocate Fxq Liuqe interpolated
        BrqL = zeros(nq, nt); % preallocate Brq
        BzqL = zeros(nq, nt); % preallocate Bzq

        start = tic; % start timer
        for i = 1:nt % loop over time points
            [FxgN(:,i), BrgN(:,i), BzgN(:,i)] = net_forward_mex(phys(:, i), r, z);
            [FxqN(:,i), BrqN(:,i), BzqN(:,i)] = net_forward_mex(phys(:, i), rq, zq);
            % fprintf('Time %d: avg(norm(Fxq)) = %.4f, avg(norm(Brq)) = %.4f, avg(norm(Bzq)) = %.4f\n', i, avg_norm_Fxq, avg_norm_Brq, avg_norm_Bzq);
            
%             drx = gr(2)-gr(1); dzx = gz(2) - gz(1);
%             inp.n = 9; qpM = qintc(inp,drx,dzx); % qintmex consolidation
%            [FxqL(:,i), BrqL(:,i), BzqL(:,i)] = qintmex(gr,gz,squeeze(d.Fx(:,:,i)),double(rq),double(zq),qpM);
           FxqL(:,i) = interp2(gr,gz,squeeze(d.Fx(:,:,i)),double(rq),double(zq));
           BrqL(:,i) = interp2(gr,gz,squeeze(d.Br(:,:,i)),double(rq),double(zq));
           BzqL(:,i) = interp2(gr,gz,squeeze(d.Bz(:,:,i)),double(rq),double(zq));
        end
        elapsed_time = toc(start); % measure elapsed time
        fprintf('Elapsed time for shot %d: %.2f seconds\n', shot_number, elapsed_time);

        % reshape results to match the grid size
        FxgN = reshape(FxgN, 65, 28, nt);
        BrgN = reshape(BrgN, 65, 28, nt);
        BzgN = reshape(BzgN, 65, 28, nt);

        assert(all(size(FxgN) == size(d.Fx)), 'Fx has wrong size');
        assert(all(size(BrgN) == size(d.Br)), 'Br has wrong size');
        assert(all(size(BzgN) == size(d.Bz)), 'Bz has wrong size');

        % save results in a .mat file
        save_file = fullfile(save_dir, sprintf('%d_net.mat', shot_number));
        save(save_file, 'FxgN', 'BrgN', 'BzgN', 'FxqN', 'BrqN', 'BzqN', 'FxqL', 'BrqL', 'BzqL');
        fprintf('Saved results to: %s\n', save_file);

%         % plot
%         for i = 1 : 50 : nt
%           bar([Fxq(:,i), FxqL(:,i)])
%           legend('net','true')
%           pause(1e-2)
%         end
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
