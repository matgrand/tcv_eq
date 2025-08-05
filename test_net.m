

%% init
clear all; close all; clc;

% Directory to save the output .mat files
OUT_DIR = 'test_shots'; % more space available

TIME_INTERV = [0.4, 0.9]; % time interval
DEC = 1; % decimation factor

% if ~exist(OUT_DIR, 'dir') mkdir(OUT_DIR); fprintf('Output directory created: %s\n', OUT_DIR);
% else delete(fullfile(OUT_DIR, '*')); fprintf('Output directory already exists. Old files deleted: %s\n', OUT_DIR);
% end % Create output directory if it doesn't exist

try
    mdsconnect('tcvdata.epfl.ch'); % Connect to the MDSplus server
catch 
end

shots = [
    % 79742 % single null
    % 86310 % double null
    % 78893 % negative triangularity
    % 83848 % ?
    78071 % standard, test ctrl pts (t=0.571) (warn: theta is wrong)
];

test_io_directly = true;

%% net stuff
% load the ONNX model
addpath([pwd '/onnx_net_forward']);
addpath(genpath([pwd '/data']));
% ONNX_NET_PATH = '/home/grandin/repos/liuqe-ml/data/3011842/net.onnx'; % seems best, no 0*Iu required
ONNX_NET_PATH = '/home/grandin/repos/liuqe-ml/data/3048577/net.onnx'; 
net_forward_mex(ONNX_NET_PATH); % first call to load the model

% dummy control points
nq = 5; % number of control points
thetaq = linspace(0,2*pi,nq+1); thetaq = thetaq(1:end-1)';
rq = 0.88 + 0.15*cos(thetaq);
zq = 0.20 + 0.45*sin(thetaq);

% % fixed control points
% rq = [0.7038, 0.6722, 0.6516, 0.7108, 0.9376, 1.0843, 1.0931, 0.9414, 0.8023, 0.6240];
% zq = [-0.1195, 0.1285, 0.3775, 0.6159, 0.6127, 0.4150, 0.1691, -0.0246, -0.7500, -0.1229];
% nq = length(rq); % number of control points

fprintf('Control points: %d\n', nq);

%load tcv grid
tcv_grid = load('tcv_params/grid.mat');
gr = tcv_grid.r; gz = tcv_grid.z;
[rg, zg] = meshgrid(gr, gz); % create meshgrid
rg = rg(:); zg = zg(:); % flatten 

%% loop over the shots
fprintf('Shots: %s\n', mat2str(shots));
fprintf('\nStarting tests...\n');

for si = 1:length(shots)
    % NOTE: for now, do only Fx. Br and Bz later
    % NOTE2: L = liuqe, N = net, g = grid, q = ctrl pts

    shot = shots(si);
    [t, Fx, Br, Bz, Bm, Ff, Ft, Ia, Ip, Iu, rBt] = load_shot_mg(shot, OUT_DIR); % load shot data

    tidxs = find(t >= TIME_INTERV(1) & t <= TIME_INTERV(2)); % find time indices
    tidxs = tidxs(1:DEC:end); % decimate the time samples
    nt = numel(tidxs); % number of time samples
    assert(nt >= 1, 'No time samples in the specified interval');
    fprintf('Time samples: %d\n', nt);

    % LIUQE/true values
    t = t(tidxs); % time vector
    FxLg = Fx(:,:,tidxs); % Fx on grid

    % preallocate
    FxNg = zeros(65*28, nt); % preallocate Fx on grid
    FxLq = zeros(nq, nt); % preallocate Fx on control points
    FxNq = zeros(nq, nt); % preallocate Fx on control points
    
    % run net inference + interpolate
    phys = [Bm; Ff; Ft; Ia; Ip; 0*Iu; rBt]; % net inputs
    phys = phys(:, tidxs);

    for i = 1:nt % loop over time points
        [FxNg(:, i), ~, ~] = net_forward(phys(:, i), rg, zg); % inference on grid
        [FxNq(:, i), ~, ~] = net_forward(phys(:, i), rq, zq); % inference on control points
        FxLq(:, i) = interp2(gr, gz, squeeze(FxLg(:,:,i)), rq, zq);
    end % end time loop
    
    % stats on the results
    fprintf('Stats for shot %d:\n', shot);
    FxLg = reshape(FxLg, [], nt); % reshape FxLg to [65*28, nt]
    Fxg_abs_err = abs(FxLg - FxNg); % absolute error
    Fxq_abs_err = abs(FxLq - FxNq); % absolute error on control points
    Fx_range = [min(FxLg, [], 1); max(FxLg, [], 1)]; % [2, nt]: min and max for each time step
    assert(all(size(Fx_range) == [2, nt]), 'Fx_range has wrong size');
    Fxg_perc_err = 100 * Fxg_abs_err ./ (Fx_range(2, :) - Fx_range(1, :)); % percentage error on grid
    Fxq_perc_err = 100 * Fxq_abs_err ./ (Fx_range(2, :) - Fx_range(1, :)); % percentage error on control points


    fprintf('Avg range value: %.4f', mean(Fx_range(2, :)));
    fprintf('Fx on grid: \n  abs: avg %.4f, std %.4f, max %.4f \n  perc: avg %.2f%%, std %.2f%%, max %.2f%%\n', ...
        mean(Fxg_abs_err(:)), std(Fxg_abs_err(:)), max(Fxg_abs_err(:)), ...
        mean(Fxg_perc_err(:)), std(Fxg_perc_err(:)), max(Fxg_perc_err(:)));
    fprintf('Fx on control points: \n  abs: avg %.4f, std %.4f, max %.4f \n  perc: avg %.2f%%, std %.2f%%, max %.2f%%\n', ...
        mean(Fxq_abs_err(:)), std(Fxq_abs_err(:)), max(Fxq_abs_err(:)), ...
        mean(Fxq_perc_err(:)), std(Fxq_perc_err(:)), max(Fxq_perc_err(:))); 

    % plot (grid)
    grid_t_idx = 1; % plot only the first time step for the grid
    figure('Name', sprintf('Shot %d Fx Comparison', shot), 'Position', [10, 100, 1800, 800]);
    for row = 1:2
        for col = 1:4
            subplot(2,4,(row-1)*4+col);
            switch col
                case 1
                    data = FxLg(:,grid_t_idx); title_str = 'FxLg (True)';
                case 2
                    data = FxNg(:,grid_t_idx); title_str = 'FxNg (Net)';
                case 3
                    data = Fxg_abs_err(:,grid_t_idx); title_str = 'Abs Error';
                case 4
                    data = Fxg_perc_err(:,grid_t_idx); title_str = 'Perc Error (%)';
            end
            if row == 1
                scatter(rg, zg, 30, data, 'filled');
                colorbar;
            else
                contourf(reshape(rg,65,28), reshape(zg,65,28), reshape(data,65,28));
                colorbar;
            end
            axis equal tight;
            title(title_str);
            xlabel('R [m]'); ylabel('Z [m]');
        end
    end
    sgtitle(sprintf('Shot %d, t=%.3f s', shot, t(1)));
    
    % plot (control points)
    figure('Name', sprintf('Shot %d Fx Control Points', shot), 'Position', [10, 100, 1800, 800]);
    for k = 1:nq
        subplot(nq,3,3*(k-1)+1);
        plot(t, FxLq(k,:), 'b-', 'LineWidth', 2); hold on;
        plot(t, FxNq(k,:), 'r--', 'LineWidth', 2);
        legend('LIUQE', 'Net');
        title(sprintf('Ctrl Pt %d: Fx', k));
        xlabel('Time [s]'); ylabel('Fx [Wb]');
        grid on;

        subplot(nq,3,3*(k-1)+2);
        plot(t, Fxq_abs_err(k,:), 'k-', 'LineWidth', 2);
        title(sprintf('Ctrl Pt %d: Abs Error', k));
        xlabel('Time [s]'); ylabel('Error [Wb]');
        grid on;

        subplot(nq,3,3*(k-1)+3);
        plot(t, Fxg_perc_err(k,:), 'k-', 'LineWidth', 2);
        title(sprintf('Ctrl Pt %d: Error (%)', k));
        xlabel('Time [s]'); ylabel('Error [%]');
        grid on;
    end
    sgtitle(sprintf('Shot %d Fx at Control Points', shot));

end % end shots loop

try 
    mdsdisconnect; % Disconnect from MDSplus
catch
end
fprintf('\nProcessing complete for all shots.\n');

if test_io_directly
    % generated on lac8
    net_input_log = load('test_shots/net_input_log.mat').net_input_log;
    net_output_log = load('test_shots/net_output_log.mat').net_output_log;
    phys = net_input_log(1:136, :);      % (136, nt)
    ri   = net_input_log(137:161, :);    % (25, nt)
    zi   = net_input_log(162:186, :);    % (25, nt)
    nt = size(phys, 2); % number of time samples
    net_output = zeros(25, 3, nt); % preallocate output (25 control points, 3 outputs, nt time samples)
    for i = 1:nt-1
        [Fx, Br, Bz] = net_forward(phys(:, i), ri(:, i), zi(:, i)); % inference on control points
        Fx_lac8 = net_output_log(:,1,i)'; % output from lac8
        Br_lac8 = net_output_log(:,2,i)'; % output from lac8
        Bz_lac8 = net_output_log(:,3,i)'; % output from lac8
        net_output(:, 1, i) = Fx; % store Fx
        net_output(:, 2, i) = Br; % store Br
        net_output(:, 3, i) = Bz; % store Bz
        eFx = max(abs(Fx - Fx_lac8));
        eBr = max(abs(Br - Br_lac8));
        eBz = max(abs(Bz - Bz_lac8));
        if eFx >= 1e-6, fprintf('Fx error too high at time %d: %.8f\n', i, eFx); break; end
        if eBr >= 1e-6, fprintf('Br error too high at time %d: %.8f\n', i, eBr); break; end
        if eBz >= 1e-6, fprintf('Bz error too high at time %d: %.8f\n', i, eBz); break; end
    end
    % print average errors +. std
    overall_errors = abs(net_output(:, :, :) - net_output_log(:, :, :)); % overall errors
    fprintf('Average error: %.5e, std: %.5e\n', ...
        mean(overall_errors(:)), std(overall_errors(:)));
    fprintf('Test finished.\n');
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% FUNCTIONS

% load shot
function [t, Fx, Br, Bz, Bm, Ff, Ft, Ia, Ip, Iu, rBt] = load_shot_mg(shot, OUT_DIR)
    save_file = fullfile(OUT_DIR, sprintf('%d_cache.mat', shot));

    if exist(save_file, 'file') % use cached data if available
        load(save_file, 't', 'Fx', 'Iy', 'Br', 'Bz', 'Bm', 'Ff', 'Ft', 'Ia', 'Ip', 'Iu', 'rBt');
        fprintf('Loaded cached data for shot %d from: %s\n', shot, save_file);
    else
        MIN_TIME_SAMPLES = 10; % Minimum number of time samples to keep the shot
        MAX_IP_PERC_DIFF = 2.5; % Maximum percentage difference between IPLIUQE and IP
        mdsopen('tcv_shot', shot); % Open the MDSplus connection to the TCV database

        %% Load liuqe data
        [L, LY] = mds2meq(shot, 'LIUQE.M'); % get liuqe outputs from mdsplus
        [L, LX] = liuqe(shot, LY.t); % get liuqe inputs 
        
        t = LY.t'; % time vector
        [t2, ip2] = tcvget('IP', t); % calculated using magnetics at liuqe times

        % analyze the time vector
        assert(max(abs(t2 - t)) < 1e-8, 'Time vectors do not coincide');
        assert(numel(t) > 1, sprintf('Time vector has insufficient elements: t:%s', mat2str(size(t))));
        assert(max(abs(t - t2)) < 1e-8, 'Times do not coincide');
        nt = numel(t); % number of time samples
        
        % calculate magnetic fields (copied from meqpost)
        i4pirxdzx = 1./(4*pi*L.dzx*L.rx');
        i4pirxdrx = 1./(4*pi*L.drx*L.rx');
        [Brx,Bzx] = meqBrBz(LY.Fx,i4pirxdzx,i4pirxdrx,L.nzx,L.nrx);

        %% extract quantities
        % Ouputs
        Fx = LY.Fx; % Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
        Iy = LY.Iy; % Plasma current density map | `(ry,zy,t)` | `[A/m^2]` |
        Br = Brx;
        Bz = Bzx;

        % Inputs
        Bm = LX.Bm; 
        Ff = LX.Ff;
        Ft = LX.Ft;
        Ia = LX.Ia; 
        Ip = LX.Ip;
        Iu = LX.Iu;
        rBt = LX.rBt; 

        % check the dimensions
        assert(all(size(Fx) == [65, 28, nt]), 'Fx has wrong size');
        assert(all(size(Iy) == [63, 26, nt]), 'Iy has wrong size');
        assert(all(size(Br) == [65, 28, nt]), 'Brx has wrong size');
        assert(all(size(Bz) == [65, 28, nt]), 'Bzx has wrong size');
        assert(all(size(Bm) == [38, nt]), 'Bm has wrong size');
        assert(all(size(Ff) == [38, nt]), 'Ff has wrong size');
        assert(all(size(Ft) == [1, nt]), 'Ft has wrong size');
        assert(all(size(Ia) == [19, nt]), 'Ia has wrong size');
        assert(all(size(Ip) == [1, nt]), 'Ip has wrong size');
        assert(all(size(Iu) == [38, nt]), 'Iu has wrong size');
        assert(all(size(rBt) == [1, nt]), 'rBt has wrong size');
        mdsclose; % Close the MDSplus connection

        % save data into a .mat file
        save(save_file, 't', 'Fx', 'Iy', 'Br', 'Bz', ...
            'Bm', 'Ff', 'Ft', 'Ia', 'Ip', 'Iu', 'rBt');
        fprintf('Saved data for shot %d to: %s\n', shot, save_file);
    end

end % load_shot_mg

% network inference
function [Fx, Br, Bz] = net_forward(phys, r, z)
    [Fx, Br, Bz] = net_forward_mex(single(phys), single(r), single(z));
    Fx = double(Fx); Br = double(Br); Bz = double(Bz); % convert to double
end

% calc Br, z, copied from meqpost
function [Br,Bz] = meqBrBz(Fx,i4pirdz,i4pirdr,nz,nr)
    % [Br,Bz] = meqBrBz(Fx,i4pirdz,i4pirdr,nz,nr)
    % Compute Br,Bz fields
    % General version that also accepts time-varying Fx

    [Br,Bz] = deal(zeros(nz,nr,size(Fx,3))); % init
    % Br = -1/(2*pi*R)* dF/dz
    % Central differences dF/dz[i] =  F[i-1] - F[i+1]/(2*dz)
    Br(2:end-1,:,:) = -i4pirdz.* (Fx(3:end,:,:) - Fx(1:end-2,:,:));
    % At grid boundary i, use: dF/dz[i] = (-F(i+2) + 4*F(i+1) - 3*F(i))/(2*dz)
    Br(end,:  ,:) = -i4pirdz          .* (+Fx(end-2,:,:) - 4*Fx(end-1,:,:) + 3*Fx(end,:,:));
    Br(1  ,:  ,:) = -i4pirdz          .* (-Fx(    3,:,:) + 4*Fx(    2,:,:) - 3*Fx(  1,:,:));

    % Bz = 1/(2*pi*R)* dF/dr
    Bz(:,2:end-1,:) =  i4pirdr(2:end-1) .* (+Fx(:,  3:end,:) - Fx(:,1:end-2,:));
    % Same as for Br
    Bz(:,end    ,:) =  i4pirdr(end)     .* (+Fx(:,end-2,:) - 4*Fx(:,end-1,:) + 3*Fx(:,end,:));
    Bz(:,1      ,:) =  i4pirdr(1)       .* (-Fx(:,    3,:) + 4*Fx(:,    2,:) - 3*Fx(:,  1,:));
end    
