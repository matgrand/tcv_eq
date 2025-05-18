clear all; close all; clc;

% TODO consider adding vessel currents: I_VESSEL

START_SHOT = 77662; % Dec 2022, https://spcwiki.epfl.ch/wiki/Alma_database
END_SHOT = 85804; % April 2025
N_SHOTS = 30; % Number of shots to process
% N_SHOTS = END_SHOT-START_SHOT; % Number of shots to process

% Directory to save the output .mat files
% OUT_DIR = 'ds'; % testing
OUT_DIR = '/NoTivoli/grandin/ds'; % more space available

% DECIMATION = 6; % Decimation factor for the time vector
DECIMATION = 60; % Decimation factor for the time vector

MIN_TIME_SAMPLES = 10; % Minimum number of time samples to keep the shot
MAX_IP_PERC_DIFF = 2.5; % Maximum percentage difference between IPLIUQE and IP

fprintf('Parameters:\n');
fprintf('\tSTART_SHOT: %d\n', START_SHOT);
fprintf('\tEND_SHOT: %d\n', END_SHOT);
fprintf('\tN_SHOTS: %d\n', N_SHOTS);
fprintf('\tOUT_DIR: %s\n', OUT_DIR);
fprintf('\tDECIMATION: %d\n', DECIMATION);
fprintf('\tMIN_TIME_SAMPLES: %d\n', MIN_TIME_SAMPLES);
fprintf('\tMAX_IP_PERC_DIFF: %.2f\n', MAX_IP_PERC_DIFF);
fprintf('\n');

fprintf('Deleting old figures...\n');
delete(fullfile('figs', 'ip_*.png')); % Delete old figures

if ~exist(OUT_DIR, 'dir') mkdir(OUT_DIR); fprintf('Output directory created: %s\n', OUT_DIR);
else delete(fullfile(OUT_DIR, '*')); fprintf('Output directory already exists. Old files deleted: %s\n', OUT_DIR);
end % Create output directory if it doesn't exist

mdsconnect('tcvdata.epfl.ch'); % Connect to the MDSplus server

% get a reference for theta, to make sure they are all the same
mdsopen('tcv_shot', END_SHOT);
theta0 = mdsdata('tcv_eq("THETA", "LIUQE.M", "NOEVAL")'); 
mdsclose; % Close the MDSplus connection


shots = randi([START_SHOT, END_SHOT], 1, N_SHOTS);
total_shots = 0;
fprintf('Shots: %s\n', mat2str(shots));
fprintf('\nStarting data retrieval loop...\n');
for i = 1:length(shots)
    start_time = tic; % Start timer for processing each shot

    shot = shots(i);
    fprintf('\x1b[33mProcessing shot %d (%d of %d)\x1b[0m\n', shot, i, length(shots));

    try 
        mdsopen('tcv_shot', shot); % Open the MDSplus connection to the TCV database

        [t, ip1] = tcvget('IPLIUQE'); % precalculated using liuqe
        ip1_mds = mdsdata('tcv_eq("I_PL", "LIUQE.M", "NOEVAL")');

        assert(~isempty(t), 'No time vector found');
        assert(numel(t) > 1, sprintf('Time vector has insufficient elements: t:%s', mat2str(size(t))));
        
        % fprintf('\tIPLIUQE size: %s, IPLIUQE (MDS) size: %s\n', mat2str(size(ip1)), mat2str(size(ip1_mds)));
        assert(all(size(ip1) == size(ip1_mds)), 'IPLIUQE and IPLIUQE (MDS) have different sizes');
        assert(max(abs(ip1 - ip1_mds)) < 1e-8, 'IPLIUQE and IPLIUQE (MDS) are different');

        [t2, ip2] = tcvget('IP', t); % calculated using magnetics at liuqe times
        
        % analyze the time vector
        t_diff = abs(t - t2);
        fprintf('\ttime steps -> n: %d, mean: %.2f [µs], std: %.2f [µs]\n', numel(t), mean(diff(t) * 1e6), std(diff(t) * 1e6));
        % fprintf('\t# of time samples: %d\n', numel(t));
        % fprintf('\tTime difference: mean: %.2e [s], max: %.2e [s]\n', mean(t_diff), max(t_diff));
        assert(max(t_diff) < 1e-8, 'Times do not coincide');
        
        avg_ip = mean(abs(ip2));
        ip_diff = abs(ip1 - ip2);
        avg_diff = mean(ip_diff);
        perc_diff = ip_diff ./ abs(ip2) * 100;
        fprintf('\tip average difference -> %.2f [A] (%.1f%%)\n', avg_diff, mean(perc_diff));
        
        % keep only the samples where IP and IPLIUQE are similar
        % assert(mean(perc_diff) < MAX_IP_PcERC_DIFF, 'Difference between IPLIUQE and IP is too high'); % very strict
        ip_valid1 = perc_diff < MAX_IP_PERC_DIFF;
        fprintf('\tip filtered -> %.1f%%, remaining -> %d/%d \n', 100*(1-sum(ip_valid1)/numel(t)), sum(ip_valid1), numel(t));
        assert(sum(ip_valid1) > 0.8 * numel(t), 'IP MEAS and IPLIUQE are different in too many samples');

        % %% [RECALCULATE] liuqe equilibrium at the good plasma current times 
        % % NOTE: change the x_valid section (the times are not matched here)
        % [L,LX,LY] = liuqe(shot, t(ip_valid1));
        % Fx = LY.Fx; % Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
        % Iy = LY.Iy; % Plasma current density map | `(ry,zy,t)` | `[A/m^2]` |
        % Ia = LY.Ia; % Fitted poloidal field coil currents | `(*,t)` | `[A]` |
        % Bm = LY.Bm; % Simulated magnetic probe measurements | `(*,t)` | `[T]` |
        % Uf = LY.Uf; % Simulated flux loop poloidal flux | `(*,t)` | `[Wb]` |
        % Ip = ip1; % Plasma current | `(*,t)` | `[A]` |

        %% [LOAD PRECALC] liuqe equilibrium at the plasma current times
        Fx = mdsdata('tcv_eq("PSI", "LIUQE.M", "NOEVAL")');         % Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
        Iy = mdsdata('tcv_eq("J_TOR", "LIUQE.M", "NOEVAL")');       % Plasma current density map | `(ry,zy,t)` | `[A/m^2]` |
        Ia = mdsdata('tcv_eq("I_POL", "LIUQE.M", "NOEVAL")');       % Fitted poloidal field coil currents | `(*,t)` | `[A]` |
        Bm = mdsdata('tcv_eq("B_PROBE", "LIUQE.M", "NOEVAL")');     % Simulated magnetic probe measurements | `(*,t)` | `[T]` |
        Uf = mdsdata('tcv_eq("PSI_LOOP", "LIUQE.M", "NOEVAL")');    % Simulated flux loop poloidal flux | `(*,t)` | `[Wb]` |
        Ip = mdsdata('tcv_eq("I_PL", "LIUQE.M", "NOEVAL")');        % Plasma current | `(*,t)` | `[A]` |
        
        % last closed flux surface (LCFS) 
        rq = mdsdata('tcv_eq("R_EDGE", "LIUQE.M", "NOEVAL")'); % LCFS r coordinate
        zq = mdsdata('tcv_eq("Z_EDGE", "LIUQE.M", "NOEVAL")'); % LCFS z coordinate
        theta = mdsdata('tcv_eq("THETA", "LIUQE.M", "NOEVAL")'); 

        % check that theta is the same as theta0, first size, then values
        assert(all(size(theta) == size(theta0)), 'Theta and theta0 have different sizes');
        assert(all(abs(theta(:) - theta0(:)) < 1e-5), 'Theta and theta0 are not close enough');

        % check the time dimensions are the same
        fprintf('\tsizes -> Fx:%s, Iy:%s, Ia:%s, Bm:%s, Uf:%s, t:%s, Ip:%s\n', mat2str(size(Fx)), mat2str(size(Iy)), mat2str(size(Ia)), mat2str(size(Bm)), mat2str(size(Uf)), mat2str(size(t)),  mat2str(size(Ip)));
        assert(size(Fx, 3) == numel(t), 'Fx has wrong time dimension');
        assert(size(Iy, 3) == numel(t), 'Iy has wrong time dimension');
        assert(size(Ia, 2) == numel(t), 'Ia has wrong time dimension');
        assert(size(Bm, 2) == numel(t), 'Bm has wrong time dimension');
        assert(size(Uf, 2) == numel(t), 'Uf has wrong time dimension');
        assert(size(Ip, 1) == numel(t), 'Ip has wrong time dimension');
        
        % filter out the NaN/Inf values [MILD]
        Fx_valid  = reshape(all(all(~isnan(Fx) & ~isinf(Fx),1),2), [],1);
        Iy_valid  = reshape(all(all(~isnan(Iy) & ~isinf(Iy),1),2), [],1);
        Ia_valid  = reshape(all(~isnan(Ia) & ~isinf(Ia), 1), [],1);
        Bm_valid  = reshape(all(~isnan(Bm) & ~isinf(Bm), 1), [],1);
        Uf_valid  = reshape(all(~isnan(Uf) & ~isinf(Uf), 1), [],1);
        t_valid   = ~isnan(t) & ~isinf(t);
        ip_valid2 = ~isnan(Ip) & ~isinf(Ip);
        % fprintf('\t*_valid sizes -> Fx %s, Iy %s, Ia %s, Bm %s, Uf %s, t %s, ip2: %s, ip1 %s\n', mat2str(size(Fx_valid)), mat2str(size(Iy_valid)), mat2str(size(Ia_valid)), mat2str(size(Bm_valid)), mat2str(size(Uf_valid)), mat2str(size(t_valid)), mat2str(size(ip_valid2)), mat2str(size(ip_valid1)));
        valid = Fx_valid & Iy_valid & Ia_valid & Bm_valid & Uf_valid & t_valid & ip_valid2 & ip_valid1;
        fprintf('\tvalid samples -> [TOT: %.1f%%, %d] Fx:%.1f%%, Iy:%.1f%%, Ia:%.1f%%, Bm:%.1f%%, Uf:%.1f%%, t:%.1f%%, ip2:%.1f%%, ip1:%.1f%%\n', ...
                100*sum(valid)/numel(valid), sum(valid), 100*sum(Fx_valid)/numel(Fx_valid), 100*sum(Iy_valid)/numel(Iy_valid), 100*sum(Ia_valid)/numel(Ia_valid), ...
                100*sum(Bm_valid)/numel(Bm_valid), 100*sum(Uf_valid)/numel(Uf_valid), 100*sum(t_valid)/numel(t_valid), 100*sum(ip_valid2)/numel(ip_valid2), 100*sum(ip_valid1)/numel(ip_valid1));

        % fprintf('\tvalid samples -> %.1f%%, remaining -> %d/%d \n', 100*(1-sum(valid)/numel(t)), sum(valid), numel(t));
        assert(sum(valid) > 0.5 * numel(t), 'Nan/Inf filter -> not enough valid samples');
                
        Fx = Fx(:,:,valid);
        Iy = Iy(:,:,valid);
        Ia = Ia(:,valid);
        Bm = Bm(:,valid);
        Uf = Uf(:,valid);
        t = t(valid);
        Ip = Ip(valid);

        % decimate
        Fx = Fx(:,:,1:DECIMATION:end);
        Iy = Iy(:,:,1:DECIMATION:end);
        Ia = Ia(:,1:DECIMATION:end);
        Bm = Bm(:,1:DECIMATION:end);
        Uf = Uf(:,1:DECIMATION:end);
        t = t(1:DECIMATION:end);
        Ip = Ip(1:DECIMATION:end);

        % check none of the variables contains NaN 
        assert(~any(isnan(Fx(:))), 'Fx contains NaN values');
        assert(~any(isnan(Iy(:))), 'Iy contains NaN values');
        assert(~any(isnan(Ia(:))), 'Ia contains NaN values');
        assert(~any(isnan(Bm(:))), 'Bm contains NaN values'); 
        assert(~any(isnan(Uf(:))), 'Uf contains NaN values');
        assert(~any(isnan(t(:))), 't contains NaN values');
        assert(~any(isnan(Ip(:))), 'Ip contains NaN values');
        
        % check none of the variables contains Inf
        assert(~any(isinf(Fx(:))), 'Fx contains Inf values');
        assert(~any(isinf(Iy(:))), 'Iy contains Inf values');
        assert(~any(isinf(Ia(:))), 'Ia contains Inf values');
        assert(~any(isinf(Bm(:))), 'Bm contains Inf values');
        assert(~any(isinf(Uf(:))), 'Uf contains Inf values');
        assert(~any(isinf(t(:))), 't contains Inf values');
        assert(~any(isinf(Ip(:))), 'Ip contains Inf values');
        
        % convert to single precision to save space
        Fx = single(Fx); % Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
        Iy = single(Iy); % Plasma current density map | `(ry,zy,t)` | `[A/m^2]` |
        Ia = single(Ia); % Fitted poloidal field coil currents | `(*,t)` | `[A]` |
        Bm = single(Bm); % Simulated magnetic probe measurements | `(*,t)` | `[T]` |
        Uf = single(Uf); % Simulated flux loop poloidal flux | `(*,t)` | `[Wb]` |
        t = single(t);
        Ip = single(Ip);

        % print final sizes
        fprintf('\tFinal sizes -> Fx:%s, Iy:%s, Ia:%s, Bm:%s, Uf:%s, t:%s, Ip:%s\n', mat2str(size(Fx)), mat2str(size(Iy)), mat2str(size(Ia)), mat2str(size(Bm)), mat2str(size(Uf)), mat2str(size(t)), mat2str(size(Ip)));

        mdsclose; % Close the MDSplus connection
        total_shots = total_shots + 1;

        % save data into a .mat file
        save_file = fullfile(OUT_DIR, sprintf('%d.mat', shot));
        save(save_file, 't', 'Ip', 'Fx', 'Iy', 'Ia', 'Bm', 'Uf');
        fprintf('\x1b[32m\tData saved to: %s\x1b[0m\n', save_file);
        
        shot_processing_times(i) = toc(start_time);
        fprintf('\tProc time: %.2f s, ETA: %.0f min\n', shot_processing_times(i), sum(shot_processing_times) / i * (length(shots) - i) / 60);
        
    catch ME
        fprintf('\x1b[31m\tError processing shot %d: %s\x1b[0m\n', shot, ME.message);
        continue; % Skip to the next shot on error
    end % try-catch
end % end shots loop

mdsdisconnect; % Disconnect from MDSplus

fprintf('\nProcessing complete for all shots.\n');
fprintf('Output files saved in: %s\n', OUT_DIR);

% save a random mat in the /NoTivoli/grandin/ directory to show it is finished
save(fullfile('/NoTivoli/grandin/', 'ds_done.mat'), 'shot_processing_times');