clear all; close all; clc;

% TODO consider adding vessel currents: I_VESSEL


% Directory to save the output .mat files
OUT_DIR = 'test_shots'; % more space available

DECIMATION = 1; % 6 Decimation factor for the time vector

MIN_TIME_SAMPLES = 10; % Minimum number of time samples to keep the shot
MAX_IP_PERC_DIFF = 2.5; % Maximum percentage difference between IPLIUQE and IP


if ~exist(OUT_DIR, 'dir') mkdir(OUT_DIR); fprintf('Output directory created: %s\n', OUT_DIR);
else delete(fullfile(OUT_DIR, '*')); fprintf('Output directory already exists. Old files deleted: %s\n', OUT_DIR);
end % Create output directory if it doesn't exist

mdsconnect('tcvdata.epfl.ch'); % Connect to the MDSplus server

% get a reference for theta, to make sure they are all the same
% some old shots have theta in [-pi, pi], the new ones in [0, 2*pi], but more versions are possible
mdsopen('tcv_shot', 85804);
theta0 = mdsdata('tcv_eq("THETA", "LIUQE.M", "NOEVAL")'); 
mdsclose; % Close the MDSplus connection

shots = [
    79742, % single null
    86310, % double null
    78893, % negative triangularity
    83848, % ?
];
disp('Shots to process:');
disp(shots);
total_shots = 0;
fprintf('Shots: %s\n', mat2str(shots));
fprintf('\nStarting data retrieval loop...\n');
for i = 1:length(shots)
    start_time = tic; % Start timer for processing each shot

    shot = shots(i);
    fprintf('\x1b[33mProcessing shot %d (%d of %d)\x1b[0m\n', shot, i, length(shots));

    try 
        mdsopen('tcv_shot', shot); % Open the MDSplus connection to the TCV database

        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Load liuqe data
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        [L, LY] = mds2meq(shot, 'LIUQE.M'); % get liuqe outputs from mdsplus
        [L, LX] = liuqe(shot, LY.t); % get liuqe inputs 
        
        t = LY.t'; % time vector
        [t2, ip2] = tcvget('IP', t); % calculated using magnetics at liuqe times

        % last closed flux surface (LCFS) (unfortunately it's not in mds2meq outputs (yet))
        rq = mdsdata('tcv_eq("R_EDGE", "LIUQE.M", "NOEVAL")'); % LCFS r coordinate
        zq = mdsdata('tcv_eq("Z_EDGE", "LIUQE.M", "NOEVAL")'); % LCFS z coordinate
        theta = mdsdata('tcv_eq("THETA", "LIUQE.M", "NOEVAL")'); 
        % check that theta is the same as theta0, first size, then values
	    assert(all(size(theta) == size(theta0)), 'theta and theta0 have different sizes');
        assert(all(abs(theta0 - theta) < 1e-6), 'theta and theta0 are different');

        % analyze the time vector
        assert(max(abs(t2 - t)) < 1e-8, 'Time vectors do not coincide');
	    assert(numel(t) > 1, sprintf('Time vector has insufficient elements: t:%s', mat2str(size(t))));
        nt = numel(t); % number of time samples
        t_diff = abs(t - t2);
        fprintf('\ttime steps -> n: %d, mean: %.2f [µs], std: %.2f [µs]\n', numel(t), mean(diff(t) * 1e6), std(diff(t) * 1e6));
        assert(max(t_diff) < 1e-8, 'Times do not coincide');
        
        % analyze the plasma current
        %ip1 = LY.Ip'; % IPLIUQE
	%    assert(all(size(ip1) == size(ip2)), 'Ip sizes wrong');
        %avg_ip = mean(abs(ip2));
        %ip_diff = abs(ip1 - ip2);
        %avg_diff = mean(ip_diff);
        %perc_diff = ip_diff ./ abs(ip2) * 100;
        %fprintf('\tip average difference -> %.2f [A] (%.1f%%)\n', avg_diff, mean(perc_diff));
        %% keep only the samples where IP and IPLIUQE are similar
        %% assert(mean(perc_diff) < MAX_IP_PcERC_DIFF, 'Difference between IPLIUQE and IP is too high'); % very strict
        %ip_valid1 = perc_diff < MAX_IP_PERC_DIFF;
        %fprintf('\tip filtered -> %.1f%%, remaining -> %d/%d \n', 100*(1-sum(ip_valid1)/numel(ip_valid1)), sum(ip_valid1), numel(ip_valid1));
        %assert(sum(ip_valid1) > 0.8 * numel(ip_valid1), 'IP MEAS and IPLIUQE are different in too many samples');

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
        rq = rq; % LCFS r coordinate
        zq = zq; % LCFS z coordinate

        % Inputs ("real")
        Bm0 = LX.Bm; 
        Ff0 = LX.Ff;
        Ft0 = LX.Ft;
        Ia0 = LX.Ia; 
        Ip0 = LX.Ip;
        Iu0 = LX.Iu;
        rBt0 = LX.rBt; 
        % Inputs (fitted)
        Bm1 = LY.Bm;
        Ff1 = LY.Ff;
        Ft1 = LY.Ft;
        Ia1 = LY.Ia;
        Ip1 = LY.Ip;
        Iu1 = LY.Iu; 
        rBt1 = LY.rBt;

        % check the dimensions
        assert(all(size(Fx) == [65, 28, nt]), 'Fx has wrong size');
        assert(all(size(Iy) == [63, 26, nt]), 'Iy has wrong size');
        assert(all(size(Br) == [65, 28, nt]), 'Brx has wrong size');
        assert(all(size(Bz) == [65, 28, nt]), 'Bzx has wrong size');
        assert(all(size(rq) == [129, nt]), 'rq has wrong size');
        assert(all(size(zq) == [129, nt]), 'zq has wrong size');

        fprintf('\tBm0 size: %s, Bm1 size: %s\n', mat2str(size(Bm0)), mat2str(size(Bm1)));
        assert(all(size(Bm0) == size(Bm1) & size(Bm1) == [38, nt]), 'Bm has wrong size');
        assert(all(size(Ff0) == size(Ff1) & size(Ff1) == [38, nt]), 'Ff has wrong size');
        assert(all(size(Ft0) == size(Ft1) & size(Ft1) == [1, nt]), 'Ft has wrong size');
        assert(all(size(Ia0) == size(Ia1) & size(Ia1) == [19, nt]), 'Ia has wrong size');
        assert(all(size(Ip0) == size(Ip1) & size(Ip1) == [1, nt]), 'Ip has wrong size');
        assert(all(size(Iu0) == size(Iu1) & size(Iu1) == [38, nt]), 'Iu has wrong size');
        assert(all(size(rBt0) == size(rBt1) & size(rBt1) == [1, nt]), 'rBt has wrong size');
        
        %% filter out the NaN/Inf values [MILD]
        %mFx = reshape(all(all(~isnan(Fx) & ~isinf(Fx),1),2), [], 1);
        %mIy = reshape(all(all(~isnan(Iy) & ~isinf(Iy),1),2), [], 1);
        %mBr = reshape(all(all(~isnan(Br) & ~isinf(Br),1),2), [], 1);
        %mBz = reshape(all(all(~isnan(Bz) & ~isinf(Bz),1),2), [], 1);
        %mrq = reshape(all(~isnan(rq) & ~isinf(rq), 1), [], 1);
        %mzq = reshape(all(~isnan(zq) & ~isinf(zq), 1), [], 1);

        %mBm0 = reshape(all(~isnan(Bm0) & ~isinf(Bm0), 1), [], 1);
        %mBm1 = reshape(all(~isnan(Bm1) & ~isinf(Bm1), 1), [], 1);
        %mFf0 = reshape(all(~isnan(Ff0) & ~isinf(Ff0), 1), [], 1);
        %mFf1 = reshape(all(~isnan(Ff1) & ~isinf(Ff1), 1), [], 1);
        %mFt0 = reshape(~isnan(Ft0) & ~isinf(Ft0), [], 1);
        %mFt1 = reshape(~isnan(Ft1) & ~isinf(Ft1), [], 1);
        %mIa0 = reshape(all(~isnan(Ia0) & ~isinf(Ia0), 1), [], 1);
        %mIa1 = reshape(all(~isnan(Ia1) & ~isinf(Ia1), 1), [], 1);
        %mIp0 = reshape(~isnan(Ip0) & ~isinf(Ip0), [], 1);
        %mIp1 = reshape(~isnan(Ip1) & ~isinf(Ip1), [], 1);
        %mIu0 = reshape(all(~isnan(Iu0) & ~isinf(Iu0), 1), [], 1);
        %mIu1 = reshape(all(~isnan(Iu1) & ~isinf(Iu1), 1), [], 1);
        %mrBt0 = reshape(~isnan(rBt0) & ~isinf(rBt0), [], 1);
        %mrBt1 = reshape(~isnan(rBt1) & ~isinf(rBt1), [], 1);

        % valid = mFx & mIy & mBr & mBz & mrq & mzq & ...
        %valid = mFx & mIy & mrq & mzq & ...
        %    mBm0 & mBm1 & mFf0 & mFf1 & mFt0 & mFt1 & ...
        %    mIa0 & mIa1 & mIp0 & mIp1 & mIu0 & mIu1 & ...
        %    mrBt0 & mrBt1 & ip_valid1; % keep only the samples where all quantities are valid and IP is valid
        %fprintf('\tvalid samples -> [TOT: %.1f%%, %d] Fx:%.1f%%, Iy: %.1f%%, rq: %.1f%%, zq: %.1f%%, \n\tBm0: %.1f%%, Bm1: %.1f%%, Ff0: %.1f%%, Ff1: %.1f%%, Ft0: %.1f%%, Ft1: %.1f%%, Ia0: %.1f%%, Ia1: %.1f%%, Ip0: %.1f%%, Ip1: %.1f%%, Iu0: %.1f%%, Iu1: %.1f%%, rBt0: %.1f%%, rBt1: %.1f%%\n', ...
        %    100*sum(valid)/nt, sum(valid), ...
        %    100*sum(mFx)/nt, 100*sum(mIy)/nt, 100*sum(mrq)/nt, 100*sum(mzq)/nt, ...
        %    100*sum(mBm0)/nt, 100*sum(mBm1)/nt, 100*sum(mFf0)/nt, 100*sum(mFf1)/nt, ...
        %    100*sum(mFt0)/nt, 100*sum(mFt1)/nt, 100*sum(mIa0)/nt, 100*sum(mIa1)/nt, ...
        %    100*sum(mIp0)/nt, 100*sum(mIp1)/nt, 100*sum(mIu0)/nt, 100*sum(mIu1)/nt, ...
        %    100*sum(mrBt0)/nt, 100*sum(mrBt1)/nt);
        %assert(sum(valid) > 0.5 * nt, 'Nan/Inf filter -> not enough valid samples');
        
        % keep only the valid samples
        %t = t(valid);
        %Fx = Fx(:,:,valid);
        %Iy = Iy(:,:,valid);
        %Br = Br(:,:,valid);
        %Bz = Bz(:,:,valid);
        %rq = rq(:,valid);
        %zq = zq(:,valid);
        %Bm0 = Bm0(:,valid);
        %Bm1 = Bm1(:,valid);
        %Ff0 = Ff0(:,valid);
        %Ff1 = Ff1(:,valid);
        %Ft0 = Ft0(valid);
        %Ft1 = Ft1(valid);
        %Ia0 = Ia0(:,valid);
        %Ia1 = Ia1(:,valid);
        %Ip0 = Ip0(valid);
        %Ip1 = Ip1(valid);
        %Iu0 = Iu0(:,valid);
        %Iu1 = Iu1(:,valid);
        %rBt0 = rBt0(valid);
        %rBt1 = rBt1(valid);

        % decimate
        t = t(1:DECIMATION:end);
        Fx = Fx(:,:,1:DECIMATION:end);
        Iy = Iy(:,:,1:DECIMATION:end);
        Br = Br(:,:,1:DECIMATION:end);
        Bz = Bz(:,:,1:DECIMATION:end);
        rq = rq(:,1:DECIMATION:end);
        zq = zq(:,1:DECIMATION:end);
        Bm0 = Bm0(:,1:DECIMATION:end);
        Bm1 = Bm1(:,1:DECIMATION:end);
        Ff0 = Ff0(:,1:DECIMATION:end);
        Ff1 = Ff1(:,1:DECIMATION:end);
        Ft0 = Ft0(1:DECIMATION:end);
        Ft1 = Ft1(1:DECIMATION:end);
        Ia0 = Ia0(:,1:DECIMATION:end);
        Ia1 = Ia1(:,1:DECIMATION:end);
        Ip0 = Ip0(1:DECIMATION:end);
        Ip1 = Ip1(1:DECIMATION:end);
        Iu0 = Iu0(:,1:DECIMATION:end);
        Iu1 = Iu1(:,1:DECIMATION:end);
        rBt0 = rBt0(1:DECIMATION:end);
        rBt1 = rBt1(1:DECIMATION:end);

        % convert to single precision to save space
        Fx = single(Fx); 
        Iy = single(Iy);
        Br = single(Br);
        Bz = single(Bz);
        rq = single(rq);
        zq = single(zq);
        Bm1 = single(Bm1);
        Bm0 = single(Bm0);
        Ff1 = single(Ff1);
        Ff0 = single(Ff0);
        Ft1 = single(Ft1);
        Ft0 = single(Ft0);
        Ia1 = single(Ia1);
        Ia0 = single(Ia0);
        Ip1 = single(Ip1);
        Ip0 = single(Ip0);
        Iu1 = single(Iu1);
        Iu0 = single(Iu0);
        rBt1 = single(rBt1);
        rBt0 = single(rBt0);

        % print final sizes
        fprintf('\tFinal sizes -> Fx:%s, Iy:%s, Br:%s, Bz:%s, rq:%s, zq:%s, \n\tBm0:%s, Bm1:%s, Ff0:%s, Ff1:%s, Ft0:%s, Ft1:%s, Ia0:%s, Ia1:%s, Ip0:%s, Ip1:%s, Iu0:%s, Iu1:%s, rBt0:%s, rBt1:%s\n', ...
            mat2str(size(Fx)), mat2str(size(Iy)), mat2str(size(Br)), mat2str(size(Bz)), mat2str(size(rq)), mat2str(size(zq)), ...
            mat2str(size(Bm0)), mat2str(size(Bm1)), mat2str(size(Ff0)), mat2str(size(Ff1)), ...
            mat2str(size(Ft0)), mat2str(size(Ft1)), mat2str(size(Ia0)), mat2str(size(Ia1)), ...
            mat2str(size(Ip0)), mat2str(size(Ip1)), mat2str(size(Iu0)), mat2str(size(Iu1)), ...
            mat2str(size(rBt0)), mat2str(size(rBt1)));
        mdsclose; % Close the MDSplus connection
        total_shots = total_shots + 1;

        % save data into a .mat file
        save_file = fullfile(OUT_DIR, sprintf('%d.mat', shot));
        save(save_file, 't', 'Fx', 'Iy', 'Br', 'Bz', 'rq', 'zq', ...
            'Bm0', 'Bm1', 'Ff0', 'Ff1', 'Ft0', 'Ft1', ...
            'Ia0', 'Ia1', 'Ip0', 'Ip1', 'Iu0', 'Iu1', ...
            'rBt0', 'rBt1');
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

% copied from meqpost
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
