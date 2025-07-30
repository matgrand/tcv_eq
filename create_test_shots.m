clear all; close all; clc;

% TODO consider adding vessel currents: I_VESSEL

% Directory to save the output .mat files
OUT_DIR = 'test_shots'; % more space available

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
    79742 % single null
    86310 % double null
    78893 % negative triangularity
    83848 % ?
    78071 % standard, test ctrl pts (t=0.571) (warn: theta is wrong)
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

        % % last closed flux surface (LCFS) (unfortunately it's not in mds2meq outputs (yet))
%         rq = mdsdata('tcv_eq("R_EDGE", "LIUQE.M", "NOEVAL")'); % LCFS r coordinate
%         zq = mdsdata('tcv_eq("Z_EDGE", "LIUQE.M", "NOEVAL")'); % LCFS z coordinate
%         theta = mdsdata('tcv_eq("THETA", "LIUQE.M", "NOEVAL")'); 
        % check that theta is the same as theta0, first size, then values
	    % assert(all(size(theta) == size(theta0)), 'theta and theta0 have different sizes');
        % assert(all(abs(theta0 - theta) < 1e-6), 'theta and theta0 are different');

        % analyze the time vector
        assert(max(abs(t2 - t)) < 1e-8, 'Time vectors do not coincide');
	    assert(numel(t) > 1, sprintf('Time vector has insufficient elements: t:%s', mat2str(size(t))));
        nt = numel(t); % number of time samples
        t_diff = abs(t - t2);
        fprintf('\ttime steps -> n: %d, mean: %.2f [µs], std: %.2f [µs]\n', numel(t), mean(diff(t) * 1e6), std(diff(t) * 1e6));
        assert(max(t_diff) < 1e-8, 'Times do not coincide');
        
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
        % rq = rq; % LCFS r coordinate
        % zq = zq; % LCFS z coordinate

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
        % assert(all(size(rq) == [129, nt]), 'rq has wrong size');
        % assert(all(size(zq) == [129, nt]), 'zq has wrong size');

        assert(all(size(Bm) == [38, nt]), 'Bm has wrong size');
        assert(all(size(Ff) == [38, nt]), 'Ff has wrong size');
        assert(all(size(Ft) == [1, nt]), 'Ft has wrong size');
        assert(all(size(Ia) == [19, nt]), 'Ia has wrong size');
        assert(all(size(Ip) == [1, nt]), 'Ip has wrong size');
        assert(all(size(Iu) == [38, nt]), 'Iu has wrong size');
        assert(all(size(rBt) == [1, nt]), 'rBt has wrong size');

        % print final sizes
        fprintf('\tFinal sizes -> Fx:%s, Iy:%s, Br:%s, Bz:%s, \n\tBm:%s, Ff:%s, Ft:%s, Ia:%s, Ip:%s, Iu:%s, rBt:%s\n', ...
            mat2str(size(Fx)), mat2str(size(Iy)), mat2str(size(Br)), mat2str(size(Bz)), ...
            mat2str(size(Bm)), mat2str(size(Ff)), mat2str(size(Ft)), mat2str(size(Ia)), ...
            mat2str(size(Ip)), mat2str(size(Iu)), mat2str(size(rBt)));
        mdsclose; % Close the MDSplus connection
        total_shots = total_shots + 1;

        % save data into a .mat file
        save_file = fullfile(OUT_DIR, sprintf('%d.mat', shot));
        save(save_file, 't', 'Fx', 'Iy', 'Br', 'Bz', ...
            'Bm', 'Ff', 'Ft', 'Ia', 'Ip', 'Iu', 'rBt');
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
