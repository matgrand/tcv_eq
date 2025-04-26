clear all; close all; clc;

START_SHOT = 77662; % Dec 2022, https://spcwiki.epfl.ch/wiki/Alma_database
END_SHOT = 85804; % April 2025
N_SHOTS = 10000; % Number of shots to process

% Directory to save the output .mat files
% OUT_DIR = 'ds'; % testing
OUT_DIR = '/NoTivoli/grandin/ds'; % more space available

DECIMATION = 5; % Decimation factor for the time vector

MIN_TIME_SAMPLES = 10; % Minimum number of time samples to keep the shot
MAX_IP_PERC_DIFF = 1.0; % Maximum percentage difference between IPLIUQE and IP

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

        t = t(1:DECIMATION:end); % decimate the time vector
        ip1 = ip1(1:DECIMATION:end); % decimate the plasma current vector
        
        [t2, ip2] = tcvget('IP', t); % calculated using magnetics at liuqe times
        
        t_diff = abs(t - t2);
        
        fprintf('\tTime difference: mean: %.2e [s], max: %.2e [s]\n', mean(t_diff), max(t_diff));
        fprintf('\t# of time samples: %d\n', numel(t));
        assert(max(t_diff) < 1e-8, 'Times do not coincide');
        assert(numel(t) > MIN_TIME_SAMPLES, 'Not enough time samples');
        fprintf('\ttime steps -> n: %d, mean: %.2f [µs], std: %.2f [µs]\n', numel(t), mean(diff(t) * 1e6), std(diff(t) * 1e6));
        
        avg_ip = mean(abs(ip2));
        ip_diff = abs(ip1 - ip2);
        avg_diff = mean(ip_diff);
        perc_diff = (avg_diff / avg_ip) * 100;
        
        fprintf('\tAverage difference: %.2f [A] (%.2f%%)\n', avg_diff, perc_diff);
        assert(perc_diff < MAX_IP_PERC_DIFF, 'Difference between IPLIUQE and IP is too high');

        % calculate liuqe equilibrium at the good plasma current time
        [L,LX,LY] = liuqe(shot, t);
        Fx = single(LY.Fx); % Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
        Iy = single(LY.Iy); % Plasma current density map | `(ry,zy,t)` | `[A/m^2]` |
        Ia = single(LY.Ia); % Fitted poloidal field coil currents | `(*,t)` | `[A]` |
        Bm = single(LY.Bm); % Simulated magnetic probe measurements | `(*,t)` | `[T]` |
        Uf = single(LY.Uf); % Simulated flux loop poloidal flux | `(*,t)` | `[Wb]` |
        t = single(t);
        Ip = single(ip1);

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
        
        mdsclose; % Close the MDSplus connection
        total_shots = total_shots + 1;

        % save data into a .mat file
        save_file = fullfile(OUT_DIR, sprintf('%d.mat', shot));
        save(save_file, 't', 'Ip', 'Fx', 'Iy', 'Ia', 'Bm', 'Uf');
        fprintf('\x1b[32m   Data saved to: %s\x1b[0m\n', save_file);
        
        shot_processing_times(i) = toc(start_time);
        fprintf('   Proc time: %.2f s, ETA: %.0f min\n', shot_processing_times(i), sum(shot_processing_times) / i * (length(shots) - i) / 60);
        
    catch ME
        fprintf('\x1b[31m\tError processing shot %d: %s\x1b[0m\n', shot, ME.message);
        continue; % Skip to the next shot on error
    end % try-catch
end % end shots loop

fprintf('\nProcessing complete for all shots.\n');
fprintf('Output files saved in: %s\n', OUT_DIR);