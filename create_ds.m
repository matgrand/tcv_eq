clear; clc; close all;

%% --- Configuration ---

% Input file containing shot numbers (one per line)
SHOT_LIST_FILE = 'good_shots.txt';

N_DS = 100; % Number of shots to process

% Directory to save the output .mat files
% OUT_DIR = 'ds'; % testing
OUT_DIR = '/NoTivoli/grandin/ds' % more space available

DECIMATION = 10; % Decimation factor for the time vector

IP_THRSH = 25000; % I plasma threshold to filter time (TODO: the method should be improved)

%% --- Setup ---

% Check if the shot list file exists
if ~exist(SHOT_LIST_FILE, 'file') error('Shot list file not found: %s', SHOT_LIST_FILE); end
if ~exist(OUT_DIR, 'dir') mkdir(OUT_DIR); end % Create output directory if it doesn't exist

%% --- Read Shot List ---
fprintf('Reading shot list from: %s\n', SHOT_LIST_FILE);
fid = fopen(SHOT_LIST_FILE, 'r');
if fid == -1 error('Could not open shot list file: %s', SHOT_LIST_FILE); end
% Read all lines and process shot numbers
shots = [];
while ~feof(fid)
    line = strtrim(fgetl(fid)); % Read a line and trim whitespace
    if ~isempty(line)
        numbers = sscanf(line, '%d,'); % Extract numbers separated by commas
        shots = [shots; numbers]; % Append to the shot list
    end
end
fclose(fid);
fprintf('Found %d shot numbers to process.\n', length(shots));


% % keep only the first x shots
% shots = shots(1:15);

% % keep x random shots (+ the first one)
% shots = [shots(1); shots(randperm(length(shots), N_DS-1))]; %

% keep x random shots (+ the first one)
shots = shots(randperm(length(shots), N_DS)); %


%% --- Main Processing Loop ---

fprintf('\nStarting data retrieval loop...\n');

shot_processing_times = zeros(length(shots), 1); % Preallocate array for shot processing times

for i = 1:length(shots)
    try
        start_time = tic; % Start timer for processing each shot

        shot = shots(i); % Get the current shot number
        fprintf('Processing shot %d (%d of %d)\n', shot, i, length(shots));

        mdsopen('tcv_shot', shot); % Open the MDSplus connection to the TCV database

        % Call tcvget to retrieve the plasma current (IP)
        % We omit the 'time' argument to get the full time trace.
        [t_ip, ip_data] = tcvget('IP'); % calculated using magnetics
        % [t_ip, ip_data] = tcvget('IPLIUQE'); % calculated using liuqe

        % Filter the time vector to remove values below the threshold
        good_ip_idxs = find(abs(ip_data) > IP_THRSH);

        good_ip_idxs = good_ip_idxs(1:DECIMATION:end); % decimate the idxs for now 
        assert(numel(good_ip_idxs) > 0, 'No valid plasma current data found.');

        
        filtered_percentage = (1 - numel(good_ip_idxs) / numel(ip_data)) * 100;
        remaining_samples = numel(good_ip_idxs);
        fprintf('   Filtered -> %.2f%%, left -> %d samples\n', filtered_percentage, remaining_samples); 
        
        % find the mean and std of the time step
        dts = diff(t_ip(good_ip_idxs));
        dt_mean = mean(dts);
        dt_std = std(dts);
        fprintf('   time step ->  mean: %.2f µs, std: %.2f µs\n', dt_mean * 1e6, dt_std * 1e6);
        % calculate liuqe equilibrium at the good plasma current time
        [L,LX,LY] = liuqe(shot, t_ip(good_ip_idxs));
        Fx = single(LY.Fx); % Plasma poloidal flux map | `(rx,zx,t)` | `[Wb]` |
        Iy = single(LY.Iy); % Plasma current density map | `(ry,zy,t)` | `[A/m^2]` |
        Ia = single(LY.Ia); % Fitted poloidal field coil currents | `(*,t)` | `[A]` |
        
        Bm = single(LY.Bm); % Simulated magnetic probe measurements | `(*,t)` | `[T]` |
        Uf = single(LY.Uf); % Simulated flux loop poloidal flux | `(*,t)` | `[Wb]` |
        
        t = single(t_ip(good_ip_idxs));
        Ip = single(ip_data(good_ip_idxs));

        mdsclose; % Close the MDSplus connection

        % save data into a .mat file
        save_file = fullfile(OUT_DIR, sprintf('%d.mat', shot));
        save(save_file, 't', 'Ip', 'Fx', 'Iy', 'Ia', 'Bm', 'Uf');
        fprintf('   Data saved to: %s\n', save_file);

        shot_processing_times(i) = toc(start_time);
        fprintf('   Proc time: %.2f s, ETA: %.0f min\n', shot_processing_times(i), sum(shot_processing_times) / i * (length(shots) - i) / 60);
    catch ME
        fprintf('   Error processing shot %d: %s\n', shot, ME.message);
        continue; % Skip to the next shot on error
    end
end % end shots loop

fprintf('\nProcessing complete for all shots.\n');
fprintf('Output files saved in: %s\n', OUT_DIR);