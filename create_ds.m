% MATLAB Script: process_tcv_shots.m
%
% Iterates through a list of TCV shot numbers, retrieves specified signals
% using tcvget, and saves the data for each shot into a separate .mat file.

clear; clc; close all;

%% --- Configuration ---

% Input file containing shot numbers (one per line)
shotListFile = 'data/good_shots.txt';

% Directory to save the output .mat files
outputDir = 'data'; % Will be created if it doesn't exist

% List of signals to retrieve using tcvget
% Add or remove signals as needed. Make sure they are valid tcvget keywords.
signalList = {'IP', 'IPLIUQE', 'IPREF', 'BT', 'SPLASMA', 'ZMAG'};

% TCV MDSplus Tree 
tcvTree = 'tcv_shot';

%% --- Setup ---

% Check if the shot list file exists
if ~exist(shotListFile, 'file') error('Shot list file not found: %s', shotListFile); end

%% --- Read Shot List ---
fprintf('Reading shot list from: %s\n', shotListFile);
fid = fopen(shotListFile, 'r');
if fid == -1
    error('Could not open shot list file: %s', shotListFile);
end
% Read all lines and process shot numbers
shotList = [];
while ~feof(fid)
    line = strtrim(fgetl(fid)); % Read a line and trim whitespace
    if ~isempty(line)
        numbers = sscanf(line, '%d,'); % Extract numbers separated by commas
        shotList = [shotList; numbers]; % Append to the shot list
    end
end
fclose(fid);
fprintf('Found %d shot numbers to process.\n', length(shotList));


% keep only the first 3 shots
shotList = shotList(1:100);


%% --- Main Processing Loop ---

fprintf('\nStarting data retrieval loop...\n');

for i = 1:length(shotList)
    currentShot = shotList(i);
    fprintf('--- Processing Shot %d (%d/%d) ---\n', currentShot, i, length(shotList));

    % Define the output filename for this shot
    outputFilename = sprintf('%d.mat', currentShot);
    outputFilepath = fullfile(outputDir, outputFilename);

    % Skip if the output file already exists (optional)
    % if exist(outputFilepath, 'file')
    %     fprintf('Output file already exists, skipping: %s\n', outputFilepath);
    %     continue;
    % end

    % Structure to hold data for the current shot
    shotData = struct();
    shotData.shotNumber = currentShot; % Store shot number for reference

    mdsConnectionOpened = false; % Flag to track MDSplus connection

    try
        % 1. Open MDSplus connection for the current shot
        fprintf('Opening MDSplus connection for shot %d...\n', currentShot);
        mdsopen(tcvTree, currentShot);
        mdsConnectionOpened = true;
        fprintf('Connection opened.\n');

        % 2. Loop through the requested signals
        for j = 1:length(signalList)
            signalName = signalList{j};
            fprintf('Retrieving signal: %s ... ', signalName);

            % Use a try-catch block for tcvget in case a signal is missing
            try
                [t, x] = tcvget(signalName);

                % Store data in the structure
                % Create valid field names (e.g., t_IP, data_IP)
                timeFieldName = sprintf('t_%s', signalName);
                dataFieldName = signalName; % Keep original signal name for data field

                shotData.(timeFieldName) = t;
                shotData.(dataFieldName) = x;
                fprintf('Done.\n');

            catch ME_tcvget
                fprintf('WARNING: Failed to retrieve signal %s for shot %d.\n', signalName, currentShot);
                fprintf('         Error: %s\n', ME_tcvget.message);
                % Store NaN or empty to indicate failure
                timeFieldName = sprintf('t_%s', signalName);
                 dataFieldName = signalName;
                shotData.(timeFieldName) = NaN;
                shotData.(dataFieldName) = NaN;
            end % end inner try-catch for tcvget
        end % end signal loop

        % 3. Close MDSplus connection
        fprintf('Closing MDSplus connection for shot %d...\n', currentShot);
        mdsclose;
        mdsConnectionOpened = false; % Reset flag
        fprintf('Connection closed.\n');

        % 4. Save the data structure to a .mat file
        fprintf('Saving data to: %s\n', outputFilepath);
        % Use '-struct' flag to save fields of shotData as individual variables
        save(outputFilepath, '-struct', 'shotData');
        fprintf('Save complete.\n');

    catch ME_main
        fprintf('\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');
        fprintf('CRITICAL ERROR processing shot %d: %s\n', currentShot, ME_main.message);
        fprintf('Stack Trace:\n');
        disp(ME_main.getReport());
        fprintf('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!\n');

        % Attempt to close MDSplus connection if it was opened and an error occurred
        if mdsConnectionOpened
            try
                fprintf('Attempting to close MDSplus connection after error...\n');
                mdsclose;
                fprintf('Connection closed.\n');
            catch ME_close
                fprintf('WARNING: Could not close MDSplus connection after error: %s\n', ME_close.message);
            end
        end
        % Decide whether to continue with the next shot or stop
        % continue; % Uncomment to try processing next shot despite error
        % break;    % Uncomment to stop the script on critical error
    end % end main try-catch

    fprintf('--- Finished Shot %d ---\n\n', currentShot);

end % end shot loop

fprintf('\nProcessing complete for all shots.\n');
fprintf('Output files saved in: %s\n', outputDir);