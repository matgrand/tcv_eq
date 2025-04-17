clc; clear all; close all;

% % connect to MDSplus server
% assert(mdsconnect('tcvdata.epfl.ch') == 1, 'MDSplus connection failed');

% SHOT = 79999; %85516; %85517; % shot number 
% disp(['SHOT number: ', num2str(SHOT)]);

% ts = [0.0:0.001:1.5]; % time vector

% % % [L,LX,LY] = liuqe(SHOT, ts);
% % [L,LX,LY] = liuqe(SHOT);

% % Fx = LY.Fx;

% % save(['data/Fx_', num2str(SHOT)], 'Fx');



% [t, ip] = tcvget('IP', 70000, ts);


% % plot and save the figure
% figure;
% plot(t, ip);
% xlabel('Time (s)');
% ylabel('Ip (A)');
% title('TCV Ip');
% saveas(gcf, 'tcv_ip.png');



% 1. Set the desired shot number
shot = 83995;
mdsopen('tcv_shot', shot); % Replace 'tcv_shot' if your TCV tree name is different

% 2. Call tcvget to retrieve the plasma current (IP)
%    We omit the 'time' argument to get the full time trace.
[t_ip, ip_data] = tcvget('IP');

% 3. (Optional) Close the MDSplus connection
mdsclose;

% 4. (Optional) You can now use the data, for example, plot it:
if exist('t_ip', 'var') && exist('ip_data', 'var') && ~isempty(t_ip)
    figure;
    plot(t_ip, ip_data / 1e3); % Plot current in kA
    xlabel('Time (s)');
    ylabel('Plasma Current IP (kA)');
    title(['Plasma Current for Shot #', num2str(shot)]);
    grid on;
    % Save the figure
    saveas(gcf, ['tcv_ip_shot_', num2str(shot), '.png']);
else
    disp(['Could not retrieve IP data for shot ', num2str(shot)]);
end