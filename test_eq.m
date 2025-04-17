clear all; close all; clc;


% 1. Set the desired shot number
shot = 79999; % 79999, 83995
mdsopen('tcv_shot', shot); % Open the MDSplus connection to the TCV database

% 2. Call tcvget to retrieve the plasma current (IP)
%    We omit the 'time' argument to get the full time trace.
[t_ip, ip_data] = tcvget('IP');

mdsclose; % Close the MDSplus connection


% plot and save figure
figure
plot(t_ip, ip_data)
xlabel('Time (s)')
ylabel('Plasma current (A)')
title(['Plasma current for shot ' num2str(shot)])
grid on
saveas(gcf, ['figs/shot_' num2str(shot) '_ip.svg']) % Save the figure as an SVG file
