clc; clear all; close all;

SHOT = 85508; %85516; %85517; % shot number 
% ts = [0.0:0.001:1.5]; % time vector

disp(['SHOT number: ', num2str(SHOT)]);
% [L,LX,LY] = liuqe(SHOT, ts);
[L,LX,LY] = liuqe(SHOT);

Fx = LY.Fx;

save(['data/Fx_', num2str(SHOT)], 'Fx');


% % save everything with the shot number
% save(['L_', num2str(SHOT)], 'L');
% save(['LX_', num2str(SHOT)], 'LX');
% save(['LY_', num2str(SHOT)], 'LY');
