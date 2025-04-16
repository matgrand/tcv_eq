clc; clear all; close all;

SHOT = 85444; % shot number

T = 0.8; % [s] time istant when to calculate the eq

[L,LX,LY] = liuqe(SHOT, T)

Fx = LY.Fx;

% save Fx as a mat file
save('eq.mat', 'Fx')
