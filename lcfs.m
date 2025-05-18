clc; clear all; close all;

shot = 85804; % TCV shot number

mdsconnect('tcvdata.epfl.ch'); % Connect to the MDSplus server

mdsopen('tcv_shot', shot);

[t, ip] = tcvget('IPLIUQE'); % precalculated using liuqe

t = t(1:30:end)

[L,LX,LY] = liuqe(shot, t);

