clear all; close all; clc;


shot = 86256:



mdsconnect('tcvdata.epfl.ch'); % Connect to the MDSplus server


mdsopen('tcv_shot', shot); % Open the MDSplus connection to the TCV database

% to calculate liuqe and get the info on LCFS
[L,LX,LY] = liuqe(shot, 1, 'iterq', 10);


% to get liuqe from mdsplus database, without recalculating
[L, LY] = mds2meq(shot, 'LIUQE.M');

times = LY.t;

% %decimate the data
% times = times(1:10:end);

% to get only the inputs, basically the same as RT, (caveat maybe some filtering on magentics)
[L, LX] = liuqe(shot, times)