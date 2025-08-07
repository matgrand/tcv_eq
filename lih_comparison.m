clear all; close all; clc;

disp('LIH Net LIUQE comparison...');

mdsconnect('tcvdata.epfl.ch'); % Connect to the MDSplus server


shots = [
    % 79742 % single null
    % 86310 % double null
    % 78893 % negative triangularity
    % 83848 % ?
    % 78071 % standard, test ctrl pts (t=0.571) (warn: theta is wrong)
    87188
];


disp('Shots to process:');
disp(shots);

addpath([pwd '/onnx_net_forward']);
addpath(genpath([pwd '/data']));

% ONNX_NET_PATH = '/home/grandin/repos/liuqe-ml/data/3011842/net.onnx'; % seems best, no 0*Iu required
ONNX_NET_PATH = '/home/grandin/repos/liuqe-ml/data/3048577/net.onnx'; 
net_forward_mex(ONNX_NET_PATH); % first call to load the model

% dummy control points
nq = 10; % number of control points
thetaq = linspace(0,2*pi,nq+1); thetaq = thetaq(1:end-1)';
rq = 0.88 + 0.15*cos(thetaq);
zq = 0.20 + 0.45*sin(thetaq);

%% init variables
% full grid
Fx_liuqe = []; Br_liuqe = []; Bz_liuqe = [];
Fx_lih = []; Br_lih = []; Bz_lih = [];
Fx_net = []; Br_net = []; Bz_net = [];
% control points
cFx_liuqe = []; cBr_liuqe = []; cBz_liuqe = [];
cFx_lih = []; cBr_lih = []; cBz_lih = [];
cFx_net = []; cBr_net = []; cBz_net = [];

for si = 1:length(shots) % 1->liuqe, 2->lih, 3->net
    shot = shots(si);

    [L2, LX2, LY2] = lih('tcv', shot, [], 'debug', 1);
    t = LY2.t'; % time vector
    [L1, LX1, LY1] = liuqe('tcv', shot, t);

    Fx_liuqe = [Fx_liuqe, LY1.Fx];
    Fx_lih = [Fx_lih, LY2.Fx];

    % calculate magnetic fields (copied from meqpost)
    i4pirxdzx = 1./(4*pi*L.dzx*L.rx');
    i4pirxdrx = 1./(4*pi*L.drx*L.rx');
    [Br1, Bz1] = meqBrBz(LY1.Fx, i4pirxdzx, i4pirxdrx, L.nzx, L.nrx);
    [Br2, Bz2] = meqBrBz(LY2.Fx, i4pirxdzx, i4pirxdrx, L.nzx, L.nrx);

    Br_liuqe = [Br_liuqe, Br1];
    Bz_liuqe = [Bz_liuqe, Bz1];
    Br_lih = [Br_lih, Br2];
    Bz_lih = [Bz_lih, Bz2];

    r = L.rrx(:);
    z = L.zzx(:);
    [Fx3, Br3, Bz3] = net_forward(LX_liuqe, r, z); % network inference
    Fx_net = [Fx_net, Fx3];
    Br_net = [Br_net, Br3];
    Bz_net = [Bz_net, Bz3];

end

% calculate magnetic fields using the ONNX net
% network inference
function [Fx, Br, Bz] = net_forward(LX, r, z)
    phys = [LX.Bm, LX.Ff, LX.Ft, LX.Ia, LX.Ip, LX.Iu, LX.rBt];
    [Fx, Br, Bz] = net_forward_mex(single(phys), single(r), single(z));
    Fx = double(Fx); Br = double(Br); Bz = double(Bz); % convert to double
end

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
