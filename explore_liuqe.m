clear all; close all; clc;


shot = 86256;



mdsconnect('tcvdata.epfl.ch'); % Connect to the MDSplus server


mdsopen('tcv_shot', shot); % Open the MDSplus connection to the TCV database

% % to calculate liuqe and get the info on LCFS
% [L,LX,LY] = liuqe(shot, 1, 'iterq', 10);

% % to get mag fields Brx, Bzx, Btx
% [L,LX,LY] = liuqe(shot, 1, 'ifield', true);


% to get liuqe from mdsplus database, without recalculating
[L, LY] = mds2meq(shot, 'LIUQE.M');

times = LY.t;

% %decimate the data
% times = times(1:10:end);

% to get only the inputs, basically the same as RT, (caveat maybe some filtering on magentics)
[L, LX] = liuqe(shot, times);



% calculatin Br, Bz, Bt fields, copying functions from source code of meqpost (because mds2meq does
% not allow to run meqpost)

i4pirxdzx = 1./(4*pi*L.dzx*L.rx');
i4pirxdrx = 1./(4*pi*L.drx*L.rx');
[Brx,Bzx] = meqBrBz(LY.Fx,i4pirxdzx,i4pirxdrx,L.nzx,L.nrx);
% Btx = meqBt(L,LY.Fx,Opy,ag,rBt,F0,F1,TQ);
% LY = meqlarg(LY,Brx,Bzx,Btx);

Bx = sqrt(Brx.^2 + Bzx.^2); % total magnetic field

% plot in 2 separate figures Brx, Bzx
figure;
rr, zz = L.rrx, L.zzx;
nt = 1200;
subplot(3,1,1);
scatter(rr, zz, 10, Brx(:,:,nt), 'filled');
title('Brx');
subplot(3,1,2);
scatter(rr, zz, 10, Bzx(:,:,nt), 'filled');
title('Bzx');
subplot(3,1,3);
scatter(rr, zz, 10, Bx(:,:,nt), 'filled');
title('Bx');

%plot Fx
figure;
scatter(rr, zz, 10, LY.Fx(:,:,nt), 'filled');
title('Fx');

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

% function Btx = meqBt(L,Fx,Opy,ag,rBt,F0,F1,TQ)
%     % Btx = meqBt(L,Fx,Opy,ag,rBt,F0,F1,TQ)
%     % Computes toroidal field on x grid

%     Btx  = rBt*repmat(1./L.rx',L.nzx,1);
%     Bty = Btx(L.lxy);

%     nB = numel(F1);
%     for iB = 1:nB
%         Opyi = (Opy==iB); % mask for this domain    
%         if isequal(func2str(L.bfct),'bf3pmex')
%             assert(false, 'not implemented yet');
%             Btyi = L.bfct(8,L.bfp, Fx,F0(iB),F1(iB),int8(Opyi),ag(:,iB),rBt,L.idsx,L.iry);
%             Bty(Opyi) = Btyi(Opyi);
%         else
%             % Mode 8 not yet available for other bfs, to be added later
%             FyN = (Fx(L.lxy)-F0(iB))/(F1(iB)-F0(iB)); % rhopol.^2
%             Bty(Opyi) = interp1(L.pQ.^2,TQ(:,iB)',FyN(Opyi))./L.rry(Opyi);
%         end    
%     end
%     Btx(L.lxy) = Bty;
% end    

