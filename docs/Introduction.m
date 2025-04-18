%% An introduction to MEQ
% In this example we will use the 'anamak' tokamak, a tokamak with a simplified 
% geometry description for educational and code testing purposes.
%% Codes In MEQ
%% 
% * FBTE (|fbt|) - Fit desired equilibrium description
% * LIUQE (|liu|) - Equilibrium reconstruction from measurements
% * FGS (|fgs|) - Forward Grad-Shafranov Static
% * FGE (|fge|) - Forward Grad-Shafranov Evolutive
%% 
% Each code is treated in more detail in later tutorials.
%% The MEQ structures
% MEQ uses several standardized structures that contain information. It is good 
% to be familiar with these structures since they appear in various places in 
% the code.
% 
% If we call any code of the |meq| suite with only one output argument, the code 
% is not run but only the so-called |L| structure is calculated. This structure 
% contains parameters, geometry, and pre-calculated data used to run the code.
% Each code produces a different |L| structure although some fields will be
% shared.
% 
% Let's first get an |L| structure by calling |fbt|. The first parameter is 
% the tokamak, where we choose |'ana'|  for the anamak tokamak. The second argument 
% represents the shot number, which will be discussed later.  

L = fbt('ana',1);
%% 
% We see that this structure contains two substructures, |P| and |G| as well 
% as other quantities. Let's first look at those.
%% The |P| structure

L.P;
%% 
% This contains parameters that will be used to run the code. These parameters 
% are documented in:
% * |meqp.m| for general parameters
% * |meqp<tok>.m| for tokamak-specific parameters (e.g. |meqpana.m|)
% * |fbtp.m| for code-specific parameters general to all tokamaks
% * |fbtp<tok>.m| for code-specific and tokamak-specific parameters, for example |fbtpana.m|

help meqp
help meqpana
help fbtp
help fbtpana
%% 
% These parameters can be customized by entering parameter, value pairs in the 
% call to the code (after the first three arguments being |tokamak|, |shot| and 
% |time|), for example

fprintf('iterq=%d\n',L.P.iterq)  % old L.P.iterq value
L = fbt('ana',1,[],'iterq',1); % set another value
fprintf('iterq=%d\n',L.P.iterq) % new L.P.iterq value
%% The |G| structure
% The |G| structure contains the geometry description of the tokamak, including 
% many useful quantities such as mutual inductances, grids, coil locations and 
% more.
L.G;
%% 
% We can plot the geometry using |meqgplot.m|

clf; meqgplot(L.G)

%%
% Or we can plot quantites we want manually by hand:

figure; clf; hold on;
plot(L.G.rw,L.G.zw,'sr','MarkerFaceColor','r');  % coil windings
plot(L.G.rv,L.G.zv,'ob'); % vessel
plot(L.G.rl,L.G.zl,'.k'); % limiter
plot(L.G.rm,L.G.zm,'sm','MarkerFaceColor','m'); % magnetic probes
plot(L.G.rf,L.G.zf,'oc','MarkerFaceColor','c'); % flux loops
axis equal;
%% 
% and add the computational grid. Note that we use the derived quantities  |L.nzx|,|L.nrx| 
% stored in the |L| structure which are the grid sizes

mesh(L.G.rx,L.G.zx,zeros(L.nzx,L.nrx),...
    'edgecolor','k','facecolor','none'); view(2); shg
axis equal

%% Fluxes and fields
% We can use the |L| structure already to do very useful things. For example, 
% we can compute the flux generated by each circuit on the computational grid, 
% as follows

Ia = zeros(L.G.na,1); % initialize current vector
ia = 3; % index of circuit to plot
Ia(ia) = 1000; % put 1kA of current
Fx = reshape(L.G.Mxa*Ia,L.nzx,L.nrx); % compute flux, and reshape it into right size for plotting
clf; hold on;
plot(L.G.rl,L.G.zl,'.'); contour(L.rx,L.zx,Fx*1e3,21); axis equal; colorbar; 
title(sprintf('Flux [mWb] produced by a %2.2f kA current in coil %s',Ia(ia)/1e3,L.G.dima{ia}),...
    'Interpreter','none');
  
%% 
% Note how |L.G.dima| contains a label for the circuit.
% 
% Similarly we can easily plot the radial, vertical and poloidal field, but 
% first we need to add the relevant matrices to |G|. (They are not included by 
% default since they are not always needed).

L.G = meqg(L.G,L.P,'Brxa','Bzxa'); % get required matrices
Br = reshape(L.G.Brxa*Ia,L.nzx,L.nrx);
Bz = reshape(L.G.Bzxa*Ia,L.nzx,L.nrx);
Bp = sqrt(Br.^2 + Bz.^2);

clf; hold on;
vars = {'Fx','Br','Bz','Bp'}, units = {'mWb','mT','mT','mT'};
for ii=1:4
  subplot(2,2,ii); hold on;
  var = eval(vars{ii}); % get variable
  plot(L.G.rl,L.G.zl,'.'); contour(L.rx,L.zx,var*1e3,21); axis equal; colorbar;
  title(sprintf('%s [%s]',...
    vars{ii},units{ii}),...
    'Interpreter','none');
end

%% Coordinate system
% Since we just plotted fluxes and fields, we should talk about coordinate
% systems. MEQ uses coordinate system as defined by COCOS=17 
% O. Sauter and S. Y. Medvedev, Comput. Phys. Commun., 184(2), 293–302, (2013)
% 
% Specifically it 
% * Uses a right-handed $(R,phi,Z)$ coordinate system
% * Defines flux [in Wb] as $\psi = \int B_z dS_z$.
% * Uses a right-handed toroidal coordinate system $(\rho,\theta,phi)$

%% |meq| data
% Calling a meq code with two arguments gives the input data structure |LX| 
% required for the code to run. We separate the data from the structure |L|, where  
% |LX| contains (time/shot-dependent) data while |L| is intended to contain fixed 
% tokamak/parameter information. Let's use another code for this: the equilibrium 
% reconstruction code |liuqe|

[L,LX] = liu('ana',1,0);
LX
%% 
% The meaning of these quantities is documented in

help liux
%% 
% In the case of |liuqe|, the data structure contains measurements that can be 
% used to reconstruct the equilibrium, for example |Bm| are the fields measured 
% at the probes and |Ff| are fluxes measured at flux loops. From |LX.t| you can 
% see that only one time point is provided for this (ficticious) shot. 
% 
% Let's plot some quantities

figure; hold on;
subplot(131);
barh(1:L.G.nm,LX.Bm); 
set(gca,'YTick',1:L.G.nm,'YTickLabels',L.G.dimm,'TickLabelInterpreter','none');
subplot(132);
barh(1:L.G.nf,LX.Ff); 
set(gca,'YTick',1:L.G.nf,'YTickLabels',L.G.dimf,'TickLabelInterpreter','none');
subplot(133);
barh(1:L.G.na,LX.Ia); 
set(gca,'YTick',1:L.G.na,'YTickLabels',L.G.dima,'TickLabelInterpreter','none');
%% 
% With our previous example, we can plot the vacuum flux generated by the measured 
% coil currents, showing a dominantly vertical field.

Fx = reshape(L.G.Mxa*LX.Ia,L.nzx,L.nrx);
figure; hold on;
plot([L.G.rl;L.G.rl(1)],[L.G.zl;L.G.zl(1)],'-k'); % plot closed limiter contour
contour(L.rx,L.zx,Fx,21); axis equal;
%% My first meq equilibrium
% Let's now calculate our first meq equilibrium using the |FBT| code. As you 
% know, this code determines the coil currents required to maintain a desired 
% equilibrium. We now call the function with 3 output arguments to force the calculation.

[L,~,LY] = fbt('ana',1);
%% 
% We now get an output structure |LY| that contains the equilibrium. In this 
% case only 1 time slice. The parameters contained in |LY| are documented in 
% |meqt.m| for generic quantities, and |<code>t.m| for code-specific
% quantities, for example |fbtt.m|

help meqt
help fbtt
%% 
% we can plot the key quantities for illustration:

figure;
subplot(121); hold on;
contourf(L.rx,L.zx,LY.Fx,21);
contour(L.rx,L.zx,LY.Fx,LY.FB*[1 1],'linecolor','w','linewidth',2); % LCFS
plot(LY.rA,LY.zA,'or'); % magnetic axis
plot([L.G.rl;L.G.rl(1)],[L.G.zl;L.G.zl(1)],'-k','linewidth',2); % plot closed limiter contour
axis equal; title('flux distribution')
subplot(122); hold on; axis xy
imagesc(L.ry,L.zy,LY.Iy); % current distribution (note different grid)
contour(L.rx,L.zx,LY.Fx,LY.FB*[1 1],'linecolor','w','linewidth',2); % LCFS
plot(LY.rIp/LY.Ip,LY.zIp/LY.Ip,'*r'); % current centroid
plot([L.G.rl;L.G.rl(1)],[L.G.zl;L.G.zl(1)],'-k','linewidth',2); % plot closed limiter contour
title('current distribution');
axis equal;
%% 
% As you can see this is a rather dull circular equilibrium. Let's plot something 
% more exciting, and use the function |meqplott| that does all the plotting for 
% us.

[L,~,LY] = fbt('ana',2);
figure; meqplott(L,LY);
%% 
% If you want to see more details of the flux in the region around the coils, 
% as well as the coil currents, use |meqplotfancy| function. This needs a specific 
% parameter izgrid to be set first to compute the flux on a secondary grid.

[L,~,LY] = fbt('ana',2,[],'izgrid',true);
figure; meqplotfancy(L,LY);
