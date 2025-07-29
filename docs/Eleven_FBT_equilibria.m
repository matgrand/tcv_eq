%% Eleven FBT equilibria for Anamak
% This tutorial illustrates the use of constraints to specify equilibria in 
% FBT.
% 
% The function of |fbt|, the meaning of various variables, as well as the setup 
% of the constrained optimization problem are described by

help fbthelp
% Initialize L structure and weights for coil
% We first initialize an 'empty' case with no constraints

shot = 0; t=0; % loads empty case - no shot parameters defined
L = fbt('ana',shot,t,...
  'izgrid',1,... % add extended grid to plot flux beyond limiter
  'iterq',50,... % switch on post processing with 50 iterations
  'pq',[],'npq',10);  % Override default and set number of rho points for contours

%% Set some global plasma quantities
LX0.Ip = 200e3; % plasma current
LX0.bp = 1;     % beta poloidal
LX0.qA =  1;     % q on axis
LX0.rBt = 1;     % rBt

%% Cost function weights/errors

% Each constraint group has a global absolute error |gp?d| (a scalar) and a
% relative error |gp?e| (a vector with as many elements as elements in the
% group), such that the expected error for each element is |gp?d*gp?e|.
%
% The global errors should be set such that a normalized error of one is
% acceptable. For example on anamak the gpfd error is set to
% |0.01*L.Fx0*Ip/L.Ip0| which is 1 percent of the typical poloidal flux
% value for the machine at this Ip value.
%
% Hint: when running |fbt| with |debugplot=2| the bar plot on the right
% side should show residuals of order 1.

% We then set the global errors for the currents

% % global errors
LX0.gpfd = 1e-2*L.Fx0*sum(LX0.Ip,1)/L.Ip0;  % Global errors for flux points (~ 0.008 Wb)
LX0.gpid = 5e-2*max(abs(L.Ia0));             % Global error for Current constraints (~ 80 kA)
LX0.gpbd = LX0.gpfd/100;                     % Global error for Field constraints (~ 8e-5 T)
LX0.gpcd = LX0.gpfd/100;                     % Global error for Hessian constraints (~ 8e-5 Wb/m^2)

%% 1: My first FBT equilibrium
% Let's make our first equilibrium by specifying three points on which we want 
% the flux value to be the same. We specify this as a cost function term. We use 
% the auxiliary function  |fbtgp| to populate these parameters.

help fbtgp
%% 
% We specify a set of points on a circle, which will correspond to a set of
% cost function terms encouraging the flux to be equal on all these points

th = (1:10)'/10*2*pi;
a0 = 0.4;
rr = 1 + a0.*cos(th);
zz =   + a0.*sin(th);
gpb  = 1; % the points are on the main plasma domain boundary
gpfa = 0; % relative flux offset 
gpfb = 1; % Scale w.r.t. boundary flux
gpfe = 1; % These are all cost function terms and not equality constraints

% assign to P structure
LX = LX0; % init
LX = fbtgp(LX,rr,zz,gpb,gpfa,gpfb,gpfe,[],[],[],[],[],[],[],[],[],[],[],[]);

% check and set defaults
LX = fbtx(L,LX);
%% 
% The resulting |LX| structure contains the cost function and constraint terms 
% that will be used in the equilibrium calculation in |fbtt|. Its contents can 
% be displayed using |fbtxdisp|

fbtxdisp(L,LX);  
%% 
% We now run the optimization and plot the result.

LY  = fbtt(L,LX);
clf; fbtplot(L,LX,LY);snapnow;

%% 
% Note that the plasma boundary flux |FB| is not necessarily the same as the 
% fitting parameter |Fb| because none of the control points ended up being on the LCFS

disp([LY.Fb, LY.FB])
%% 
% Store this for later

LX1 = LX;

%% 2: Force points to be exactly on the boundary
% To force a point to be on the boundary two conditions must be met:
%  * The point is the one defining the separatrix, e.g. a limiter point or
%    an x point
%  * The constraints must be exact.
%
% We set an equality constraint by |gpfe=0|.

LX = LX1; % init
LX = fbtgp(LX,0.55,0,1,0,1,0,[],[],[],[],[],[],[],[],[],[],[],[]);
LX = fbtx(L,LX); % check and add defaults

fbtxdisp(L,LX);
LY  = fbtt(L,LX);

clf;fbtplot(L,LX,LY);snapnow;

disp([LY.Fb, LY.FB])

%% 3: Constrain Flux value on one point
% We may want to impose the absolute value of the flux at one or more points. 
% We can do this by setting |gpfb=0| for that point. In this example we set the 
% absolute value for the limiter point flux. In order to make sure the other points 
% have the same flux, we also set a constraint with |gpfb=1| on the same r,z point. 
% So the same point appears twice in the |fbtgp| call.

FBB = 0.2; % target boundary flux value                          
LX = LX1; % init
%             r   ,z,b,fa ,fb,fe,br,bz,ba,be,cr,cz,ca,ce,vrr,vrz,vzz,ve
LX = fbtgp(LX,0.55,0,1,FBB, 0, 0,[],[],[],[],[],[],[],[], [], [], [],[]);
LX = fbtgp(LX,0.55,0,1,  0, 1, 0,[],[],[],[],[],[],[],[], [], [], [],[]);
LX = fbtx(L,LX); % check and add defaults

fbtxdisp(L,LX);
LY  = fbtt(L,LX);

clf;fbtplot(L,LX,LY);snapnow;
disp([LY.Fb, LY.FB])
LX3 = LX; % store for later

%% 4: Constrain one current value
% We can constrain one current value by setting |gpie=0| and imposing the value 
% via |gpia|

LX = LX1; % Start again from case 1. 

ia_set = 2; % index of coil current to set
LX.gpie(ia_set) = 0;     % Set Ia(2)=gpia(2) to be an equality constraint
LX.gpia(ia_set) = +0e3; % Assign target value

LX = fbtx(L,LX); % check and add defaults
fbtxdisp(L,LX);
% note that 1/w for PF_002 is zero.

LY  = fbtt(L,LX);
% Check result
fprintf('Ia(2)=%2.2g',LY.Ia(ia_set))

clf;fbtplot(L,LX,LY);snapnow;

%% 5: X point(s)
% We now generate a new point distribution, where two points are X points with Br=Bz=0. 
% One of them is set to be on the same surface as the other LCFS point, while 
% one is given an offset. The one with an offset is assigned gpb=0
LX = LX0;

% Elongated distribution of points
a0 = 0.3; kap = 1.3; % minor radius, elongation
th = (1:10)'/10*2*pi+0.2;
rr = 1    +     a0.*cos(th);
zz = 0.05 + kap*a0.*sin(th);

gpb  = 1; % the points are on the main plasma domain boundary
gpfa = 0; % relative flux offset 
gpfb = 1; % Scale w.r.t. boundary flux
gpfe = 1; % These are all cost function terms and not equality constraints

% assign to P structure
% Elongated set of points
%               r,    z,  b,  fa,  fb,  fe,  br,bz,ba,be,  cr,cz,ca,ce,vrr,vrz,vzz,ve)
LX = fbtgp(LX, rr,   zz,gpb,gpfa,gpfb,gpfe,  [],[],[],[],  [],[],[],[], [], [], [],[]);
% Primary X-point with exact flux constraint
LX = fbtgp(LX,0.8,-0.35,  1,   0,   1,   0,   0, 0,[], 0,  [],[],[],[], [], [], [],[]);

% Second x-point outside vacuum vessel, at specified flux offset w.r.t. primary x point 
%                r,  z, b, fa,fb,fe,  br,bz,ba,be,  cr,cz,ca,ce,vrr,vrz,vzz,ve)
FFB= -0.02; % flux ofset value
LX = fbtgp(LX,0.95,0.6, 0,FFB, 1, 0,   0, 0,[], 0,  [],[],[],[], [], [], [],[]);

LX = fbtx(L,LX); % check and add defaults
fbtxdisp(L,LX);

LY  = fbtt(L,LX);
LX5 = LX; % save for later
clf;fbtplot(L,LX,LY);snapnow;

%% 6: Add a strike point
% A strike point has the same flux as the other LCFS points, but is not on the 
% main plasma domain boundary

% assign to P structure
LX = LX5; % init
LX = fbtgp(LX,0.82,-0.45,0,0,1,0,[],[],[],[],[],[],[],[],[],[],[],[]);
LX = fbtx(L,LX); % check and add defaults
fbtxdisp(L,LX);
LY  = fbtt(L,LX);

clf;fbtplot(L,LX,LY);snapnow;

LX6 = LX; % save for later

%% 7: Add magnetic field angle
% We now force the angle of the magnetic field on the strike point
% To allow this constraint to be satisfied exctly we also release the
% equality constraint on the x point.

LX = LX6; % start from example 6

rs=LX.gpr(end); zs=LX.gpz(end); % strike point r,z
% Add another constraint on the last point, to force the magnetic field angle value
%              r, z, b,  fa,  fb,  fe,  br,  bz,        ba,    be,cr,cz,ca,ce,vrr,vrz,vzz,ve
LX = fbtgp(LX,rs,zs, 0, NaN, NaN, Inf,  NaN,NaN,   0.45*pi,     0,[],[],[],[], [], [], [],[]);
LX = fbtx(L,LX); % check and add defaults

ix=(LX.gpbr==0) | LX.gpbz==0; % indices of x point field constriants
LX.gpbe(ix)=1; % Set cost function term instead of equality constraint
fbtxdisp(L,LX);
LY  = fbtt(L,LX);

clf;fbtplot(L,LX,LY);snapnow;

%% 8: Add flux expansion in radial direction

LX=LX6;
% Add constraint for psirr (d2/dr2 Psi) at the strike point point
%             r,  z,b,fa ,fb ,fe, br, bz, ba, be,  cr, cz, ca,ce,vrr,vrz,vzz,ve
LX = fbtgp(LX,rs,zs,0,[] ,[], [] ,[] ,[] ,[] ,[],-1.2,NaN,NaN, 0, [], [], [],[]);
LX = fbtx(L,LX); % check and add defaults

fbtxdisp(L,LX);
LY  = fbtt(L,LX);
clf;fbtplot(L,LX,LY);snapnow;

%%
% Note how this yields tighter flux surfaces around the divertor and
% removes the second x point under the vessel w.r.t. case 7.

%% 9: Zero hessian at X point (snowflake)

LX=LX0; % Start from scratch

% Elongated distribution of points
a0 = 0.25; kap = 1.5; % minor radius, elongation
nt = 10;
th = (1:nt)'/nt*2*pi+0.2;
rr = 1    +     a0.*cos(th);
zz = 0.02 + kap*a0.*sin(th);

%             [r, z   b ,fa, fb,fe,       br, bz, ba, be, cr, cz,ca, ce ,vrr,vrz,vzz,ve]
LX = fbtgp(LX,rr,zz,gpb,gpfa,gpfb,gpfe,   [],[],[],[],[],[],[],[],[],[],[],[]);

% Primary monkey-saddle-point (B=gradB=0) with exact flux constraint
% Add constraint for Br,Bz to be both zero (exact)
% Add constraint for psirz and psirr to be both zero at that point (exact)
%                r,    z,b,fa,fb,fe,   br,bz,ba,be,    cr,cz,ca,ce   ,vrr,vrz,vzz,ve
LX = fbtgp(LX,0.84,-0.48,1, 0, 1, 0,    0 ,0,[], 0,     0, 0,[], 0   , [], [], [],[]);
LX = fbtx(L,LX); % check and add defaults

fbtxdisp(L,LX);
LY  = fbtt(L,LX);
clf;fbtplot(L,LX,LY);snapnow;

%%
% Two observations:
%
% * It is enough to constrain psirr,psirz to be 0 to ensure that gradB=0 because 
%   the grad-shafranov equation will give us the final constraint psizz=0 
%   if we assume a zero current density at the X-point. This is ensured by 
%   this particular choice of basis functions, but may not hold in general.
% * The resulting equilibria will possibly not have an exact monkey point 
%   because of the difference in the way we impose the contraints 
%   (expressing psi as a combination of green functions) and the way we 
%   actually solve for psi (by inverting the GS operator).

%% 10: Vacuum field curvature

LX = LX3; % start from limited equilibrium
LX.gpvd = 1;
%             r,z, b, fa, fb, fe, br, bz, ba, be, cr, cz, ca, ce,vrr,  vrz,vzz,ve
LX = fbtgp(LX,1,0, 0, [], [], [], [], [], [], [], [], [], [], [], [],+0.05,[] , 0);
LX = fbtx(L,LX); % check and add defaults

fbtxdisp(L,LX);

LY  = fbtt(L,LX);

clf;
subplot(121)
fbtplot(L,LX,LY);

subplot(122)
% calculate and plot resulting vacuum field
meqplott(L,LY); hold on;
L.G = meqg(L.G,L.P,'Brxa','Bzxa');
LY.Br0x = reshape(L.G.Brxa*LY.Ia,L.nzx,L.nrx);
LY.Bz0x = reshape(L.G.Bzxa*LY.Ia,L.nzx,L.nrx);
LY.F0x  = reshape(L.G.Mxa *LY.Ia,L.nzx,L.nrx);
meqplotfield(L,LY,'vacuum',true)
snapnow

%% 11: Apply current limits
LX = LX6;
L.P.limu =  300e3;
L.P.liml = -300e3;
L.P.limm = 0.95*ones(L.G.na,1); % margin
L.P.limc = eye(L.G.na);
LX.gpfe(:) = 1; % make flux constraints non-exact
LX = fbtx(L,LX); % check and add defaults

fbtxdisp(L,LX);

LY  = fbtt(L,LX);

LX11 = LX; % save this

clf;
subplot(1,3,[1,2])
fbtplot(L,LX,LY);

subplot(1,3,3)
barh(LY.Ia/1e3); hold on;
plot(L.P.limm*L.P.limu*[1 1]/1e3,[1,L.G.na]);
plot(L.P.limm*[1,1]*L.P.liml/1e3,[1,L.G.na]);
set(gca,'xlim',500*[-1 1]); xlabel('kA'); 
set(gca,'TickLabelInterpreter','none');
aa=set(gca,'YTickLabel',L.G.dima);
title('Coil currents and limits')
snapnow;
% Notice how this makes a less shaped plasma at the expense of less
% accurate flux point fitting

%% Bonus: Internal profiles constraints in FBT
% In FBT, the internal profiles are fully constrained by a set of
% (potentially non-linear) constraints as set by the |fbtagcon| parameter.
% This parameter is analogous to the |agcon| parameter for FGS and FGE.
% As many constraints as basis functions must be provided.

% For example the default settings will use 3 basis functions and constrain
% the values of Ip, Wk and qA.
disp(L.ng);disp(L.P.fbtagcon)
% Note that the value of Wk is computed using bp and Ip, so that the user 
% only needs to provide the values of Ip, bp and qA

%%
% Three degrees of freedom in the basis functions are assumed, by default
% the basis function |bfab| has 1 (linear in psiN) basis function for p' 
% and 2 basis functions a (linear+quadratic in psiN) for TT'. 
% These are reflected in |bfp|
help(func2str(L.bfct))
disp(L.bfp)

%%
% We now scan qA and bp
LX=LX5;

ii=1;
clf;
Ip=150e3;
for bp = [0 1 2]
  for qA = [0.5 1 2]
    subplot(3,3,ii)
    LX.Ip  = Ip;
    LX.bp  = bp;
    LX.qA = qA;
    LX = rmfield(LX,{'Wk'}); % removing it causes it to be recomputed from `Ip`,`bp` in `fbtx`
    LX = fbtx(L,LX); % check and add defaults
    LY(ii) = fbtt(L,LX);
    fbtplot(L,LX,LY(ii));
    legend('off');
    title(sprintf('Ip=%3.0f[kA]\nqA=%2.1f, bp=%3.1f',LY(ii).Ip/1e3,LY(ii).qA,LY(ii).bp));
    ii=ii+1;
  end
end
snapnow;
%%
% For a few of these, let's also plot the internal profiles
LYcell=num2cell(LY(7:9));
clf;
meqplotQ(L,LYcell{:})
snapnow;

%% Bonus 2: Custom basis functions
% We can use bf3imex to specify custom basis functions:
n=21;
GN = [linspace(1,0,n)',linspace(1,0,n)',zeros(n,1)];
GN(end-2:end,1) = 1; % add a little pressure pedestal
IGN = bfprmex(GN); % integrate to get IGN
FP = [1;0;0]; FT = [0;1;0]; % first BF applies to p', second to TT', third to none.

% define parameters
bfct = @bf3imex;
bfp = struct('gNg',GN,'IgNg',IGN,'fPg',FP,'fTg',FT)';

fbtagcon = {'Ip','Wk','ag'}; % third one will be ag=0 for the third, unused basis function
[L,LX,LY] = fbt('ana',1,0,'bfct',bfct,'bfp',bfp,'fbtagcon',fbtagcon);

meqplotQ(L,LY);

