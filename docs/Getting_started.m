%% Getting Started with MEQ
%
% Check out the <matlab:web('../README.md') README.md> file for general
% setup and compilation instructions.
%
%% Tutorials
% Several <matlab:web(fullfile(docroot,'3ptoolbox','meqtoolbox','doc','Tutorials','Tutorials.html')) tutorials> are available that can be
% run either as live scripts or as |.m| files.
%
%% General help
% Help on output naming:
%
% * <matlab:doc('meqt.m') General equilibrium quantities>
% * <matlab:doc('fbtt.m') FBT specific output naming>
% * <matlab:doc('fgst.m') FGS specific output naming>
% * <matlab:doc('fget.m') FGE specific output naming)>
% * <matlab:doc('rzpt.m') RZP specific output naming)>
%
% [+MEQ MatlabEQuilibrium Toolbox+]

%    Copyright 2022-2025 Swiss Plasma Center EPFL
%
%   Licensed under the Apache License, Version 2.0 (the "License");
%   you may not use this file except in compliance with the License.
%   You may obtain a copy of the License at
%
%       http://www.apache.org/licenses/LICENSE-2.0
%
%   Unless required by applicable law or agreed to in writing, software
%   distributed under the License is distributed on an "AS IS" BASIS,
%   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
%   See the License for the specific language governing permissions and
%   limitations under the License.

%%
% Help on parameter naming:
%
% * <matlab:doc('meqp.m') General parameter naming>
% * <matlab:doc('fbtp.m') FBT parameter naming>
% * <matlab:doc('fgsp.m') FGS parameter naming>
% * <matlab:doc('fgep.m') FGE parameter naming>
% * <matlab:doc('rzpp.m') RZP parameter naming>

%%
% Get help on variable naming scheme:
%
% * <matlab:doc('meqhelp') meq help>
% * <matlab:doc('bfhelp') basis function help>

%% Understanding meq's function naming scheme
% Codes:
% 
% * <matlab:doc('fbt') fbt> Inverse equilibrium problem: given constraints and cost function in terms of $\psi$, B, coil (etc), find coil currents |Ia|.
% * <matlab:doc('liu') liu> Equilibrium reconstruction: given measurements, find $\psi(R,Z)$
% * <matlab:doc('fgs') fgs> Forward solver: given |Ia|, |Iv|, |p'|, |TT'| (or integral parameters), find $\psi(R,Z)$
% * <matlab:doc('fge') fge> Forward evolutive solver: given |Va(t)|,
% integral parameters (t), find |Ia(t)|, Iv(t), $\psi(R,Z,t)$
% * <matlab:doc('meq') meq> Generic functions for all three codes
%
% Tokamaks |[tok]|:
%
% * |tcv| : TCV tokamak 
% * |rfx| : RFX-mod
% * |create| : CREATE ITER definition files
% * |ana| : tokamak with user-specified geometry (through analytical formulas) for stand-alone tests
%
% Functions naming scheme:
%
% * |[code].m| : Main calling function
% * |[code]x.m|: Retrieval of data
% * |[code]g.m|: Code specific geometry calculations
% * |[code]c.m|: Consolidate parameters, geometry and ancillary data for code
% * |[code]t.m|: Solver / time stepper
% * |[code]x[tok].m|: retrieve data for specific tokamak
% * |[code]w[tok].m|: MDS writer for code output
% * |[code]g[tok].m|: Code-specific geometry calculations
% * |meqg[tok].m|: Geometry definition for given tokamak [tok]
% * |meqp[tok].m|: Main parameters of a given device
%
%% Main data structures
%
% * |L|: General parameters structure
% * |L.P|: User-set Parameters
% * |L.G|: Geometry parameters
% * |L.T|: Tunable parameters that only affect |[code]t.m| [not used yet]
% * |L.*|: Parameters computed from |L.P|, |L.G| by |[code]c.m|
% * |LX|: Input data structure
% * |LY|: Output data structure
