function out = simulatePlant(prof, dist, p0, p1, sched, fixedStep)
%SIMULATEPLANT Run models/tractor_plant.slx on one maneuver profile.
%   out = simulatePlant(prof, dist, p0, p1, sched, fixedStep)
%     prof   from makeManeuverProfile (t, delta, Vx)
%     dist   from makeDisturbance (Fyd, Mzd); [] for no disturbance
%     p0     parameters at the start; p1 parameters after the change (default p0)
%     sched  .tStart, .tEnd: linear blend from p0 to p1 (tEnd = tStart: step)
%     fixedStep  solver step in s (default 0.01; smaller only for verification)
%   Returns clean (noise-free) signals as columns on prof.t.

if nargin < 2 || isempty(dist)
    dist = struct('Fyd', zeros(size(prof.t)), 'Mzd', zeros(size(prof.t)));
end
if nargin < 4 || isempty(p1), p1 = p0; end
if nargin < 5 || isempty(sched), sched = struct('tStart', 0, 'tEnd', 0); end
if nargin < 6 || isempty(fixedStep), fixedStep = 0.01; end

mdl = 'tractor_plant';
if ~bdIsLoaded(mdl)
    load_system(fullfile(fileparts(fileparts(mfilename('fullpath'))), 'models', [mdl '.slx']));
end

vec = @(p) [p.m; p.lf; p.lr; p.Izz; p.Caf; p.Car; p.sigmaF];
setv = @(in, name, val) in.setVariable(name, val, 'Workspace', mdl);

in = Simulink.SimulationInput(mdl);
in = in.setExternalInput([prof.t, prof.delta, prof.Vx, dist.Fyd, dist.Mzd]);
in = in.setModelParameter('FixedStep', num2str(fixedStep));
in = setv(in, 'tStop', prof.t(end));
in = setv(in, 'p0', vec(p0));
in = setv(in, 'p1', vec(p1));
in = setv(in, 'tStart', sched.tStart);
in = setv(in, 'tEnd', sched.tEnd);
in = setv(in, 'imuXRearAxleM', p0.imuXFromRearAxleM);
in = setv(in, 'x0_plant', zeros(6, 1));

so = sim(in);
y = so.yout;

out.t = so.tout;
if fixedStep ~= 0.01   % bring verification runs back onto the profile grid
    idx = round(prof.t/fixedStep) + 1;
    y = y(idx, :);
    out.t = so.tout(idx);
end
out.delta = prof.delta; out.Vx = prof.Vx;
out.r = y(:,1); out.ay = y(:,2); out.ydot = y(:,3); out.alphaF = y(:,4);
out.X = y(:,5); out.Y = y(:,6); out.psi = y(:,7);
out.params = y(:,8:14);   % [m lf lr Izz Caf Car sigmaF]
end
