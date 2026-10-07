function [tt, meta] = generateRun(rc, p0)
%GENERATERUN Simulate one dataset run from a row of buildRunIndex (clean, no sensor noise).
%   tt    timetable at 100 Hz, single precision
%   meta  struct with the run config, profile info and parameter vectors
%
%   Per-sample labels: dm [kg] and kf (true values), cls (0 nominal, 1 A, 2 B, 3 A+B),
%   segment (1 straight, 2 speed transition, 3 turn), turnIdx, swath,
%   params = [m lf lr Izz Caf Car sigmaF].

if isa(rc, 'table'), rc = table2struct(rc); end

pBase = change_detector.applyScenario(p0, struct('soilCaf', rc.soilCaf, 'soilCar', rc.soilCar));
pChg = change_detector.applyScenario(p0, struct('addedMassKg', rc.dm, 'mountXM', rc.mountXM, ...
    'kf', rc.kf, 'soilCaf', rc.soilCaf, 'soilCar', rc.soilCar));

prof = change_detector.makeManeuverProfile(struct(), rc.profileSeed);
dist = change_detector.makeDisturbance(prof.t, struct(), rc.distSeed);

if strcmp(rc.changeType, 'constant')
    pStart = pChg; pEnd = pChg;
elseif rc.reverse
    pStart = pChg; pEnd = pBase;
else
    pStart = pBase; pEnd = pChg;
end
out = change_detector.simulatePlant(prof, dist, pStart, pEnd, struct('tStart', rc.tStart, 'tEnd', rc.tEnd));

% true values of the labelled properties over time
s = min(max((prof.t - rc.tStart)/max(rc.tEnd - rc.tStart, 1e-6), 0), 1);
if strcmp(rc.changeType, 'constant')
    active = ones(size(s));
elseif rc.reverse
    active = 1 - s;
else
    active = s;
end
dmT = active*rc.dm;
kfT = 1 + active*(rc.kf - 1);
cls = uint8((dmT >= 250) + 2*(abs(kfT - 1) >= 0.15));

tt = timetable(seconds(prof.t), single(prof.delta), single(prof.Vx), single(out.r), single(out.ay), ...
    single(out.ydot), single(out.alphaF), single(out.X), single(out.Y), single(out.psi), ...
    single(out.params), single(dmT), single(kfT), cls, prof.segment, prof.turnIdx, prof.swath, ...
    'VariableNames', {'delta', 'Vx', 'r', 'ay', 'ydot', 'alphaF', 'X', 'Y', 'psi', ...
    'params', 'dm', 'kf', 'cls', 'segment', 'turnIdx', 'swath'});

vec = @(p) [p.m p.lf p.lr p.Izz p.Caf p.Car p.sigmaF];
meta.run = rc;
meta.profile = prof.info;
meta.pStart = vec(pStart);
meta.pEnd = vec(pEnd);
meta.pNominal = vec(p0);
meta.paramNames = {'m', 'lf', 'lr', 'Izz', 'Caf', 'Car', 'sigmaF'};
meta.fs = prof.fs;
meta.matlabVersion = version;
end
