% EXPORT_PI_TEST_VECTORS  Reference run for the Python plant parity test (P10).
%   One full run with disturbance and a parameter step, written to pi/test_vectors/plant_run.mat.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
p0 = change_detector.nominalParams();
pA = change_detector.applyScenario(p0, struct('addedMassKg', 1500, 'mountXM', -1.2, 'kf', 0.8));
prof = change_detector.makeManeuverProfile(struct(), 3);
dist = change_detector.makeDisturbance(prof.t, struct(), 1003);
sched = struct('tStart', 100, 'tEnd', 100);
out = change_detector.simulatePlant(prof, dist, p0, pA, sched);

vec = @(p) [p.m p.lf p.lr p.Izz p.Caf p.Car p.sigmaF];
ref = struct('t', prof.t, 'delta', prof.delta, 'Vx', prof.Vx, 'Fyd', dist.Fyd, 'Mzd', dist.Mzd, ...
    'p0', vec(p0), 'p1', vec(pA), 'tStart', sched.tStart, 'tEnd', sched.tEnd, 'imuX', p0.imuXFromRearAxleM, ...
    'r', out.r, 'ay', out.ay, 'ydot', out.ydot, 'alphaF', out.alphaF, 'X', out.X, 'Y', out.Y, 'psi', out.psi, ...
    'params', out.params); %#ok<NASGU>
folder = fullfile(projectRoot, 'pi', 'test_vectors');
if ~isfolder(folder), mkdir(folder); end
save(fullfile(folder, 'plant_run.mat'), '-struct', 'ref', '-v7');
fprintf('wrote %s (%d samples)\n', fullfile(folder, 'plant_run.mat'), numel(prof.t));
