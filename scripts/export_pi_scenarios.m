% EXPORT_PI_SCENARIOS  Demo scenario inputs for pi/app.py (H5 step 5).
%   Writes pi/scenarios/scenario_<A|B|AB>.mat (MATLAB v7, read with scipy.io.loadmat):
%     t, delta, Vx, Fyd, Mzd   maneuver and disturbance at 100 Hz (inputs of pi/plant.py)
%     p0, p1                   plant parameters [m lf lr Izz Caf Car sigmaF] before / after the change
%     tStart, tEnd, imuX       change time (step: tStart = tEnd) and IMU position, as in plant.simulate
%     noise                    N x 4 additive sensor noise [delta Vx r ay], seed 777 as in demo_reference_*.mat
%     dmTrue, kfTrue, changeTurn, name   the true change, for display
%   clean plant output + noise reproduces the measured signals of pi/scenarios/demo_reference_<name>.mat
%   (up to single precision there).
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
outDir = fullfile(projectRoot, 'pi', 'scenarios');
if ~isfolder(outDir), mkdir(outDir); end
vec = @(p) [p.m p.lf p.lr p.Izz p.Caf p.Car p.sigmaF];

for name = ["A", "B", "AB"]
    sc = change_detector.demoScenario(char(name));
    ref = load(fullfile(outDir, "demo_reference_" + name + ".mat"), 'noiseSeed');
    t = sc.prof.t;
    z = zeros(size(t));
    nz = change_detector.addSensorNoise(timetable(seconds(t), z, z, z, z, 'VariableNames', {'delta', 'Vx', 'r', 'ay'}), [], ref.noiseSeed);
    s = struct('name', char(name), 't', t, 'delta', sc.prof.delta, 'Vx', sc.prof.Vx, 'Fyd', sc.dist.Fyd, 'Mzd', sc.dist.Mzd, ...
        'p0', vec(sc.pStart), 'p1', vec(sc.pEnd), 'tStart', sc.sched.tStart, 'tEnd', sc.sched.tEnd, ...
        'imuX', sc.pStart.imuXFromRearAxleM, 'noise', double([nz.deltaMeas nz.VxMeas nz.rMeas nz.ayMeas]), ...
        'dmTrue', sc.dm, 'kfTrue', sc.kf, 'changeTurn', sc.changeTurn);
    file = fullfile(outDir, "scenario_" + name + ".mat");
    save(file, '-struct', 's', '-v7');
    fprintf('wrote %s (%d samples, %.0f s, change at %.0f s)\n', file, numel(t), t(end), sc.sched.tStart);
end
