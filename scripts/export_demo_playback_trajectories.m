function export_demo_playback_trajectories(names)
%EXPORT_DEMO_PLAYBACK_TRAJECTORIES Write full A/B/AB MATLAB playback files.
%   export_demo_playback_trajectories()
%   export_demo_playback_trajectories({'AB'})
%
% The files contain noisy measured signals for the Pi contract plus plant
% diagnostics and scenario truth for PC-side visualization. The edge stream
% remains the first three measured columns: [deltaMeas, VxMeas, rMeas].

if nargin < 1
    names = {'A', 'B', 'AB'};
end

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
outDir = fullfile(projectRoot, 'data', 'generated_trajectories', 'matlab');
if ~isfolder(outDir)
    mkdir(outDir);
end

noiseSeed = 777;
for k = 1:numel(names)
    scenario = change_detector.demoScenario(names{k});
    plant = change_detector.simulatePlant( ...
        scenario.prof, scenario.dist, scenario.pStart, scenario.pEnd, scenario.sched);
    fs = scenario.prof.fs;
    tt = timetable( ...
        seconds(scenario.prof.t), scenario.prof.delta, scenario.prof.Vx, ...
        plant.r, plant.ay, ...
        'VariableNames', {'delta', 'Vx', 'r', 'ay'});
    tt = change_detector.addSensorNoise(tt, [], noiseSeed);
    meas = double([tt.deltaMeas, tt.VxMeas, tt.rMeas, tt.ayMeas]);

    meta = struct( ...
        'scenario', scenario.name, ...
        'className', names{k}, ...
        'changeTurn', scenario.changeTurn, ...
        'changeTimeS', scenario.sched.tStart, ...
        'addedMassKg', scenario.dm, ...
        'kf', scenario.kf, ...
        'noiseSeed', noiseSeed);
    output = struct( ...
        'meas', meas, ...
        'fs', fs, ...
        'plant', plant, ...
        'meta', meta);
    path = fullfile(outDir, ['demo_' lower(names{k}) '.mat']);
    save(path, '-struct', 'output', '-v7');
    fprintf('wrote %s (%d samples, %.1f s)\n', path, size(meas, 1), size(meas, 1) / fs);
end
end
