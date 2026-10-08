function export_phase3_replay_vectors()
%EXPORT_PHASE3_REPLAY_VECTORS Write MATLAB references for Phase 3 parity tests.
%   The generated detector_<case>.mat files contain the measured edge
%   channels, per-turn feature/model references, and the streaming verdict
%   sequence produced by detectorStep.

projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);

fixtureFolder = fullfile(projectRoot, 'pi', 'test_vectors');
weightsPath = fullfile(projectRoot, 'models', 'export', 'weights.mat');
if ~isfile(weightsPath)
    error('export_phase3_replay_vectors:MissingWeights', ...
        'Run export_weights before exporting Phase 3 vectors');
end
W = load(weightsPath);

cases = ["nominal", "A", "B", "AB"];
for name = cases
    source = load(fullfile(fixtureFolder, "features_" + name + ".mat"));
    ref = buildReference(source, W, char(name));
    save(fullfile(fixtureFolder, "detector_" + name + ".mat"), ...
        '-struct', 'ref', '-v7');
    fprintf('wrote detector_%s.mat (%d samples, %d verdicts)\n', ...
        name, size(source.meas, 1), numel(ref.turnIndex));
end

% Reuse the deterministic plant input from the P10 vector to create both
% transition types with MATLAB-generated measured signals.
plantPath = fullfile(fixtureFolder, 'plant_run.mat');
if isfile(plantPath)
    plant = load(plantPath);
    p0 = vectorToParams(plant.p0);
    p1 = vectorToParams(plant.p1);
    profile = struct('t', plant.t(:), 'delta', plant.delta(:), 'Vx', plant.Vx(:));
    disturbance = struct('Fyd', plant.Fyd(:), 'Mzd', plant.Mzd(:));

    schedules = {
        'step', struct('tStart', 100, 'tEnd', 100)
        'ramp', struct('tStart', 100, 'tEnd', 180)
    };
    for j = 1:size(schedules, 1)
        kind = schedules{j, 1};
        schedule = schedules{j, 2};
        output = change_detector.simulatePlant(profile, disturbance, p0, p1, schedule);
        measured = [profile.delta, profile.Vx, output.r];
        ref = buildReference(struct('meas', measured, 'fs', 100), W, kind);
        ref.tStart = schedule.tStart;
        ref.tEnd = schedule.tEnd;
        save(fullfile(fixtureFolder, "detector_" + kind + ".mat"), ...
            '-struct', 'ref', '-v7');
        fprintf('wrote detector_%s.mat (%d samples, %d verdicts)\n', ...
            kind, size(measured, 1), numel(ref.turnIndex));
    end
else
    warning('export_phase3_replay_vectors:MissingPlantVector', ...
        'Skipping step/ramp vectors because plant_run.mat is missing');
end
end

function ref = buildReference(source, W, caseName)
rawMeasured = double(source.meas(:, 1:3));
measured = rawMeasured;
fs = double(source.fs);
% Keep enough tail samples for the final turn's post-exit margin. The
% Python consumer can flush a pending turn, while detectorStep is driven by
% samples only; repeating the final sample makes both paths observe the same
% emission point without changing any feature window.
tailSamples = round(5 * fs) + 1;
measured = [measured; repmat(measured(end, :), tailSamples, 1)];
if isfield(source, 'P')
    P = double(source.P);
    regression = double(source.regression);
    selectedIdx = double(source.selectedIdx(:))';
    X = double(source.X(:, selectedIdx));
    turnData = source;
else
    cfg = change_detector.turnTriggerConfig();
    turns = change_detector.findTurns(rawMeasured(:, 1), rawMeasured(:, 3), fs, cfg);
    names = change_detector.selectedFeatureNames();
    allNames = change_detector.turnFeatureNames();
    [ok, selectedIdx] = ismember(names, allNames);
    if ~all(ok)
        error('export_phase3_replay_vectors:FeatureNames', ...
            'Selected feature names are missing from the full feature list');
    end
    X = zeros(numel(turns), numel(names));
    i0 = zeros(numel(turns), 1); i1 = i0; s0 = i0; s1 = i0;
    for j = 1:numel(turns)
        [win, straight, i0(j), i1(j), iS] = ...
            change_detector.turnWindow(rawMeasured, turns(j), fs, cfg);
        s0(j) = iS(1); s1(j) = iS(end);
        X(j, :) = change_detector.selectedTurnFeatures(win, straight, fs);
    end
    P = zeros(numel(turns), 4);
    regression = NaN(numel(turns), 2);
    for j = 1:numel(turns)
        prediction = change_detector.mlpForward(X(j, :), W);
        P(j, :) = prediction.classProbabilities;
        regression(j, :) = [prediction.deltaM_kg prediction.kF];
    end
    turnData = struct('iEntry', [turns.iEntry]', 'iExit', [turns.iExit]', ...
        'i0', i0, 'i1', i1, 's0', s0, 's1', s1, 'trigger', cfg);
end

% Run the streaming MATLAB detector to capture exact emission timing and
% aggregation semantics. The per-turn feature/model arrays above are kept
% separately so a failure can be localized to windowing or inference.
W.reset = true;
verdictP = zeros(size(P));
verdictDm = NaN(size(P, 1), 1);
verdictKf = NaN(size(P, 1), 1);
turnIndex = zeros(size(P, 1), 1);
timestampS = zeros(size(P, 1), 1);
classIndex = zeros(size(P, 1), 1);
newCount = 0;
for k = 1:size(measured, 1)
    [out, newVerdict] = change_detector.detectorStep(measured(k, :), W);
    W.reset = false;
    if newVerdict
        newCount = newCount + 1;
        verdictP(newCount, :) = out.classProbabilities;
        verdictDm(newCount) = out.deltaM_kg;
        verdictKf(newCount) = out.kF;
        turnIndex(newCount) = out.turnIndex;
        timestampS(newCount) = out.timestamp_s;
        classIndex(newCount) = out.classIndex;
    end
end
if newCount ~= size(P, 1)
    error('export_phase3_replay_vectors:VerdictCount', ...
        '%s emitted %d verdicts; expected %d', caseName, newCount, size(P, 1));
end

ref = struct( ...
    'schema_version', 1, ...
    'feature_version', 1, ...
    'model_version', double(W.model_version), ...
    'case_name', caseName, ...
    'meas', measured, ...
    'fs', fs, ...
    'X_selected', X, ...
    'per_turn_probabilities', P, ...
    'per_turn_regression', regression, ...
    'verdict_probabilities', verdictP, ...
    'verdict_deltaM_kg', verdictDm, ...
    'verdict_kF', verdictKf, ...
    'turnIndex', turnIndex, ...
    'timestamp_s', timestampS, ...
    'classIndex', classIndex, ...
    'selectedIdx', selectedIdx, ...
    'iEntry', double(turnData.iEntry(:)), ...
    'iExit', double(turnData.iExit(:)), ...
    'i0', double(turnData.i0(:)), ...
    'i1', double(turnData.i1(:)), ...
    's0', double(turnData.s0(:)), ...
    's1', double(turnData.s1(:)), ...
    'trigger', turnData.trigger);
end

function p = vectorToParams(v)
fields = {'m', 'lf', 'lr', 'Izz', 'Caf', 'Car', 'sigmaF'};
p = struct();
for k = 1:numel(fields)
    p.(fields{k}) = double(v(k));
end
p.imuXFromRearAxleM = 0;
end
