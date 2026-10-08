function F = build_features(opts)
%BUILD_FEATURES One feature row per detected turn for all runs (H1) -> data/features.mat.
%   F = build_features()                         all runs in data/index.mat
%   F = build_features(struct('maxRuns', 40))    quick test on the first 40 runs
%   Sensor noise seed per run: 5000 + run id. Turns come from findTurns on the measured
%   signals only; labels are taken from the turn core (entry to exit) afterwards.
%   F columns: runId, split, class, changeType, turn (detected), trueTurn (turnIdx label in
%   the core, 0 if none), isOmega, cls, dm, kf, X (feature matrix, names in featureNames).

if nargin < 1, opts = struct(); end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
dataDir = fullfile(projectRoot, 'data');
idx = load(fullfile(dataDir, 'index.mat')).idx;
if isfield(opts, 'maxRuns'), idx = idx(1:min(opts.maxRuns, height(idx)), :); end
cfg = change_detector.turnTriggerConfig();
featureNames = change_detector.turnFeatureNames();
nF = numel(featureNames);

rows = cell(height(idx), 1);
nTurnsFound = zeros(height(idx), 1);
labelsMatch = false(height(idx), 1);
t0 = tic;
for i = 1:height(idx)
    r = load(fullfile(dataDir, 'runs', idx.file(i)));
    fs = r.meta.fs;
    tt = change_detector.addSensorNoise(r.tt, [], 5000 + idx.id(i));
    meas = double([tt.deltaMeas tt.VxMeas tt.rMeas tt.ayMeas]);
    turns = change_detector.findTurns(meas(:,1), meas(:,3), fs, cfg);
    n = numel(turns);
    nTurnsFound(i) = n;

    X = zeros(n, nF); trueTurn = zeros(n, 1); isOmega = false(n, 1);
    cls = zeros(n, 1); dm = zeros(n, 1); kf = zeros(n, 1);
    for j = 1:n
        [win, straight] = change_detector.turnWindow(meas, turns(j), fs, cfg);
        X(j,:) = change_detector.turnFeatures(win, straight, fs);
        c = turns(j).iEntry:turns(j).iExit;
        trueTurn(j) = mode(double(tt.turnIdx(c)));
        if trueTurn(j) > 0, isOmega(j) = r.meta.profile.turns(trueTurn(j)).isOmega; end
        cls(j) = mode(double(tt.cls(c)));
        dm(j) = median(double(tt.dm(c)));
        kf(j) = median(double(tt.kf(c)));
    end
    labelsMatch(i) = n == 3 && isequal(trueTurn(:)', 1:3);

    rows{i} = table(repmat(idx.id(i), n, 1), repmat(idx.split(i), n, 1), repmat(idx.class(i), n, 1), ...
        repmat(idx.changeType(i), n, 1), (1:n)', trueTurn, isOmega, cls, dm, kf, X, ...
        'VariableNames', {'runId', 'split', 'class', 'changeType', 'turn', 'trueTurn', 'isOmega', 'cls', 'dm', 'kf', 'X'});
    if mod(i, 100) == 0 || i == height(idx)
        fprintf('%5d/%d runs, %.0f s\n', i, height(idx), toc(t0));
    end
end
F = vertcat(rows{:});

fprintf('turn detection: %d of %d runs (%.1f %%) give exactly turns 1, 2, 3 of the label\n', ...
    nnz(labelsMatch), numel(labelsMatch), 100*mean(labelsMatch));
if any(~labelsMatch)
    fprintf('runs that do not: %s\n', mat2str(idx.id(~labelsMatch)'));
end
fprintf('non-finite feature values: %d\n', nnz(~isfinite(F.X)));

if ~isfield(opts, 'maxRuns')
    info = struct('cfg', cfg, 'noiseSeed', '5000 + run id', 'created', datetime('now'), ...
        'detectionRate', mean(labelsMatch)); %#ok<NASGU>
    save(fullfile(dataDir, 'features.mat'), 'F', 'featureNames', 'info');
    fprintf('saved %d turns to data/features.mat\n', height(F));
end
end
