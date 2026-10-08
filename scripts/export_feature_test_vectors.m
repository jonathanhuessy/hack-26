% EXPORT_FEATURE_TEST_VECTORS  Reference data for the Python port of the turn trigger and
%   turnFeatures (Track C). Writes pi/test_vectors/features_<class>.mat (MATLAB v7, read with
%   scipy.io.loadmat), one file per class, each one full run with sensor noise:
%     meas        N x 4 [delta Vx r ay] measured signals (ay is not needed by the 18 features)
%     fs          sample rate [Hz]
%     iEntry, iExit, headingRad   findTurns result per turn (indices 1-based, as in MATLAB)
%     i0, i1      turn window rows (1-based, inclusive)
%     s0, s1      straight rows (1-based, inclusive, contiguous)
%     X           nTurns x 35 turnFeatures output
%     featureNames         comma-separated names of the 35 columns
%     selectedIdx          1-based columns of the 18 selected features
%     trigger              turnTriggerConfig values
%     bLp, aLp             trigger low-pass;  bBand, aBand: band-pass per row of bands
%     bands                band edges [Hz] used by bpE_* (rows 1-5) and bpS_1.0-2.0 (row 6)
%     bAR, aAR             band-pass 0.3-3 Hz used before the AR(2) fit
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
idx = load(fullfile(projectRoot, 'data', 'index.mat')).idx;
cfg = change_detector.turnTriggerConfig();
names = change_detector.turnFeatureNames();
[~, sel] = ismember(change_detector.selectedFeatureNames(), names);
folder = fullfile(projectRoot, 'pi', 'test_vectors');
if ~isfolder(folder), mkdir(folder); end

classes = ["nominal", "A", "B", "AB"];
for c = classes
    k = find(idx.class == c & idx.split == "val" & idx.changeType == "constant", 1);
    fixtureFile = fullfile(folder, "features_" + c + ".mat");
    sourceFile = fullfile(projectRoot, 'data', 'runs', idx.file(k));
    modelPath = fullfile(projectRoot, 'models', 'export', 'weights.mat');
    if ~isfile(sourceFile) && isfile(fixtureFile) && isfile(modelPath)
        existing = load(fixtureFile);
        W = load(modelPath);
        P = zeros(size(existing.X, 1), 4);
        regression = NaN(size(existing.X, 1), 2);
        for j = 1:size(existing.X, 1)
            prediction = change_detector.mlpForward(existing.X(j, existing.selectedIdx), W);
            P(j,:) = prediction.classProbabilities;
            regression(j,:) = [prediction.deltaM_kg prediction.kF];
        end
        verdictClassProbabilities = mean(P, 1);
        save(fixtureFile, 'P', 'regression', 'verdictClassProbabilities', '-append');
        fprintf('updated %s with model outputs\n', fixtureFile);
        continue
    end
    r = load(sourceFile);
    fs = r.meta.fs;
    tt = change_detector.addSensorNoise(r.tt, [], 5000 + idx.id(k));
    meas = double([tt.deltaMeas tt.VxMeas tt.rMeas tt.ayMeas]);
    turns = change_detector.findTurns(meas(:,1), meas(:,3), fs, cfg);

    n = numel(turns);
    X = zeros(n, numel(names)); i0 = zeros(n, 1); i1 = i0; s0 = i0; s1 = i0;
    for j = 1:n
        [win, straight, i0(j), i1(j), iS] = change_detector.turnWindow(meas, turns(j), fs, cfg);
        assert(isempty(iS) || all(diff(iS) == 1), 'straight of run %d turn %d is not contiguous', idx.id(k), j);
        s0(j) = iS(1); s1(j) = iS(end);
        X(j,:) = change_detector.turnFeatures(win, straight, fs);
    end
    P = []; regression = []; verdictP = [];
    if isfile(modelPath)
        W = load(modelPath);
        P = zeros(n, 4); regression = NaN(n, 2);
        for j = 1:n
            prediction = change_detector.mlpForward(X(j, sel), W);
            P(j,:) = prediction.classProbabilities;
            regression(j,:) = [prediction.deltaM_kg prediction.kF];
        end
        verdictP = mean(P, 1);
    end

    bands = [0.2 0.6; 0.6 1.0; 1.0 1.5; 1.5 2.0; 2.0 3.0; 1.0 2.0];
    bBand = zeros(size(bands, 1), 5); aBand = bBand;
    for b = 1:size(bands, 1)
        [bBand(b,:), aBand(b,:)] = butter(2, bands(b,:)/(fs/2), 'bandpass');
    end
    [bLp, aLp] = butter(2, cfg.lowpassHz/(fs/2));
    [bAR, aAR] = butter(2, [0.3 3]/(fs/2), 'bandpass');

    ref = struct('runId', idx.id(k), 'class', char(c), 'meas', meas, 'fs', fs, ...
        'iEntry', [turns.iEntry]', 'iExit', [turns.iExit]', 'headingRad', [turns.headingRad]', ...
        'i0', i0, 'i1', i1, 's0', s0, 's1', s1, 'X', X, ...
        'classProbabilities', P, 'regression', regression, 'verdictClassProbabilities', verdictP, ...
        'featureNames', strjoin(names, ','), 'selectedIdx', sel(:)', 'trigger', cfg, ...
        'bLp', bLp, 'aLp', aLp, 'bands', bands, 'bBand', bBand, 'aBand', aBand, 'bAR', bAR, 'aAR', aAR);
    file = fullfile(folder, "features_" + c + ".mat");
    save(file, '-struct', 'ref', '-v7');
    fprintf('wrote %s: run %d, %d turns\n', file, idx.id(k), n);
end
