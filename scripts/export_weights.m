% EXPORT_WEIGHTS  models/trained.mat -> models/export/weights.mat (contract 3), then check that
%   change_detector.mlpForward reproduces predict of the MATLAB models on 100 val turns.
%   Also writes pi/test_vectors/model_forward.mat (those 100 feature rows and expected outputs).
%   The two regressors (dm in tonnes, kf) are stacked block-diagonally into one network with
%   2 outputs [dm kg; kf]; the tonne-to-kg factor is folded into the output layer.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
T = load(fullfile(projectRoot, 'models', 'trained.mat'));
cls = T.cls; rDm = T.rDm; rKf = T.rKf;
assert(numel(cls.LayerWeights) == 3 && numel(rDm.LayerWeights) == 3 && numel(rKf.LayerWeights) == 3, ...
    'export expects 2 hidden layers');
assert(all(strcmp(cls.Activations, 'relu')) && all(strcmp(rDm.Activations, 'relu')) && all(strcmp(rKf.Activations, 'relu')));

W = struct();
W.mu = T.mu; W.sigma = T.sigma;
W.featureNames = strjoin(T.featureNames, ',');
W.classNames = strjoin(cellstr(T.classNames), ',');
[~, ord] = ismember(T.classNames, string(cls.ClassNames));   % output rows in classNames order
W.W1 = cls.LayerWeights{1}; W.b1 = cls.LayerBiases{1};
W.W2 = cls.LayerWeights{2}; W.b2 = cls.LayerBiases{2};
W.W3 = cls.LayerWeights{3}(ord, :); W.b3 = cls.LayerBiases{3}(ord);

W.V1 = [rDm.LayerWeights{1}; rKf.LayerWeights{1}];  W.c1 = [rDm.LayerBiases{1}; rKf.LayerBiases{1}];
W.V2 = blkdiag(rDm.LayerWeights{2}, rKf.LayerWeights{2}); W.c2 = [rDm.LayerBiases{2}; rKf.LayerBiases{2}];
W.V3 = blkdiag(1000*rDm.LayerWeights{3}, rKf.LayerWeights{3}); W.c3 = [1000*rDm.LayerBiases{3}; rKf.LayerBiases{3}];

outDir = fullfile(projectRoot, 'models', 'export');
if ~isfolder(outDir), mkdir(outDir); end
save(fullfile(outDir, 'weights.mat'), '-struct', 'W', '-v7');

%% Forward-pass check on 100 val turns
S = load(fullfile(projectRoot, 'data', 'features.mat'));
[~, cols] = ismember(T.featureNames, S.featureNames);
va = S.F(S.F.split == "val" & S.F.trueTurn > 0, :);
X = va.X(1:min(100, height(va)), cols);
Z = (X - T.mu)./T.sigma;
[~, post] = predict(cls, Z);
Pref = post(:, ord);
dmRef = 1000*predict(rDm, Z); kfRef = predict(rKf, Z);
Wl = load(fullfile(outDir, 'weights.mat'));
[P, dm, kf] = change_detector.mlpForward(X, Wl);
err = [max(abs(P - Pref), [], 'all'), max(abs(dm - dmRef))/1000, max(abs(kf - kfRef))];
fprintf('mlpForward vs predict on %d val turns: max |dP| %.1e, max |ddm| %.1e t, max |dkf| %.1e\n', size(X, 1), err);
assert(all(err < 1e-5), 'forward-pass check failed');
fprintf('wrote models/export/weights.mat: %d features, classifier %s, regressor %s. PASS\n', numel(T.featureNames), ...
    mat2str([size(W.W1, 1) size(W.W2, 1)]), mat2str([size(W.V1, 1) size(W.V2, 1)]));

%% Test vector for the Python forward pass (pi/model.py): raw features in, expected outputs
tv = struct('X', X, 'P', Pref, 'dm', dmRef, 'kf', kfRef, 'runId', va.runId(1:size(X, 1)), ...
    'featureNames', W.featureNames, 'classNames', W.classNames);
tvDir = fullfile(projectRoot, 'pi', 'test_vectors');
if ~isfolder(tvDir), mkdir(tvDir); end
save(fullfile(tvDir, 'model_forward.mat'), '-struct', 'tv', '-v7');
fprintf('wrote pi/test_vectors/model_forward.mat (%d feature rows)\n', size(X, 1));
