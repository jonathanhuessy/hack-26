% TRAIN_MODELS  Classifier and regressors on the 18 selected features (H2).
%   Train split for fitting, val split for choosing the network size and regularisation.
%   Test stays untouched until H3. Writes models/trained.mat (model objects, normalisation,
%   val results); scripts/export_weights.m turns it into models/export/weights.mat.
%   Classifier: fitcnet, 4 classes, scored per run (class probabilities averaged over the turns).
%   Regressors: one fitrnet for dm (trained on turns with A, target in tonnes) and one for kf
%   (trained on turns with B), scored per run (median over the turns), MAE.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
S = load(fullfile(projectRoot, 'data', 'features.mat'));
featureNames = change_detector.selectedFeatureNames();
[~, cols] = ismember(featureNames, S.featureNames);
F = S.F(S.F.changeType == "constant" & S.F.trueTurn > 0, :);
tr = F(F.split == "train", :); va = F(F.split == "val", :);
Xtr = tr.X(:, cols); Xva = va.X(:, cols);
classNames = ["nominal", "A", "B", "AB"];
yTr = categorical(classNames(tr.cls + 1), classNames)'; yVa = categorical(classNames(va.cls + 1), classNames)';

mu = mean(Xtr); sigma = std(Xtr); sigma(sigma == 0) = 1;
Ztr = (Xtr - mu)./sigma; Zva = (Xva - mu)./sigma;
[gVa, runVa] = findgroups(va.runId);
yRunVa = splitapply(@(y) y(1), yVa, gVa);

%% Classifier: small grid on val, per-run accuracy; smallest network within 0.01 of the best wins
layerGrid = {[16 8], [32 16], [64 32]};
lambdaGrid = [1e-4 1e-3 3e-3 1e-2];
seeds = 1:3;
cand = struct('layers', {}, 'lambda', {}, 'acc', {}, 'nParams', {}, 'mdl', {}, 'accSeed', {});
fprintf('Classifier (val accuracy per turn / per run, mean of %d seeds)\n', numel(seeds));
for L = layerGrid
    for lam = lambdaGrid
        acc = zeros(numel(seeds), 2); mdls = cell(numel(seeds), 1);
        for s = seeds
            rng(s);
            mdls{s} = fitcnet(Ztr, yTr, 'LayerSizes', L{1}, 'Lambda', lam, 'Standardize', false);
            [p, post] = predict(mdls{s}, Zva);
            acc(s, :) = [mean(p == yVa), runAccuracy(post, gVa, yRunVa, mdls{s}.ClassNames, classNames)];
        end
        fprintf('  layers %-8s lambda %.0e: %.3f / %.3f\n', mat2str(L{1}), lam, mean(acc));
        [~, sBest] = max(acc(:, 2));
        cand(end+1) = struct('layers', L{1}, 'lambda', lam, 'acc', mean(acc(:, 2)), ...
            'nParams', sum(cellfun(@numel, mdls{1}.LayerWeights)), 'mdl', mdls{sBest}, 'accSeed', acc(sBest, :)); %#ok<SAGROW>
    end
end
ok = find([cand.acc] >= max([cand.acc]) - 0.01);
[~, k] = min([cand(ok).nParams] - 1e-3*[cand(ok).acc]);
best = cand(ok(k));
cls = best.mdl;
fprintf('chosen: layers %s, lambda %.0e (mean per-run %.3f; kept seed: per turn %.3f, per run %.3f)\n', ...
    mat2str(best.layers), best.lambda, best.acc, best.accSeed);

%% Linear reference (if almost as good, prefer it)
rng(1);
lin = fitcecoc(Ztr, yTr, 'Learners', templateLinear('Learner', 'logistic', 'Lambda', 1e-4));
[pl, postl] = predict(lin, Zva);
[~, postl] = max(postl, [], 2);   % ecoc scores: take the argmax per turn, vote per run
linRun = mean(splitapply(@(k) mode(k), postl, gVa) == double(yRunVa));
fprintf('linear logistic reference: per turn %.3f, per run (majority vote) %.3f\n', mean(pl == yVa), linRun);

%% Regressors
selA = tr.cls == 1 | tr.cls == 3; selB = tr.cls == 2 | tr.cls == 3;
vA = va.cls == 1 | va.cls == 3;   vB = va.cls == 2 | va.cls == 3;
regGrid = {[16 8], [32 16]}; regLambda = [1e-3 1e-2 3e-2 1e-1];
[rDm, infoDm] = fitRegressor(Ztr(selA, :), tr.dm(selA)/1000, Zva(vA, :), va.dm(vA)/1000, va.runId(vA), regGrid, regLambda);
[rKf, infoKf] = fitRegressor(Ztr(selB, :), tr.kf(selB), Zva(vB, :), va.kf(vB), va.runId(vB), regGrid, regLambda);
fprintf('dm regressor: layers %s, lambda %.0e, val MAE per turn %.0f kg, per run %.0f kg\n', ...
    mat2str(infoDm.layers), infoDm.lambda, 1000*infoDm.maeTurn, 1000*infoDm.maeRun);
fprintf('kf regressor: layers %s, lambda %.0e, val MAE per turn %.3f, per run %.3f\n', ...
    mat2str(infoKf.layers), infoKf.lambda, infoKf.maeTurn, infoKf.maeRun);

%% Save
if ~isfolder(fullfile(projectRoot, 'models')), mkdir(fullfile(projectRoot, 'models')); end
results = struct('classifier', rmfield(best, 'mdl'), 'linearRun', linRun, 'dm', infoDm, 'kf', infoKf); %#ok<NASGU>
save(fullfile(projectRoot, 'models', 'trained.mat'), 'cls', 'rDm', 'rKf', 'mu', 'sigma', ...
    'featureNames', 'classNames', 'results');
fprintf('saved models/trained.mat\n');

function acc = runAccuracy(post, g, yRun, mdlClasses, classNames)
% Average the class probabilities over the turns of each run, then take the most likely class.
pr = splitapply(@(q) mean(q, 1), post, g);
[~, k] = max(pr, [], 2);
acc = mean(categorical(mdlClasses(k), classNames) == yRun);
end

function [mdl, info] = fitRegressor(Ztr, ytr, Zva, yva, runVa, layerGrid, lambdaGrid)
% fitrnet on a small grid, chosen by val MAE per run (median over the run's turns).
g = findgroups(runVa);
yRun = splitapply(@(y) y(1), yva, g);
info.maeRun = inf;
for L = layerGrid
    for lam = lambdaGrid
        rng(1);
        m = fitrnet(Ztr, ytr, 'LayerSizes', L{1}, 'Lambda', lam, 'Standardize', false);
        yh = predict(m, Zva);
        maeRun = mean(abs(splitapply(@median, yh, g) - yRun));
        if maeRun < info.maeRun
            mdl = m;
            info = struct('layers', L{1}, 'lambda', lam, 'maeTurn', mean(abs(yh - yva)), 'maeRun', maeRun);
        end
    end
end
end
