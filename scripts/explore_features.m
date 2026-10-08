% EXPLORE_FEATURES  First look at data/features.mat (H1): which features carry A and B,
%   and what accuracy simple models reach. Writes data/results/explore_features.png.
%   Uses train for fitting and val for scoring; test stays untouched until H3.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
S = load(fullfile(projectRoot, 'data', 'features.mat'));
F = S.F; names = string(S.featureNames);
F = F(F.changeType == "constant" & F.trueTurn > 0, :);
tr = F(F.split == "train", :); va = F(F.split == "val", :);
classNames = ["nominal", "A", "B", "AB"];
yTr = categorical(classNames(tr.cls + 1), classNames)'; yVa = categorical(classNames(va.cls + 1), classNames)';

%% Single-feature separation: AUC against nominal (0.5 = none, 1 = perfect)
auc = @(a, b) (sum(a(:) > b(:)', 'all') + 0.5*sum(a(:) == b(:)', 'all'))/(numel(a)*numel(b));
sep = @(a, b) max(auc(a, b), 1 - auc(a, b));
nom = tr.cls == 0;
isA = tr.cls == 1; isAbig = isA & tr.dm >= 1000; isB = tr.cls == 2; isAB = tr.cls == 3;
nF = numel(names);
A = zeros(nF, 4);
for j = 1:nF
    x = tr.X(:, j);
    A(j,:) = [sep(x(isA), x(nom)), sep(x(isAbig), x(nom)), sep(x(isB), x(nom)), sep(x(isA), x(isB))];
end
T = array2table(A, 'RowNames', names, 'VariableNames', {'A_vs_nom', 'Abig_vs_nom', 'B_vs_nom', 'A_vs_B'});
fprintf('Single-feature AUC (train turns), sorted by A vs nominal:\n');
disp(sortrows(T, 'A_vs_nom', 'descend'));

%% Feature ranking (minimum redundancy, maximum relevance) for the 4 classes
[rank, score] = fscmrmr(tr.X, yTr);
fprintf('fscmrmr top 15: %s\n', strjoin(names(rank(1:15)), ', '));

%% Quick baselines: per turn and per run (mean posterior over the turns of a run)
mu = mean(tr.X); sd = std(tr.X); sd(sd == 0) = 1;
z = @(X) (X - mu)./sd;
models = {
    'LDA',            @() fitcdiscr(z(tr.X), yTr, 'DiscrimType', 'pseudoLinear')
    'linear SVM/ECOC', @() fitcecoc(z(tr.X), yTr, 'Learners', templateLinear('Learner', 'logistic', 'Lambda', 1e-4))
    'MLP 32-16',      @() fitcnet(z(tr.X), yTr, 'LayerSizes', [32 16], 'Standardize', false, 'Lambda', 1e-4)
    'bagged trees',   @() fitcensemble(z(tr.X), yTr, 'Method', 'Bag', 'NumLearningCycles', 100)
    };
fprintf('\n%-16s %10s %10s\n', 'model', 'val/turn', 'val/run');
res = struct();
for m = 1:size(models, 1)
    rng(1);
    mdl = models{m, 2}();
    [pred, post] = predict(mdl, z(va.X));
    accTurn = mean(pred == yVa);
    [g, runIds] = findgroups(va.runId);
    postRun = splitapply(@(p) mean(p, 1), post, g);
    yRun = splitapply(@(y) y(1), yVa, g);
    [~, k] = max(postRun, [], 2);
    predRun = categorical(mdl.ClassNames(k), classNames);
    accRun = mean(predRun(:) == yRun(:));
    fprintf('%-16s %10.2f %10.2f\n', models{m, 1}, accTurn, accRun);
    res(m).name = models{m, 1}; res(m).predRun = predRun; res(m).yRun = yRun;
end

%% Regression check: how well do the features carry the magnitudes? (linear, val)
selA = tr.cls == 1 | tr.cls == 3; selB = tr.cls == 2 | tr.cls == 3;
bA = [ones(nnz(selA), 1) z(tr.X(selA, :))]\tr.dm(selA);
bB = [ones(nnz(selB), 1) z(tr.X(selB, :))]\tr.kf(selB);
vA = va.cls == 1 | va.cls == 3; vB = va.cls == 2 | va.cls == 3;
eA = [ones(nnz(vA), 1) z(va.X(vA, :))]*bA - va.dm(vA);
eB = [ones(nnz(vB), 1) z(va.X(vB, :))]*bB - va.kf(vB);
fprintf('\nlinear regression per turn (val, true class given): dm MAE %.0f kg (range 250-2000), kf MAE %.3f\n', ...
    mean(abs(eA)), mean(abs(eB)));

%% Plot: top features by class, and the confusion matrix of the best per-run model
outDir = fullfile(projectRoot, 'data', 'results');
if ~isfolder(outDir), mkdir(outDir); end
[~, best] = max(arrayfun(@(r) mean(r.predRun == r.yRun), res));
top = unique([rank(1:4), find(T.A_vs_nom == max(T.A_vs_nom), 1), find(T.B_vs_nom == max(T.B_vs_nom), 1)], 'stable');
fig = figure('Position', [100 100 1400 700]);
tl = tiledlayout(fig, 2, 4);
title(tl, 'H1 feature exploration (train turns, sensor noise added)');
for j = top(1:min(6, numel(top)))
    nexttile(tl);
    boxchart(yTr, tr.X(:, j)); title(names(j), 'Interpreter', 'none'); grid on;
end
nexttile(tl, [1 2]);
confusionchart(res(best).yRun, res(best).predRun, 'Title', sprintf('%s, val per run', res(best).name), ...
    'RowSummary', 'row-normalized');
exportgraphics(fig, fullfile(outDir, 'explore_features.png'), 'Resolution', 100);
