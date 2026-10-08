% SELECT_FEATURES  Which turnFeatures to keep (end of H1).
%   Permutation importance of the 32-16 MLP on val (accuracy drop when one feature is
%   shuffled, mean of 3 seeds), then val accuracy of candidate subsets. Test stays untouched.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
S = load(fullfile(projectRoot, 'data', 'features.mat'));
F = S.F; names = string(S.featureNames);
F = F(F.changeType == "constant" & F.trueTurn > 0, :);
tr = F(F.split == "train", :); va = F(F.split == "val", :);
cn = ["nominal", "A", "B", "AB"];
yTr = categorical(cn(tr.cls + 1), cn)'; yVa = categorical(cn(va.cls + 1), cn)';
g = findgroups(va.runId); yRun = splitapply(@(y) y(1), yVa, g);

mu = mean(tr.X); sd = std(tr.X); sd(sd == 0) = 1;
imp = zeros(numel(names), 1);
for s = 1:3
    rng(s);
    m = fitcnet((tr.X - mu)./sd, yTr, 'LayerSizes', [32 16], 'Lambda', 1e-4);
    base = mean(predict(m, (va.X - mu)./sd) == yVa);
    for j = 1:numel(names)
        Xp = va.X; Xp(:, j) = Xp(randperm(size(Xp, 1)), j);
        imp(j) = imp(j) + (base - mean(predict(m, (Xp - mu)./sd) == yVa))/3;
    end
end
[~, o] = sort(imp, 'descend');
disp(table(names(o)', imp(o), 'VariableNames', {'feature', 'permImportance'}));

sets = {
    'all 35', names
    'top 20', names(o(1:20))
    'top 15', names(o(1:15))
    'top 12', names(o(1:12))
    'top 8', names(o(1:8))
    };
for k = 1:size(sets, 1)
    c = ismember(names, sets{k, 2});
    acc = evalMlp(tr.X(:, c), yTr, va.X(:, c), yVa, g, yRun, cn, 1:3);
    fprintf('%-8s (%2d features): per turn %.2f, per run %.2f\n', sets{k, 1}, nnz(c), acc);
end

function acc = evalMlp(Xtr, yTr, Xva, yVa, g, yRun, cn, seeds)
% Mean val accuracy per turn and per run (posterior averaged over the run's turns).
mu = mean(Xtr); sd = std(Xtr); sd(sd == 0) = 1;
acc = zeros(numel(seeds), 2);
for i = 1:numel(seeds)
    rng(seeds(i));
    m = fitcnet((Xtr - mu)./sd, yTr, 'LayerSizes', [32 16], 'Lambda', 1e-4);
    [p, post] = predict(m, (Xva - mu)./sd);
    pr = splitapply(@(q) mean(q, 1), post, g);
    [~, k] = max(pr, [], 2);
    acc(i, :) = [mean(p == yVa), mean(categorical(m.ClassNames(k), cn) == yRun)];
end
acc = mean(acc, 1);
end
