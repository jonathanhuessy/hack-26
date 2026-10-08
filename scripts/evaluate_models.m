% EVALUATE_MODELS  Final evaluation of the exported models on the test split and the demo runs (H3).
%   Uses models/export/weights.mat through change_detector.mlpForward (the deployed maths) and
%   change_detector.aggregateVerdict (class from the mean probabilities of the last K = 3 turns,
%   magnitudes from the median, gated by the class). Writes data/results/h3_test.png and
%   data/results/h3_demo.png and prints all numbers.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
W = load(fullfile(projectRoot, 'models', 'export', 'weights.mat'));
S = load(fullfile(projectRoot, 'data', 'features.mat'));
[~, cols] = ismember(strsplit(W.featureNames, ','), S.featureNames);
F = S.F;
[F.P, F.dmHat, F.kfHat] = change_detector.mlpForward(F.X(:, cols), W);
[~, k] = max(F.P, [], 2); F.clsHat = k - 1;
classNames = string(strsplit(W.classNames, ','));
K = 3;
outDir = fullfile(projectRoot, 'data', 'results');
if ~isfolder(outDir), mkdir(outDir); end
hasA = @(c) c == 1 | c == 3; hasB = @(c) c == 2 | c == 3;

%% Test split: per turn and per run
te = F(F.split == "test" & F.changeType == "constant", :);
runs = unique(te.runId);
R = table(runs, zeros(size(runs)), zeros(size(runs)), zeros(size(runs)), zeros(size(runs)), ...
    zeros(size(runs)), zeros(size(runs)), zeros(size(runs)), zeros(size(runs)), ...
    'VariableNames', {'runId', 'cls', 'dm', 'kf', 'clsHat', 'dmHat', 'kfHat', 'dmMed', 'kfMed'});
for i = 1:numel(runs)
    m = te.runId == runs(i);
    [R.clsHat(i), ~, R.dmHat(i), R.kfHat(i)] = change_detector.aggregateVerdict(te.P(m, :), te.dmHat(m), te.kfHat(m));
    R.cls(i) = te.cls(find(m, 1)); R.dm(i) = te.dm(find(m, 1)); R.kf(i) = te.kf(find(m, 1));
    R.dmMed(i) = median(te.dmHat(m)); R.kfMed(i) = median(te.kfHat(m));
end
fprintf('Test split: %d runs, %d turns\n', height(R), height(te));
fprintf('  accuracy 4 classes:      per turn %.3f, per run %.3f\n', mean(te.clsHat == te.cls), mean(R.clsHat == R.cls));
fprintf('  "something changed":     per turn %.3f, per run %.3f\n', mean((te.clsHat > 0) == (te.cls > 0)), mean((R.clsHat > 0) == (R.cls > 0)));
fprintf('  false alarms (nominal flagged as changed): per turn %.3f, per run %.3f (%d of %d runs)\n', ...
    mean(te.clsHat(te.cls == 0) > 0), mean(R.clsHat(R.cls == 0) > 0), nnz(R.clsHat(R.cls == 0) > 0), nnz(R.cls == 0));
fprintf('  missed changes (changed run called nominal): per run %.3f\n', mean(R.clsHat(R.cls > 0) == 0));
for c = 0:3
    fprintf('  recall %-7s per run %.3f\n', classNames(c + 1), mean(R.clsHat(R.cls == c) == c));
end

mA = hasA(R.cls); mB = hasB(R.cls);
okA = mA & hasA(R.clsHat); okB = mB & hasB(R.clsHat);
fprintf('  dm, true class given (median of turns): MAE %.0f kg over %d runs\n', mean(abs(R.dmMed(mA) - R.dm(mA))), nnz(mA));
fprintf('  dm, gated by the predicted class:       MAE %.0f kg over %d runs where A was detected (A missed in %d runs)\n', ...
    mean(abs(R.dmHat(okA) - R.dm(okA))), nnz(okA), nnz(mA & ~hasA(R.clsHat)));
fprintf('  kf, true class given:                   MAE %.3f over %d runs\n', mean(abs(R.kfMed(mB) - R.kf(mB))), nnz(mB));
fprintf('  kf, gated by the predicted class:       MAE %.3f over %d runs where B was detected (B missed in %d runs)\n', ...
    mean(abs(R.kfHat(okB) - R.kf(okB))), nnz(okB), nnz(mB & ~hasB(R.clsHat)));
edges = [250 500 1000 1500 2000];
for b = 1:4
    mm = mA & R.dm >= edges(b) & R.dm < edges(b + 1);
    fprintf('    dm %4d-%4d kg: %2d runs, A detected in %2d, MAE (true class given) %.0f kg\n', edges(b), edges(b + 1), ...
        nnz(mm), nnz(mm & hasA(R.clsHat)), mean(abs(R.dmMed(mm) - R.dm(mm))));
end

fig = figure('Position', [60 60 1500 800]);
tl = tiledlayout(fig, 2, 2);
title(tl, sprintf('H3: test split (%d runs, never used for training or tuning)', height(R)));
nexttile(tl);
confusionchart(categorical(classNames(te.cls + 1), classNames), categorical(classNames(te.clsHat + 1), classNames), ...
    'Title', sprintf('Per turn: %.0f %% correct', 100*mean(te.clsHat == te.cls)), 'RowSummary', 'row-normalized');
nexttile(tl);
confusionchart(categorical(classNames(R.cls + 1), classNames), categorical(classNames(R.clsHat + 1), classNames), ...
    'Title', sprintf('Per run (3 turns): %.0f %% correct', 100*mean(R.clsHat == R.cls)), 'RowSummary', 'row-normalized');
nexttile(tl); hold on; grid on;
scatter(R.dm(mA), R.dmMed(mA), 25, double(hasA(R.clsHat(mA))), 'filled');
colormap(gca, [0.85 0.3 0.1; 0 0.45 0.75]);
plot([0 2200], [0 2200], 'k--');
xlabel('true \Deltam [kg]'); ylabel('estimated \Deltam [kg] (median of 3 turns)');
title(sprintf('Added mass: MAE %.0f kg (red: A not detected)', mean(abs(R.dmMed(mA) - R.dm(mA)))));
nexttile(tl); hold on; grid on;
scatter(R.kf(mB), R.kfMed(mB), 25, double(hasB(R.clsHat(mB))), 'filled');
colormap(gca, [0.85 0.3 0.1; 0 0.45 0.75]);
plot([0.55 1.35], [0.55 1.35], 'k--');
xlabel('true k_f'); ylabel('estimated k_f (median of 3 turns)');
title(sprintf('Front tire stiffness factor: MAE %.3f (red: B not detected)', mean(abs(R.kfMed(mB) - R.kf(mB)))));
exportgraphics(fig, fullfile(outDir, 'h3_test.png'), 'Resolution', 110);

%% Demo runs: step changes and drift
de = F(F.split == "demo", :);
idx = load(fullfile(projectRoot, 'data', 'index.mat')).idx;
fprintf('\nDemo runs (verdict after each turn from the last up to %d turns of the same run):\n', K);
fprintf('%5s %-6s %-8s %-28s %-28s %s\n', 'run', 'type', 'reverse', 'true class per turn', 'verdict per turn', 'dm estimate per turn [kg] (true)');
demo = struct('runId', {}, 'type', {}, 'trueCls', {}, 'verdict', {}, 'clsTurn', {}, 'dmTrue', {}, 'dmTurn', {});
for id = unique(de.runId)'
    m = find(de.runId == id);
    row = idx(idx.id == id, :);
    v = zeros(numel(m), 1);
    for j = 1:numel(m)
        w = m(max(1, j - K + 1):j);
        v(j) = change_detector.aggregateVerdict(de.P(w, :), de.dmHat(w), de.kfHat(w));
    end
    fprintf('%5d %-6s %-8d %-28s %-28s %s\n', id, row.changeType, row.reverse, strjoin(classNames(de.cls(m) + 1), ' '), ...
        strjoin(classNames(v + 1), ' '), strjoin(compose('%.0f (%.0f)', [de.dmHat(m) de.dm(m)]), ', '));
    demo(end+1) = struct('runId', id, 'type', row.changeType, 'trueCls', de.cls(m), 'verdict', v, ...
        'clsTurn', de.clsHat(m), 'dmTrue', de.dm(m), 'dmTurn', de.dmHat(m)); %#ok<SAGROW>
end
isStep = string({demo.type}) == "step";
firstAfter = nan(2, numel(demo));
for i = find(isStep)
    c = find(demo(i).trueCls ~= demo(i).trueCls(1), 1);
    if ~isempty(c)
        firstAfter(:, i) = [demo(i).verdict(c); demo(i).clsTurn(c)] == demo(i).trueCls(c);
    end
end
fprintf('step runs, first turn after the change: aggregated verdict right in %d of %d, single-turn class right in %d of %d\n', ...
    nnz(firstAfter(1, :) == 1), nnz(~isnan(firstAfter(1, :))), nnz(firstAfter(2, :) == 1), nnz(~isnan(firstAfter(2, :))));

fig2 = figure('Position', [60 60 800 480]);
hold on; grid on;
ramp = find(~isStep);
for i = ramp
    p = plot(1:numel(demo(i).dmTrue), demo(i).dmTrue, '--', 'LineWidth', 1.5, 'DisplayName', sprintf('run %d truth', demo(i).runId));
    plot(1:numel(demo(i).dmTurn), demo(i).dmTurn, 'o-', 'Color', p.Color, 'LineWidth', 1.5, ...
        'MarkerFaceColor', p.Color, 'DisplayName', sprintf('run %d estimate', demo(i).runId));
end
xlabel('turn'); ylabel('\Deltam [kg]'); xticks(1:3); legend('Location', 'eastoutside');
title({'H3: drift runs (mounted tank draining or filling)', 'dashed = truth, solid with dots = estimate from that turn (ungated)'});
exportgraphics(fig2, fullfile(outDir, 'h3_demo.png'), 'Resolution', 110);
