function R = run_demo_replay(names)
%RUN_DEMO_REPLAY Offline reference of the full detector on the demo scenarios (H3 -> H4/H5).
%   R = run_demo_replay()                 scenarios 'A', 'B', 'AB'
%   R = run_demo_replay({'A'})
%   Per scenario: simulate (change after turn 2 of 6), add sensor noise, detect turns, compute
%   features, run the exported models (mlpForward) and the verdict rule (aggregateVerdict, last
%   K = 3 turns). Writes data/results/demo_<name>.png (truth vs estimate per turn) and the
%   detector reference pi/scenarios/demo_reference_<name>.mat, which Simulink and the
%   Pi must reproduce.

if nargin < 1, names = {'A', 'B', 'AB'}; end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
W = load(fullfile(projectRoot, 'models', 'export', 'weights.mat'));
allNames = change_detector.turnFeatureNames();
[~, sel] = ismember(strsplit(W.featureNames, ','), allNames);
classNames = string(strsplit(W.classNames, ','));
cfg = change_detector.turnTriggerConfig();
K = 3;
noiseSeed = 777;

for s = 1:numel(names)
    sc = change_detector.demoScenario(names{s});
    out = change_detector.simulatePlant(sc.prof, sc.dist, sc.pStart, sc.pEnd, sc.sched);
    fs = sc.prof.fs;
    tt = timetable(seconds(sc.prof.t), sc.prof.delta, sc.prof.Vx, out.r, out.ay, 'VariableNames', {'delta', 'Vx', 'r', 'ay'});
    tt = change_detector.addSensorNoise(tt, [], noiseSeed);
    meas = double([tt.deltaMeas tt.VxMeas tt.rMeas tt.ayMeas]);
    t = sc.prof.t;

    turns = change_detector.findTurns(meas(:,1), meas(:,3), fs, cfg);
    n = numel(turns);
    X = zeros(n, numel(allNames)); i1 = zeros(n, 1);
    for j = 1:n
        [win, straight, ~, i1(j)] = change_detector.turnWindow(meas, turns(j), fs, cfg);
        X(j,:) = change_detector.turnFeatures(win, straight, fs);
    end
    [P, dmTurn, kfTurn] = change_detector.mlpForward(X(:, sel), W);
    [~, k] = max(P, [], 2); clsTurn = k - 1;

    clsV = zeros(n, 1); pV = zeros(n, 4); dmV = zeros(n, 1); kfV = ones(n, 1);
    for j = 1:n
        w = max(1, j - K + 1):j;
        [clsV(j), pV(j,:), dmV(j), kfV(j)] = change_detector.aggregateVerdict(P(w,:), dmTurn(w), kfTurn(w));
    end

    changed = t([turns.iEntry]) >= sc.sched.tStart;
    dmTrue = changed(:)*sc.dm; kfTrue = 1 + changed(:)*(sc.kf - 1);
    clsTrue = double(dmTrue >= 250) + 2*double(abs(kfTrue - 1) >= 0.15);

    fprintf('\nDemo %s: %s, change after turn %d (t = %.0f s), %d turns detected\n', sc.name, ...
        describe(sc), sc.changeTurn, sc.sched.tStart, n);
    fprintf('%5s %8s %-8s %-8s %-8s %10s %10s %8s %8s\n', 'turn', 'time', 'truth', 'single', 'verdict', ...
        'dm true', 'dm est', 'kf true', 'kf est');
    for j = 1:n
        fprintf('%5d %7.0fs %-8s %-8s %-8s %10.0f %10.0f %8.2f %8.2f\n', j, t(i1(j)), classNames(clsTrue(j) + 1), ...
            classNames(clsTurn(j) + 1), classNames(clsV(j) + 1), dmTrue(j), dmV(j), kfTrue(j), kfV(j));
    end

    plotDemo(projectRoot, sc, out, turns, i1, t, classNames, clsTrue, clsTurn, clsV, dmTrue, dmTurn, dmV, kfTrue, kfTurn, kfV, K);

    ref = struct('scenario', sc.name, 'meas', meas, 'fs', fs, 'noiseSeed', noiseSeed, 'K', K, ...
        'iEntry', [turns.iEntry]', 'iExit', [turns.iExit]', 'iVerdict', i1, ...
        'X', X, 'selectedIdx', sel(:)', 'P', P, 'dmTurn', dmTurn, 'kfTurn', kfTurn, 'clsTurn', clsTurn, ...
        'clsVerdict', clsV, 'pVerdict', pV, 'dmVerdict', dmV, 'kfVerdict', kfV, ...
        'clsTrue', clsTrue, 'dmTrue', dmTrue, 'kfTrue', kfTrue, 'tChange', sc.sched.tStart);
    refDir = fullfile(projectRoot, 'pi', 'scenarios');
    if ~isfolder(refDir), mkdir(refDir); end
    save(fullfile(refDir, ['demo_reference_' sc.name '.mat']), '-struct', 'ref', '-v7');
    R.(sc.name) = ref;
end
end

function s = describe(sc)
parts = strings(0);
if sc.dm > 0, parts(end+1) = sprintf('+%d kg on the rear hitch', sc.dm); end
if sc.kf ~= 1, parts(end+1) = sprintf('front stiffness k_f = %.2f', sc.kf); end
s = strjoin(parts, ' and ');
end

function plotDemo(projectRoot, sc, out, turns, i1, t, classNames, clsTrue, clsTurn, clsV, dmTrue, dmTurn, dmV, kfTrue, kfTurn, kfV, K)
n = numel(turns); j = 1:n;
fig = figure('Position', [60 60 1500 850]);
tl = tiledlayout(fig, 3, 2);
title(tl, sprintf('Demo %s: %s after turn %d (red dashed line). Truth (black) vs estimate (colour)', sc.name, describe(sc), sc.changeTurn), ...
    'FontWeight', 'bold');

ax = nexttile(tl, 1, [3 1]); hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
before = t < sc.sched.tStart;
plot(ax, out.X(before), out.Y(before), 'Color', [0.6 0.6 0.6], 'LineWidth', 1.5, 'DisplayName', 'nominal tractor');
plot(ax, out.X(~before), out.Y(~before), 'Color', [0.85 0.33 0.1], 'LineWidth', 1.5, 'DisplayName', 'changed tractor');
for q = j
    c = round((turns(q).iEntry + turns(q).iExit)/2);
    text(ax, out.X(c), out.Y(c), sprintf(' %d', q), 'FontWeight', 'bold', 'FontSize', 11);
end
ic = find(~before, 1);
plot(ax, out.X(ic), out.Y(ic), 'kp', 'MarkerSize', 14, 'MarkerFaceColor', 'y', 'DisplayName', 'change happens');
xlabel(ax, 'X [m]'); ylabel(ax, 'Y [m]'); legend(ax, 'Location', 'best');
title(ax, 'Path (numbers: detected turns)');

ax = nexttile(tl, 2); hold(ax, 'on'); grid(ax, 'on');
stairs(ax, [j n+1] - 0.5, [clsTrue; clsTrue(end)], 'k-', 'LineWidth', 3, 'DisplayName', 'truth');
plot(ax, j, clsTurn, 'o', 'MarkerSize', 9, 'LineWidth', 1.5, 'DisplayName', 'estimate from this turn only');
plot(ax, j, clsV, 's', 'MarkerSize', 12, 'MarkerFaceColor', [0 0.45 0.74], 'DisplayName', sprintf('verdict (last %d turns)', K));
yticks(ax, 0:3); yticklabels(ax, classNames); ylim(ax, [-0.5 3.5]); xticks(ax, j); xlim(ax, [0.5 n+0.5]);
xline(ax, sc.changeTurn + 0.5, 'r--', 'HandleVisibility', 'off');
legend(ax, 'Location', 'southeast'); title(ax, 'What changed? (class per turn)');

ax = nexttile(tl, 4); hold(ax, 'on'); grid(ax, 'on');
stairs(ax, [j n+1] - 0.5, [dmTrue; dmTrue(end)], 'k-', 'LineWidth', 3, 'DisplayName', 'truth');
plot(ax, j, dmTurn, 'o', 'MarkerSize', 8, 'LineWidth', 1.2, 'DisplayName', 'estimate from this turn only (ungated)');
plot(ax, j, dmV, 's', 'MarkerSize', 11, 'MarkerFaceColor', [0 0.45 0.74], 'DisplayName', 'verdict (0 unless class has A)');
xticks(ax, j); xlim(ax, [0.5 n+0.5]); ylabel(ax, '\Deltam [kg]'); ylim(ax, [-100 2300]);
xline(ax, sc.changeTurn + 0.5, 'r--', 'HandleVisibility', 'off');
legend(ax, 'Location', 'northwest'); title(ax, 'How much mass was added on the rear hitch?');

ax = nexttile(tl, 6); hold(ax, 'on'); grid(ax, 'on');
stairs(ax, [j n+1] - 0.5, [kfTrue; kfTrue(end)], 'k-', 'LineWidth', 3, 'DisplayName', 'truth');
plot(ax, j, kfTurn, 'o', 'MarkerSize', 8, 'LineWidth', 1.2, 'DisplayName', 'estimate from this turn only (ungated)');
plot(ax, j, kfV, 's', 'MarkerSize', 11, 'MarkerFaceColor', [0 0.45 0.74], 'DisplayName', 'verdict (1 unless class has B)');
xticks(ax, j); xlim(ax, [0.5 n+0.5]); xlabel(ax, 'turn'); ylabel(ax, 'k_f = C_{af}/C_{af,nominal}'); ylim(ax, [0.5 1.4]);
xline(ax, sc.changeTurn + 0.5, 'r--', 'HandleVisibility', 'off');
legend(ax, 'Location', 'northeast'); title(ax, 'Front tire stiffness factor (1 = nominal)');

exportgraphics(fig, fullfile(projectRoot, 'data', 'results', ['demo_' sc.name '.png']), 'Resolution', 110);
end
