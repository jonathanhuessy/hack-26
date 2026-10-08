function plot_run(runId, noiseSeed)
%PLOT_RUN Visual check of the turn detection on one run.
%   plot_run(runId)              noise seed 5000 + runId, as in build_features
%   plot_run(runId, noiseSeed)
%   Top: XY path with the straight used for features, the window margins and the detected
%   turn (entry o, exit x). Below: steering angle with the trigger thresholds, yaw rate and
%   speed over time; the grey bands are the labelled turns (turnIdx), for comparison.

if nargin < 2, noiseSeed = 5000 + runId; end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
dataDir = fullfile(projectRoot, 'data');
idx = load(fullfile(dataDir, 'index.mat')).idx;
row = idx(idx.id == runId, :);
r = load(fullfile(dataDir, 'runs', row.file));
fs = r.meta.fs;
cfg = change_detector.turnTriggerConfig();

tt = change_detector.addSensorNoise(r.tt, [], noiseSeed);
meas = double([tt.deltaMeas tt.VxMeas tt.rMeas tt.ayMeas]);
t = seconds(tt.Time);
X = double(tt.X); Y = double(tt.Y);
turns = change_detector.findTurns(meas(:,1), meas(:,3), fs, cfg);
[b, a] = butter(2, cfg.lowpassHz/(fs/2));
dLp = rad2deg(abs(filter(b, a, meas(:,1))));

fig = figure('Name', sprintf('run %d', runId), 'Position', [80 60 1300 900]);
tl = tiledlayout(fig, 4, 2);
title(tl, sprintf('Run %d: %s, %s, %s, dm = %.0f kg, k_f = %.2f, %d turns detected', runId, row.split, ...
    row.class, row.changeType, row.dm, row.kf, numel(turns)));

ax = nexttile(tl, 1, [4 1]); hold(ax, 'on'); grid(ax, 'on'); axis(ax, 'equal');
plot(ax, X, Y, 'Color', [0.75 0.75 0.75], 'DisplayName', 'path');
plot(ax, X(1), Y(1), 'k^', 'MarkerFaceColor', 'k', 'DisplayName', 'start');
col = lines(max(numel(turns), 1));
for j = 1:numel(turns)
    [~, ~, i0, i1, iS] = change_detector.turnWindow(meas, turns(j), fs, cfg);
    c = turns(j).iEntry:turns(j).iExit;
    plot(ax, X(iS), Y(iS), '.', 'Color', col(j,:)*0.5 + 0.5, 'MarkerSize', 4, 'DisplayName', sprintf('turn %d: straight', j));
    plot(ax, X(i0:i1), Y(i0:i1), '-', 'Color', col(j,:), 'LineWidth', 1, 'DisplayName', sprintf('turn %d: window', j));
    plot(ax, X(c), Y(c), '-', 'Color', col(j,:), 'LineWidth', 3, 'DisplayName', sprintf('turn %d: %.0f deg', j, rad2deg(turns(j).headingRad)));
    plot(ax, X(c(1)), Y(c(1)), 'o', 'Color', col(j,:), 'MarkerFaceColor', col(j,:), 'HandleVisibility', 'off');
    plot(ax, X(c(end)), Y(c(end)), 'x', 'Color', col(j,:), 'MarkerSize', 10, 'LineWidth', 2, 'HandleVisibility', 'off');
end
xlabel(ax, 'X [m]'); ylabel(ax, 'Y [m]'); legend(ax, 'Location', 'bestoutside');
title(ax, 'Path: thick = turn (entry o, exit x), thin = window, dots = straight');

sig = {dLp, 'filtered |\delta| [deg]'; rad2deg(meas(:,1)), '\delta measured [deg]'; ...
       meas(:,3), 'r measured [rad/s]'; meas(:,2)*3.6, 'V_x [km/h]'};
axs = gobjects(4, 1);
for s = 1:4
    axs(s) = nexttile(tl, 2*s); hold(axs(s), 'on'); grid(axs(s), 'on');
    shadeLabelledTurns(axs(s), t, tt.turnIdx);
    for j = 1:numel(turns)
        [~, ~, i0, i1] = change_detector.turnWindow(meas, turns(j), fs, cfg);
        xline(axs(s), t([i0 i1]), ':', 'Color', col(j,:), 'HandleVisibility', 'off');
        xline(axs(s), t([turns(j).iEntry turns(j).iExit]), '-', 'Color', col(j,:), 'LineWidth', 1.5, 'HandleVisibility', 'off');
    end
    plot(axs(s), t, sig{s, 1}, 'k');
    ylabel(axs(s), sig{s, 2});
end
yline(axs(1), rad2deg([cfg.entryRad cfg.exitRad]), '--r', {'entry', 'exit'});
xlabel(axs(4), 't [s]');
title(axs(1), 'grey: labelled turn, solid lines: detected entry/exit, dotted: window');
linkaxes(axs, 'x'); xlim(axs(1), t([1 end]));
end

function shadeLabelledTurns(ax, t, turnIdx)
for k = unique(turnIdx(turnIdx > 0))'
    m = find(turnIdx == k);
    xregion(ax, t(m(1)), t(m(end)), 'FaceColor', [0.85 0.85 0.85], 'HandleVisibility', 'off');
end
end
