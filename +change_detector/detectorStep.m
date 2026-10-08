function [out, newVerdict] = detectorStep(u, W)
%DETECTORSTEP Streaming three-input detector (delta, Vx, r).
%   [out, newVerdict] = detectorStep(u, W) consumes one 100-Hz sample. W
%   contains the mlpForward artifact and may contain reset=true to clear state.
%   The trigger/window rules are shared with findTurns and turnWindow.

persistent s
cfg = change_detector.turnTriggerConfig();
fs = 100;
if isempty(s) || (isfield(W, 'reset') && W.reset)
    [b, a] = butter(2, cfg.lowpassHz/(fs/2));
    s = struct('delta', zeros(0,1), 'vx', zeros(0,1), 'r', zeros(0,1), ...
        'b', b, 'a', a, 'inTurn', false, 'entry', 0, 'quiet', 0, ...
        'filterState', zeros(max(numel(a), numel(b))-1, 1), ...
        'pendingEntry', 0, 'pendingExit', 0, 'turnIndex', 0, ...
        'historyP', zeros(0,4), 'historyDm', zeros(0,1), 'historyKf', zeros(0,1), ...
        'last', struct('classProbabilities', ones(1,4)/4, 'deltaM_kg', NaN, 'kF', NaN, ...
                       'state', 'idle'));
end
if numel(u) ~= 3 || any(~isfinite(u))
    error('change_detector:detectorStep:Input', 'u must be finite [delta Vx r]');
end

s.delta(end+1,1) = u(1);
s.vx(end+1,1) = u(2);
s.r(end+1,1) = u(3);
k = numel(s.delta);
[dFiltered, s.filterState] = filter(s.b, s.a, u(1), s.filterState);
if ~s.inTurn
    if abs(dFiltered) > cfg.entryRad
        s.inTurn = true;
        s.entry = k;
        s.quiet = 0;
    end
else
    if abs(dFiltered) < cfg.exitRad
        s.quiet = s.quiet + 1;
    else
        s.quiet = 0;
    end
    if s.quiet >= round(cfg.holdS*fs)
        exitIndex = k - round(cfg.holdS*fs) + 1;
        heading = sum(s.r(s.entry:exitIndex))/fs;
        if abs(heading) >= cfg.minHeadingRad
            s.pendingEntry = s.entry;
            s.pendingExit = exitIndex;
        end
        s.inTurn = false;
        s.entry = 0;
        s.quiet = 0;
    end
end

newVerdict = false;
if s.pendingExit > 0 && k >= s.pendingExit + round(cfg.marginS*fs)
    i0 = max(1, s.pendingEntry - round(cfg.marginS*fs));
    i1 = min(k, s.pendingExit + round(cfg.marginS*fs));
    meas = [s.delta s.vx s.r];
    win = meas(i0:i1, :);
    cand = find(s.vx(1:i0-1) > cfg.straightMinVx);
    if isempty(cand)
        straight = zeros(0,3);
    else
        cand = cand(max(1, numel(cand)-round(cfg.straightS*fs)+1):end);
        straight = meas(cand, :);
    end
    features = change_detector.selectedTurnFeatures(win, straight, fs);
    prediction = change_detector.mlpForward(features, W);
    s.turnIndex = s.turnIndex + 1;
    s.historyP(end+1,:) = prediction.classProbabilities;
    s.historyDm(end+1,1) = prediction.deltaM_kg;
    s.historyKf(end+1,1) = prediction.kF;
    if size(s.historyP,1) > 3, s.historyP(1,:) = []; end
    if numel(s.historyDm) > 3, s.historyDm(1) = []; end
    if numel(s.historyKf) > 3, s.historyKf(1) = []; end
    s.last.classProbabilities = mean(s.historyP, 1);
    [~, classIndex] = max(s.last.classProbabilities);
    s.last.classIndex = classIndex;
    if any(isfinite(s.historyDm))
        s.last.deltaM_kg = median(s.historyDm(isfinite(s.historyDm)));
    else
        s.last.deltaM_kg = NaN;
    end
    if any(isfinite(s.historyKf))
        s.last.kF = median(s.historyKf(isfinite(s.historyKf)));
    else
        s.last.kF = NaN;
    end
    s.last.turnIndex = s.turnIndex;
    s.last.timestamp_s = (k-1)/fs;
    s.last.state = 'idle';
    s.pendingEntry = 0;
    s.pendingExit = 0;
    newVerdict = true;
end

out = s.last;
if s.inTurn
    out.state = 'in_turn';
elseif s.pendingExit > 0
    out.state = 'waiting_window';
end
end
