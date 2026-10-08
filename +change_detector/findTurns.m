function turns = findTurns(delta, r, fs, cfg)
%FINDTURNS Detect end-of-row turns from the measured steering angle and yaw rate.
%   turns = findTurns(delta, r, fs) runs the causal turn trigger (contract 1 in plan.md)
%   over a whole run and returns a struct array with fields
%     iEntry, iExit   sample indices of turn entry and exit
%     headingRad      yaw angle change between entry and exit (integrated yaw rate)
%   Trigger: |delta| low-passed at cfg.lowpassHz rises above cfg.entryRad -> entry.
%   Exit: the filtered |delta| has stayed below cfg.exitRad for cfg.holdS seconds;
%   iExit is the first sample of that quiet period. The hold time keeps the full-lock
%   reversals of an omega turn inside one turn. Turns with |heading change| below
%   cfg.minHeadingRad are dropped (operator corrections on straights).
%   The streaming detector must use the same cfg (change_detector.turnTriggerConfig).

if nargin < 4 || isempty(cfg), cfg = change_detector.turnTriggerConfig(); end
delta = double(delta(:)); r = double(r(:));

[b, a] = butter(2, cfg.lowpassHz/(fs/2));
dAbs = abs(filter(b, a, delta));
holdN = round(cfg.holdS*fs);

turns = struct('iEntry', {}, 'iExit', {}, 'headingRad', {});
inTurn = false; quiet = 0; iEntry = 0;
for k = 1:numel(delta)
    if ~inTurn
        if dAbs(k) > cfg.entryRad
            inTurn = true; iEntry = k; quiet = 0;
        end
    else
        if dAbs(k) < cfg.exitRad
            quiet = quiet + 1;
        else
            quiet = 0;
        end
        if quiet >= holdN
            iExit = k - holdN + 1;
            heading = sum(r(iEntry:iExit))/fs;
            if abs(heading) >= cfg.minHeadingRad
                turns(end+1) = struct('iEntry', iEntry, 'iExit', iExit, 'headingRad', heading); %#ok<AGROW>
            end
            inTurn = false;
        end
    end
end
end
