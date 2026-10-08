function [win, straight, i0, i1, iStraight] = turnWindow(meas, turn, fs, cfg)
%TURNWINDOW Cut the feature window of one detected turn (contract 1).
%   meas      [delta Vx r ay] measured signals of the whole run (or the ring buffer)
%   turn      one element of findTurns (iEntry, iExit)
%   win       meas from cfg.marginS before entry to cfg.marginS after exit (rows i0:i1)
%   straight  the last cfg.straightS seconds before i0 with Vx > cfg.straightMinVx (rows iStraight)

if nargin < 4 || isempty(cfg)
    cfg = change_detector.turnTriggerConfig();
end

N = size(meas, 1);
nM = round(cfg.marginS*fs);
i0 = max(1, turn.iEntry - nM);
i1 = min(N, turn.iExit + nM);
win = meas(i0:i1, :);

cand = find(meas(1:i0-1, 2) > cfg.straightMinVx);
cand = cand(max(1, numel(cand) - round(cfg.straightS*fs) + 1):end);
straight = meas(cand, :);
iStraight = cand;
end
