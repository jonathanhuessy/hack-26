function cfg = turnTriggerConfig()
%TURNTRIGGERCONFIG Thresholds of the turn trigger and window (contract 1), shared by
%   findTurns (offline), the streaming detector (Simulink) and the Pi port.

cfg.lowpassHz = 1;                  % low-pass on delta before thresholding
cfg.entryRad = deg2rad(5);          % turn entry
cfg.exitRad = deg2rad(3);           % turn exit (hysteresis)
cfg.holdS = 2;                      % exit only after this long below exitRad (omega reversals)
cfg.minHeadingRad = deg2rad(120);   % shorter heading changes are operator corrections
cfg.marginS = 5;                    % window: this long before entry and after exit
cfg.straightS = 20;                 % preceding straight: last this many seconds ...
cfg.straightMinVx = 3;              % ... with Vx above this [m/s]
end
