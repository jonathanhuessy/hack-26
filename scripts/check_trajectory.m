% CHECK_TRAJECTORY  Verify the maneuver profile on the plant (P5) and plot one example run.
%   Limits: |delta| <= 19 deg, |ddelta/dt| <= 25 deg/s, net heading change per turn
%   180 +/- 15 deg, omega turns land within 1 m of the next lane without
%   disturbances, heading drift below 5 deg per straight with disturbances, every
%   straight within 5 deg of its ideal direction, neighbouring straights never closer
%   than 60 % of the intended lane offset, steering rate incl. corrections <= 25 deg/s.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
p0 = change_detector.nominalParams();
nSeeds = 20;
ok = true;

%% Clean runs (no operator corrections, no disturbances): geometry checks
fprintf('--- Clean runs: geometry ---\n');
cfgClean = struct('operatorStdDeg', [0 0]);
maxDelta = 0; maxRate = 0; dpsiErr = 0; laneErr = 0; nOmega = 0;
for seed = 1:nSeeds
    prof = change_detector.makeManeuverProfile(cfgClean, seed);
    out = change_detector.simulatePlant(prof, [], p0);
    maxDelta = max(maxDelta, rad2deg(max(abs(prof.delta))));
    maxRate = max(maxRate, rad2deg(max(abs(diff(prof.delta)))*prof.fs));
    for k = 1:numel(prof.info.turns)
        tn = prof.info.turns(k);
        i0 = round(tn.tStart*prof.fs) + 1; i1 = round(tn.tEnd*prof.fs);
        dpsi = rad2deg(out.psi(i1) - out.psi(i0));
        dpsiErr = max(dpsiErr, abs(abs(dpsi) - 180));
        if tn.isOmega
            ex = [cos(out.psi(i0)), sin(out.psi(i0))];
            d = [out.X(i1) - out.X(i0), out.Y(i1) - out.Y(i0)];
            lateral = ex(1)*d(2) - ex(2)*d(1);          % positive left of the swath direction
            laneErr = max(laneErr, abs(lateral - tn.dir*prof.info.w));
            nOmega = nOmega + 1;
        end
    end
end
fprintf('max |delta| %.2f deg (limit 19), max rate %.2f deg/s (limit 25)\n', maxDelta, maxRate);
fprintf('max deviation of net heading change from 180 deg: %.2f deg (limit 15)\n', dpsiErr);
fprintf('omega turns (%d): max lane landing error %.2f m (limit 1)\n', nOmega, laneErr);
pass1 = maxDelta <= 19 && maxRate <= 25 && dpsiErr <= 15 && laneErr <= 1;
fprintf('geometry: %s\n', passStr(pass1));
ok = ok && pass1;

%% Full runs with corrections and disturbances: heading drift on straights
fprintf('--- Full runs: heading drift and swath parallelism ---\n');
maxDrift = 0; speedMin = inf; speedMax = 0; maxHeadErr = 0; minSep = inf; maxRateFull = 0;
for seed = 1:nSeeds
    prof = change_detector.makeManeuverProfile(struct(), seed);
    maxRateFull = max(maxRateFull, rad2deg(max(abs(diff(prof.delta)))*prof.fs));
    dist = change_detector.makeDisturbance(prof.t, struct(), 1000 + seed);
    out = change_detector.simulatePlant(prof, dist, p0);
    nSw = max(prof.swath);
    idxS = cell(nSw, 1);
    for k = 1:nSw
        idx = find(prof.swath == k & prof.segment == 1);
        idxS{k} = idx(1:10:end);
        maxDrift = max(maxDrift, rad2deg(abs(out.psi(idx(end)) - out.psi(idx(1)))));
        ideal = pi*(mod(k, 2) == 0)*prof.info.turns(1).dir;   % heading of swath k
        err = abs(angle(exp(1i*(out.psi(idx) - ideal))));
        maxHeadErr = max(maxHeadErr, rad2deg(max(err)));
    end
    for k = 1:nSw - 1
        tn = prof.info.turns(k);
        expected = tn.isOmega*prof.info.w + ~tn.isOmega*2*prof.info.R;
        a = idxS{k}; b = idxS{k+1};
        dmin = sqrt(min((out.X(a) - out.X(b)').^2 + (out.Y(a) - out.Y(b)').^2, [], 'all'));
        minSep = min(minSep, dmin/expected);
    end
    speedMin = min(speedMin, min(prof.Vx)*3.6); speedMax = max(speedMax, max(prof.Vx)*3.6);
end
fprintf('speed range %.1f-%.1f km/h, max heading change on a straight %.2f deg (limit 5)\n', speedMin, speedMax, maxDrift);
fprintf('max heading error of a straight from its ideal direction: %.2f deg (limit 5)\n', maxHeadErr);
fprintf('min distance between neighbouring straights / intended lane offset: %.2f (limit 0.6)\n', minSep);
fprintf('max steering rate incl. operator corrections: %.1f deg/s (limit 25)\n', maxRateFull);
pass2 = maxDrift < 5 && maxHeadErr <= 5 && minSep >= 0.6 && maxRateFull <= 25;
fprintf('drift: %s\n', passStr(pass2));
ok = ok && pass2;

%% Example plot
prof = change_detector.makeManeuverProfile(struct(), 3);
dist = change_detector.makeDisturbance(prof.t, struct(), 1003);
out = change_detector.simulatePlant(prof, dist, p0);
fig = figure('Name', 'check_trajectory', 'Position', [100 100 1100 800]);
tl = tiledlayout(fig, 3, 2);
nexttile(tl, 1, [3 1]);
plot(out.X, out.Y); axis equal tight; grid on; xlabel('X [m]'); ylabel('Y [m]');
title('Path (3 swaths, 3 end-of-row turns)'); hold on
for k = 1:numel(prof.info.turns)
    i0 = round(prof.info.turns(k).tStart*prof.fs) + 1;
    plot(out.X(i0), out.Y(i0), 'ro');
end
ax = nexttile(tl, 2); plot(prof.t, rad2deg(prof.delta)); grid on; ylabel('\delta [deg]'); title('Front wheel angle');
ax(2) = nexttile(tl, 4); plot(prof.t(2:end), rad2deg(diff(prof.delta))*prof.fs); grid on; ylabel('d\delta/dt [deg/s]');
ax(3) = nexttile(tl, 6); plot(prof.t, prof.Vx*3.6); grid on; ylabel('V_x [km/h]'); xlabel('t [s]');
linkaxes(ax, 'x');
exportgraphics(fig, fullfile(projectRoot, 'data', 'check_trajectory.png'), 'Resolution', 110);

fprintf('\nP5 overall: %s\n', passStr(ok));
assert(ok, 'check_trajectory failed');

function s = passStr(tf)
if tf, s = 'PASS'; else, s = 'FAIL'; end
end
