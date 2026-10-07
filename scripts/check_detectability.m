% CHECK_DETECTABILITY  Go/no-go gate for the dataset design (P6).
%   Same trajectory, disturbance and seed for nominal and changed vehicle, no sensor noise.
%   d^2 = sum(dr^2)/sigma_r^2 + sum(day^2)/sigma_ay^2 per run, split into turn and straight samples.
%   cos(theta) = <dr_A, dr_B>/(|dr_A||dr_B|) for A and B changes with similar d^2.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
p0 = change_detector.nominalParams();

sigR = 0.01;     % gyro noise [rad/s]
sigAy = 0.2;     % accelerometer noise [m/s^2]
seed = 3;
mountRear = -1.2;            % rear hitch, behind the rear axle
mountFront = p0.L + 0.9;     % front weight, ahead of the front axle

prof = change_detector.makeManeuverProfile(struct(), seed);
dist = change_detector.makeDisturbance(prof.t, struct(), 1000 + seed);
isTurn = prof.segment == 3; isStraight = prof.segment == 1;
base = change_detector.simulatePlant(prof, dist, p0);

d2 = @(o, mask) [sum((o.r(mask) - base.r(mask)).^2)/sigR^2, sum((o.ay(mask) - base.ay(mask)).^2)/sigAy^2];

%% Scenarios
rearMass = [100 250 500 1000 1500 2000];
frontMass = [100 250 500 1000 1500];
kfList = [0.6 0.7 0.8 0.9 0.95 1.1 1.2];
mix = {[1000 0.8], [500 0.9], [1500 0.7], [250 1.1]};   % [rear kg, kf]

mk = @(dm, xm, kf) change_detector.applyScenario(p0, struct('addedMassKg', dm, 'mountXM', xm, 'kf', kf));
scen = {}; label = {}; kind = [];
for dm = rearMass,  scen{end+1} = mk(dm, mountRear, 1);  label{end+1} = sprintf('A rear +%d kg', dm);  kind(end+1) = 1; end %#ok<SAGROW>
for dm = frontMass, scen{end+1} = mk(dm, mountFront, 1); label{end+1} = sprintf('A front +%d kg', dm); kind(end+1) = 2; end %#ok<SAGROW>
for kf = kfList,    scen{end+1} = mk(0, 0, kf);          label{end+1} = sprintf('B kf %.2f', kf);      kind(end+1) = 3; end %#ok<SAGROW>
for i = 1:numel(mix), scen{end+1} = mk(mix{i}(1), mountRear, mix{i}(2)); label{end+1} = sprintf('A+B rear +%d kg, kf %.2f', mix{i}); kind(end+1) = 4; end %#ok<SAGROW>

n = numel(scen);
dTurn = zeros(n, 2); dStr = zeros(n, 2); dAll = zeros(n, 2); dr = zeros(numel(prof.t), n); day = dr;
for i = 1:n
    o = change_detector.simulatePlant(prof, dist, p0, scen{i});
    dTurn(i,:) = d2(o, isTurn); dStr(i,:) = d2(o, isStraight); dAll(i,:) = d2(o, true(size(isTurn)));
    dr(:,i) = o.r - base.r; day(:,i) = o.ay - base.ay;
end

fprintf('\nd^2 per run (r part + ay part), sigma_r = %.3f rad/s, sigma_ay = %.2f m/s^2\n', sigR, sigAy);
fprintf('%-28s %12s %12s %12s\n', 'scenario', 'total', 'turns', 'straights');
for i = 1:n
    fprintf('%-28s %12.3g %12.3g %12.3g\n', label{i}, sum(dAll(i,:)), sum(dTurn(i,:)), sum(dStr(i,:)));
end

%% Smallest detectable change (d^2 >= 25)
fprintf('\nSmallest change with d^2 >= 25:\n');
sets = {1, 'A rear', rearMass; 2, 'A front', frontMass; 3, 'B', kfList};
for s = 1:size(sets, 1)
    idx = find(kind == sets{s,1});
    tot = sum(dAll(idx,:), 2);
    mags = sets{s,3};
    first = find(tot >= 25, 1);
    if s == 3
        dev = abs(mags - 1); [~, ord] = sort(dev); first = ord(find(tot(ord) >= 25, 1));
    end
    if isempty(first), fprintf('  %-8s: none of the tested magnitudes reaches 25\n', sets{s,2});
    else, fprintf('  %-8s: %g (d^2 = %.3g)\n', sets{s,2}, mags(first), tot(first)); end
end

%% Separability (matched d^2)
fprintf('\nSeparability, cos(theta) between difference signals (dr only | dr and day stacked):\n');
vec2 = @(i) [dr(:,i)/sigR; day(:,i)/sigAy];
cosf = @(a, b) dot(a, b)/(norm(a)*norm(b));
iR = find(kind == 1); iF = find(kind == 2); iB = find(kind == 3);
totd = sum(dAll, 2);
pairs = {iR, iB, 'A rear vs B'; iF, iB, 'A front vs B'; iR, iF, 'A rear vs A front'};
maxCos = 0;
for pp = 1:size(pairs, 1)
    for a = pairs{pp,1}(:)'
        [~, j] = min(abs(log(totd(pairs{pp,2})/totd(a))));
        b = pairs{pp,2}(j);
        c1 = cosf(dr(:,a), dr(:,b)); c2 = cosf(vec2(a), vec2(b));
        fprintf('  %-18s %-16s vs %-14s d2 %.2g vs %.2g: cos %+.2f | %+.2f\n', pairs{pp,3}, label{a}, label{b}, totd(a), totd(b), c1, c2);
        if pp == 1, maxCos = max(maxCos, abs(c1)); end
    end
end
fprintf('max |cos| for A rear vs B at matched d^2: %.2f (limit 0.9; front weight is informational only, A is restricted to rear mounts)\n', maxCos);

%% Operator correction amplitude: are the straights informative?
fprintf('\nStraight-only d^2 vs operator correction amplitude (A rear +1000 kg | B kf 0.8):\n');
iA = find(strcmp(label, 'A rear +1000 kg')); iBk = find(strcmp(label, 'B kf 0.80'));
idsAB = [iA iBk];
for amp = [0 0.5 1.5]
    pr = change_detector.makeManeuverProfile(struct('operatorStdDeg', [amp amp]), seed);
    ds = change_detector.makeDisturbance(pr.t, struct(), 1000 + seed);
    b0 = change_detector.simulatePlant(pr, ds, p0);
    mskS = pr.segment == 1;
    res = zeros(1, 2);
    for q = 1:2
        o = change_detector.simulatePlant(pr, ds, p0, scen{idsAB(q)});
        res(q) = sum((o.r(mskS) - b0.r(mskS)).^2)/sigR^2 + sum((o.ay(mskS) - b0.ay(mskS)).^2)/sigAy^2;
    end
    fprintf('  operator std %.1f deg: A %.3g, B %.3g\n', amp, res);
end

%% Plot
fig = figure('Name', 'check_detectability', 'Position', [100 100 1200 850]);
tl = tiledlayout(fig, 2, 2);
title(tl, 'Detectability of A (rear ballast) and B (front tire stiffness)');
subtitle(tl, {'d^2 = \Sigma(\Delta r)^2/\sigma_r^2 + \Sigma(\Delta a_y)^2/\sigma_{a_y}^2: energy of the signal difference to the nominal vehicle over one run, in units of sensor noise.', ...
    'd^2 \geq 25 counts as detectable (dashed line).'});
nexttile(tl); hold on; grid on; set(gca, 'YScale', 'log');
plot(rearMass, sum(dAll(kind == 1,:), 2), 'o-', 'DisplayName', 'A rear hitch');
plot(frontMass, sum(dAll(kind == 2,:), 2), 's-', 'DisplayName', 'front weight (not used, looks like B)');
yline(25, 'k--', 'd^2 = 25', 'HandleVisibility', 'off', 'LabelHorizontalAlignment', 'left'); legend('Location', 'southeast'); xlabel('added mass [kg]'); ylabel('d^2 (total over the run)'); title('A: mounted ballast');
nexttile(tl); hold on; grid on; set(gca, 'YScale', 'log');
plot(kfList, sum(dAll(kind == 3,:), 2), 'o-'); yline(25, 'k--'); xlabel('k_f = C_{af} / C_{af,nominal}'); ylabel('d^2 (total over the run)'); title('B: front tire stiffness factor k_f (1 = nominal, 0.8 = 20 % softer)');
t0 = prof.info.turns(1);
w = prof.t >= t0.tStart - 2 & prof.t <= t0.tEnd + 2;
nexttile(tl); hold on; grid on;
plot(prof.t(w), dr(w, strcmp(label, 'A rear +1000 kg'))); plot(prof.t(w), dr(w, strcmp(label, 'A front +1000 kg')));
plot(prof.t(w), dr(w, strcmp(label, 'B kf 0.80'))); legend('A rear +1000 kg', 'front weight +1000 kg', 'B: k_f = 0.8'); ylabel('\Delta r [rad/s]'); xlabel('t [s]'); title('Turn 1: yaw rate difference to nominal');
nexttile(tl); hold on; grid on;
plot(prof.t(w), base.r(w)); plot(prof.t(w), prof.delta(w).*prof.Vx(w)/p0.L); legend('r nominal', 'V\delta/L'); xlabel('t [s]'); title('Turn 1: nominal yaw rate vs kinematic V\delta/L');
exportgraphics(fig, fullfile(projectRoot, 'data', 'check_detectability.png'), 'Resolution', 110);
