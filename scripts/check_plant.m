% CHECK_PLANT  Verify models/tractor_plant.slx (P4). Errors out if a check fails.
%   1. Simulink vs lsim of the same state-space model at constant Vx
%   2. Steady-state yaw rate vs Vx*delta/(L + Kus*Vx^2)
%   3. Parameter step and ramp show up in the logged parameter signals and the response
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
p0 = change_detector.nominalParams();
pA = change_detector.applyScenario(p0, struct('addedMassKg', 1500, 'mountXM', -1.2));
fs = 100; t = (0:1/fs:60)';
ok = true;

%% 1. Simulink vs lsim
fprintf('--- 1. Simulink vs lsim (constant Vx) ---\n');
dist = change_detector.makeDisturbance(t, struct(), 11);
delta = deg2rad(2)*sin(2*pi*0.3*t) + deg2rad(1)*sin(2*pi*1.0*t) + deg2rad(3)*(t > 20 & t < 40);
delta = delta.*(1 - exp(-t/2));   % smooth start
relErr = zeros(2, 4, 2);   % speed x signal x (1 ms step, 10 ms step)
speedsKmh = [5 15];
steps = [0.001 0.01];
for iv = 1:2
    Vx = speedsKmh(iv)/3.6;
    prof = struct('t', t, 'delta', delta, 'Vx', Vx*ones(size(t)));
    s = change_detector.bicycleMatrices(p0, Vx);
    sys = ss(s.A, [s.B s.Bd], [s.C; eye(3)], [[zeros(2, 1) s.Dd]; zeros(3, 3)]);
    y = lsim(sys, [delta dist.Fyd dist.Mzd], t, 'foh');
    ref = [y(:,1) y(:,2) y(:,3) y(:,5)];      % r, ay, ydot, alphaF
    for is = 1:2
        out = change_detector.simulatePlant(prof, dist, p0, [], [], steps(is));
        sim_ = [out.r out.ay out.ydot out.alphaF];
        relErr(iv, :, is) = max(abs(sim_ - ref))./max(abs(ref));
        fprintf('Vx = %2d km/h, step %5.3f s: max rel err r %.1e, ay %.1e, ydot %.1e, alphaF %.1e\n', ...
            speedsKmh(iv), steps(is), relErr(iv, :, is));
    end
end
% 1 ms step: equations match lsim; 10 ms step: RK4 integration error of the production model
pass1 = all(relErr(:,:,1) < 1e-6, 'all') && all(relErr(:,:,2) < 1e-4, 'all');
fprintf('check 1: %s (limit 1e-6 at 1 ms, 1e-4 at 10 ms)\n', passStr(pass1));
ok = ok && pass1;

%% 2. Steady-state yaw rate
fprintf('--- 2. Steady-state yaw rate ---\n');
Kus = @(p) p.m/p.L*(p.lr/p.Caf - p.lf/p.Car);
Lw = p0.L;
tt = (0:1/fs:80)'; d0 = deg2rad(5);
err2 = [];
for pc = {p0, pA}
    p = pc{1};
    for Vx = [5 15]/3.6
        prof = struct('t', tt, 'delta', d0*ones(size(tt)), 'Vx', Vx*ones(size(tt)));
        out = change_detector.simulatePlant(prof, [], p);
        rAnalytic = Vx*d0/(Lw + Kus(p)*Vx^2);
        err2(end+1) = abs(out.r(end) - rAnalytic)/rAnalytic; %#ok<SAGROW>
        fprintf('lr = %.3f, Vx = %4.1f m/s: r_ss sim %.5f, analytic %.5f, rel err %.1e\n', p.lr, Vx, out.r(end), rAnalytic, err2(end));
    end
end
pass2 = all(err2 < 0.005);
fprintf('check 2: %s (limit 0.5 %%)\n', passStr(pass2));
ok = ok && pass2;

%% 3. Parameter step and ramp
fprintf('--- 3. Parameter schedule ---\n');
Vx = 15/3.6; tt = (0:1/fs:100)';
prof = struct('t', tt, 'delta', d0*ones(size(tt)), 'Vx', Vx*ones(size(tt)));
pv = @(p) [p.m p.lf p.lr p.Izz p.Caf p.Car p.sigmaF];
outS = change_detector.simulatePlant(prof, [], p0, pA, struct('tStart', 40, 'tEnd', 40));
before = outS.params(tt < 40, :); after = outS.params(tt >= 40.02, :);
passStep = max(abs(before - pv(p0)), [], 'all')/max(pv(p0)) < 1e-9 && ...
           max(abs(after - pv(pA)), [], 'all')/max(pv(pA)) < 1e-9;
rNew = Vx*d0/(Lw + Kus(pA)*Vx^2);
passStepResp = abs(outS.r(end) - rNew)/rNew < 0.005;
outR = change_detector.simulatePlant(prof, [], p0, pA, struct('tStart', 20, 'tEnd', 60));
mMid = interp1(tt, outR.params(:,1), 40);
passRamp = abs(mMid - (p0.m + pA.m)/2) < 1e-6*p0.m && abs(outR.params(end,1) - pA.m) < 1e-6*p0.m;
fprintf('step: params switch %s, response reaches new steady state %s\n', passStr(passStep), passStr(passStepResp));
fprintf('ramp: mass is halfway at midpoint and final %s\n', passStr(passRamp));
pass3 = passStep && passStepResp && passRamp;
fprintf('check 3: %s\n', passStr(pass3));
ok = ok && pass3;

fprintf('\nP4 overall: %s\n', passStr(ok));
assert(ok, 'check_plant failed');

function s = passStr(tf)
if tf, s = 'PASS'; else, s = 'FAIL'; end
end
