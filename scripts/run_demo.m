function R = run_demo(name, mode)
%RUN_DEMO Run the Simulink demo models/tractor_change_detection.slx on a demo scenario (H4).
%   R = run_demo('A')            live: plant + sensor noise + streaming detector
%   R = run_demo('AB', 'replay') replay: the measured signals of pi/scenarios/demo_reference_AB.mat
%   Scenarios from change_detector.demoScenario ('A', 'B', 'AB'): 6 swaths, change after turn 2.
%   The sensor noise is the same as in run_demo_replay (seed 777, additive), so live mode must
%   reproduce the offline reference too. Prints the verdict per turn and PASS/FAIL against
%   pi/scenarios/demo_reference_<name>.mat.

if nargin < 1, name = 'A'; end
if nargin < 2, mode = 'live'; end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
mdl = 'tractor_change_detection';
if ~bdIsLoaded(mdl), open_system(fullfile(projectRoot, 'models', [mdl '.slx'])); end

sc = change_detector.demoScenario(name);
ref = load(fullfile(projectRoot, 'pi', 'scenarios', ['demo_reference_' name '.mat']));
t = sc.prof.t;
zeroTt = timetable(seconds(t), zeros(size(t)), zeros(size(t)), zeros(size(t)), zeros(size(t)), ...
    'VariableNames', {'delta', 'Vx', 'r', 'ay'});
nz = change_detector.addSensorNoise(zeroTt, [], ref.noiseSeed);   % additive noise only
vec = @(p) [p.m; p.lf; p.lr; p.Izz; p.Caf; p.Car; p.sigmaF];

in = Simulink.SimulationInput(mdl);
vars = {'maneuverIn', [t sc.prof.delta sc.prof.Vx sc.dist.Fyd sc.dist.Mzd]; ...
        'noiseIn', [t double([nz.deltaMeas nz.VxMeas nz.rMeas nz.ayMeas])]; ...
        'replayIn', [t ref.meas]; ...
        'useReplay', double(strcmp(mode, 'replay')); ...
        'detW', load(fullfile(projectRoot, 'models', 'export', 'weights.mat')); ...
        'p0', vec(sc.pStart); 'p1', vec(sc.pEnd); 'tStart', sc.sched.tStart; 'tEnd', sc.sched.tEnd; ...
        'imuXRearAxleM', sc.pStart.imuXFromRearAxleM; 'x0_plant', zeros(6, 1); 'tStop', t(end)};
for i = 1:size(vars, 1)
    in = in.setVariable(vars{i, 1}, vars{i, 2}, 'Workspace', mdl);
end
tic; so = sim(in); tSim = toc;

k = find(so.detNew);
R = struct('k', k, 'cls', so.detCls(k), 'p', so.detP(k, :), 'dm', so.detDm(k), 'kf', so.detKf(k), ...
    'clsTurn', so.detClsTurn(k), 'simTime', tSim);
names  = ["nominal", "A", "B", "AB"];
fprintf('Simulink demo %s (%s), %.0f s simulated in %.1f s\n', name, mode, t(end), tSim);
fprintf('%5s %8s %-8s %-8s %-8s %10s %8s\n', 'turn', 'time', 'truth', 'single', 'verdict', 'dm [kg]', 'kf');
for j = 1:numel(k)
    tr = "-"; if j <= numel(ref.clsTrue), tr = names(ref.clsTrue(j) + 1); end
    fprintf('%5d %7.0fs %-8s %-8s %-8s %10.0f %8.2f\n', j, t(k(j)), tr, names(R.clsTurn(j) + 1), ...
        names(R.cls(j) + 1), R.dm(j), R.kf(j));
end
same = isequal(k, ref.iVerdict) && isequal(R.cls, ref.clsVerdict) && isequal(R.clsTurn, ref.clsTurn);
err = inf;
if same
    err = max([max(abs(R.p - ref.pVerdict), [], 'all'), max(abs(R.dm - ref.dmVerdict))/1000, max(abs(R.kf - ref.kfVerdict))]);
end
R.pass = same && err < 1e-6;
fprintf('against the offline reference: max difference %.1e -> %s\n', err, string(ifelse(R.pass, 'PASS', 'FAIL')));
end

function v = ifelse(c, a, b)
if c, v = a; else, v = b; end
end
