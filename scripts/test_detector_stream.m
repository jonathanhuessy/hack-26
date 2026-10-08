% TEST_DETECTOR_STREAM  The streaming detector, fed sample by sample, must reproduce the offline
%   reference pi/test_vectors/detector_<name>.mat (run_demo_replay): same verdict samples, classes,
%   probabilities and magnitudes. Prints PASS/FAIL per scenario and the time per sample.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
W = load(fullfile(projectRoot, 'models', 'export', 'weights.mat'));
ok = true;
for name = ["A", "B", "AB"]
    ref = load(fullfile(projectRoot, 'pi', 'test_vectors', "detector_" + name + ".mat"));
    det = change_detector.StreamingDetector('W', W, 'fs', ref.fs, 'K', ref.K);
    N = size(ref.meas, 1);
    got = struct('k', [], 'cls', [], 'p', zeros(0, 4), 'dm', [], 'kf', [], 'clsTurn', []);
    t0 = tic;
    for k = 1:N
        [cls, p, dm, kf, clsTurn, newVerdict] = det(ref.meas(k, :));
        if newVerdict
            got.k(end+1, 1) = k; got.cls(end+1, 1) = cls; got.p(end+1, :) = p;
            got.dm(end+1, 1) = dm; got.kf(end+1, 1) = kf; got.clsTurn(end+1, 1) = clsTurn;
        end
    end
    tPer = toc(t0)/N;
    same = isequal(got.k, ref.iVerdict) && isequal(got.cls, ref.clsVerdict) && isequal(got.clsTurn, ref.clsTurn);
    err = inf;
    if same
        err = max([max(abs(got.p - ref.pVerdict), [], 'all'), max(abs(got.dm - ref.dmVerdict))/1000, max(abs(got.kf - ref.kfVerdict))]);
    end
    pass = same && err < 1e-9;
    ok = ok && pass;
    fprintf('%-3s %d verdicts at samples %s | classes %s | max diff %.1e | %.0f us/sample | %s\n', name, numel(got.k), ...
        mat2str(got.k'), mat2str(got.cls'), err, 1e6*tPer, string(ifelse(pass, 'PASS', 'FAIL')));
end
assert(ok, 'streaming detector does not match the offline reference');

function v = ifelse(c, a, b)
if c, v = a; else, v = b; end
end
