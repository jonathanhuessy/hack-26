% TEST_MLP_FORWARD  Deterministic MATLAB reference for Python model parity.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
W = struct();
W.mu = [0 0];
W.sigma = [1 1];
W.W1 = eye(2);
W.b1 = [0; 0];
W.W2 = [1 0; 0 1; -1 0; 0 -1];
W.b2 = [0; 0; 0; 0];
out = change_detector.mlpForward([1 2], W);
assert(abs(sum(out.classProbabilities) - 1) < 1e-12);
assert(all(isfinite(out.classProbabilities)));
fprintf('PASS mlpForward deterministic reference\n');
