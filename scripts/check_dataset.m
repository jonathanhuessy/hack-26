% CHECK_DATASET  Exit check for the dataset (P7/P9): every run present and finite, balanced, reproducible.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
dataDir = fullfile(projectRoot, 'data');
idx = load(fullfile(dataDir, 'index.mat')).idx;
p0 = change_detector.nominalParams();
ok = true;

%% Files and finite values
missing = 0; bad = 0; bytes = 0;
for i = 1:height(idx)
    f = fullfile(dataDir, 'runs', idx.file(i));
    if ~isfile(f), missing = missing + 1; continue; end
    d = dir(f); bytes = bytes + d.bytes;
    m = matfile(f);
    tt = m.tt;
    if any(~isfinite(tt{:, {'delta', 'Vx', 'r', 'ay', 'ydot', 'alphaF', 'X', 'Y', 'psi', 'params', 'dm', 'kf'}}), 'all')
        bad = bad + 1; fprintf('non-finite values in %s\n', idx.file(i));
    end
end
fprintf('runs in index %d, missing files %d, runs with NaN/Inf %d, total size %.2f GB\n', height(idx), missing, bad, bytes/1e9);
ok = ok && missing == 0 && bad == 0;

%% Balance and splits
disp(groupsummary(idx(idx.changeType == "constant", :), {'split', 'class'}));
cnt = groupcounts(idx(idx.changeType == "constant", :), {'split', 'class'});
bal = all(cnt.GroupCount(cnt.split == "train") == cnt.GroupCount(find(cnt.split == "train", 1)));
fprintf('classes balanced within each split: %d\n', bal);
ok = ok && bal;

%% Reproducibility of a few runs from their index row
pick = unique([1, find(idx.class == "AB", 1), find(idx.changeType == "step", 1), find(idx.changeType == "ramp", 1)]);
maxDiff = 0;
for k = pick
    r = load(fullfile(dataDir, 'runs', idx.file(k)));
    tt2 = change_detector.generateRun(idx(k, :), p0);
    maxDiff = max(maxDiff, max(abs(double(tt2.r) - double(r.tt.r))));
end
fprintf('regenerated %d runs from their index rows, max difference in r: %.1e\n', numel(pick), maxDiff);
ok = ok && maxDiff == 0;

fprintf('\nDataset check: %s\n', string(ifelse(ok, 'PASS', 'FAIL')));
assert(ok, 'check_dataset failed');

function v = ifelse(c, a, b)
if c, v = a; else, v = b; end
end
