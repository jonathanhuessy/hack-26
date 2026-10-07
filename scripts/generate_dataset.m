function generate_dataset(opts)
%GENERATE_DATASET Write the labelled raw dataset to data/ (P7 and P9).
%   generate_dataset()                       full set (about 1420 runs)
%   generate_dataset(struct('nTrain',4,'nVal',2,'nTest',2,'outDir',tempdir))   small test set
%
%   data/index.mat       run table (config, labels, seeds)
%   data/config.mat      nominal parameters, MATLAB version, date, noise levels to use later
%   data/runs/run_XXXXX.mat   timetable tt + struct meta, one file per run
%   Existing run files are skipped, so an interrupted run can simply be restarted.

if nargin < 1, opts = struct(); end
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
if ~isfield(opts, 'outDir'), opts.outDir = fullfile(projectRoot, 'data'); end
runDir = fullfile(opts.outDir, 'runs');
if ~isfolder(runDir), mkdir(runDir); end

p0 = change_detector.nominalParams();
idx = change_detector.buildRunIndex(opts);
save(fullfile(opts.outDir, 'index.mat'), 'idx');
cfg = struct('nominalParams', p0, 'matlabVersion', version, 'created', datetime('now'), ...
    'opts', opts, 'sensorNoise', struct('gyroStdRadps', 0.01, 'accelStdMps2', 0.2, 'steerStdDeg', 0.2)); %#ok<NASGU>
save(fullfile(opts.outDir, 'config.mat'), 'cfg');

n = height(idx);
fprintf('%d runs, writing to %s\n', n, runDir);
t0 = tic;
for i = 1:n
    file = fullfile(runDir, idx.file(i));
    if isfile(file), continue; end
    [tt, meta] = change_detector.generateRun(idx(i,:), p0); %#ok<ASGLU>
    tmp = file + ".tmp.mat";
    save(tmp, 'tt', 'meta');
    movefile(tmp, file);
    if mod(i, 25) == 0 || i == n
        fprintf('%5d/%d done, %.0f s elapsed\n', i, n, toc(t0));
    end
end
fprintf('finished in %.0f s\n', toc(t0));
end
