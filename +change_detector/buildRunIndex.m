function idx = buildRunIndex(opts)
%BUILDRUNINDEX Deterministic run table for the dataset (one row per run).
%   idx = buildRunIndex(opts), opts fields (defaults): nTrain 240, nVal 60 and nTest 50
%   runs per class (nominal, A, B, AB), plus 20 demo runs with step and ramp changes.
%   All random draws come from the run id, so a row regenerates its run exactly.
%
%   A = mounted ballast at the rear hitch, B = front cornering stiffness factor kf.
%   Soil variation (+/-8 % on Caf and Car) is drawn for every run, also nominal.

if nargin < 1, opts = struct(); end
d = struct('nTrain', 240, 'nVal', 60, 'nTest', 50);
for f = fieldnames(d)'
    if ~isfield(opts, f{1}), opts.(f{1}) = d.(f{1}); end
end

classes = {'nominal', 'A', 'B', 'AB'};
spec = {};   % {split, class, changeType, reverse}
for ic = 1:4
    spec = [spec; repmat({'train', classes{ic}, 'constant', false}, opts.nTrain, 1)]; %#ok<AGROW>
    spec = [spec; repmat({'val',   classes{ic}, 'constant', false}, opts.nVal, 1)];   %#ok<AGROW>
    spec = [spec; repmat({'test',  classes{ic}, 'constant', false}, opts.nTest, 1)];  %#ok<AGROW>
end
demo = [repmat({'demo', 'A',  'step', false}, 8, 1);   % nominal -> A
        repmat({'demo', 'B',  'step', false}, 4, 1);   % nominal -> B
        repmat({'demo', 'A',  'step', true},  3, 1);   % A -> nominal (implement removed)
        repmat({'demo', 'A',  'ramp', true},  3, 1);   % tank draining
        repmat({'demo', 'A',  'ramp', false}, 2, 1)];  % hopper filling
spec = [spec; demo];

n = size(spec, 1);
rows = cell(n, 1);
for i = 1:n
    rows{i} = drawRun(i, spec{i,1}, spec{i,2}, spec{i,3}, spec{i,4});
end
idx = struct2table([rows{:}]);
end

function rc = drawRun(id, split, cls, changeType, reverse)
rs = RandStream('mt19937ar', 'Seed', 7000000 + id);
u = @(a, b) a + (b - a)*rand(rs);

hasA = any(strcmp(cls, {'A', 'AB'}));
hasB = any(strcmp(cls, {'B', 'AB'}));
rc.id = id;
rc.split = string(split);
rc.class = string(cls);
rc.changeType = string(changeType);
rc.reverse = reverse;
rc.dm = hasA*u(250, 2000);            % kg at the rear hitch; 250 kg is the d^2 >= 25 limit from P6
rc.mountXM = -1.2;                    % behind the rear axle
if hasB
    if rand(rs) < 0.75, rc.kf = u(0.6, 0.85); else, rc.kf = u(1.15, 1.3); end   % excludes the soil band 0.85-1.15
else
    rc.kf = 1;
end
rc.soilCaf = u(0.92, 1.08);
rc.soilCar = u(0.92, 1.08);
rc.profileSeed = 100000 + id;
rc.distSeed = 200000 + id;
rc.stepTurn = 1 + (rand(rs) > 0.5);   % step after turn 1 or 2

prof = change_detector.makeManeuverProfile(struct(), rc.profileSeed);
rc.duration = prof.t(end);
switch changeType
    case 'constant', rc.tStart = 0; rc.tEnd = 0;
    case 'step',     rc.tStart = prof.info.turns(rc.stepTurn).tEnd; rc.tEnd = rc.tStart;
    case 'ramp',     rc.tStart = 0; rc.tEnd = prof.t(end);
end
rc.file = string(sprintf('run_%05d.mat', id));
end
