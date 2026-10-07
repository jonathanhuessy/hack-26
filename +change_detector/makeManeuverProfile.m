function prof = makeManeuverProfile(cfg, seed)
%MAKEMANEUVERPROFILE Front wheel angle and speed profile for one run (open loop).
%   prof = makeManeuverProfile(cfg, seed) returns 3 swaths, each
%   straight -> decelerate -> end-of-row turn -> accelerate, sampled at cfg.fs.
%   Missing cfg fields take the defaults in fillDefaults; cfg ranges are
%   [min max] and are drawn per run with the given seed (own RandStream).
%
%   Turns: omega turn (three arcs, lands on the next lane, working width w) or
%   U-turn (one semicircle, skips a lane, w = 2R). Net direction alternates.
%   Steering ramps are raised-cosine steps with peak rate cfg.rampRateRadps.
%
%   prof fields: t, delta [rad], Vx [m/s], segment (1 straight, 2 speed
%   transition, 3 turn), swath, turnIdx (0 outside turns), fs, info.

if nargin < 1, cfg = struct(); end
if nargin < 2, seed = 1; end
cfg = fillDefaults(cfg);

fs = cfg.fs;
rs = RandStream('mt19937ar', 'Seed', seed);
u = @(r) r(1) + (r(2) - r(1))*rand(rs);

Vs    = u(cfg.straightSpeedKmh)/3.6;
Vt    = u(cfg.turnSpeedKmh)/3.6;
R     = u(cfg.turnRadiusM);
w     = u(cfg.workingWidthM);
opStd = deg2rad(u(cfg.operatorStdDeg));
dir0  = 2*(rand(rs) > 0.5) - 1;
nSw   = cfg.nSwaths;
Ls    = cfg.straightLengthM*(1 + cfg.straightLengthJitter*(2*rand(rs, nSw, 1) - 1));
isOmega = rand(rs, nSw, 1) < cfg.omegaFraction;

nS = @(T) max(1, round(T*fs));
V = cell(nSw, 2); D = cell(nSw, 2); seg = cell(nSw, 2); tIdx = cell(nSw, 2);
swath = cell(nSw, 2);
turnInfo = repmat(struct('tStart', 0, 'tEnd', 0, 'dir', 0, 'isOmega', false, 'R', R, 'beta', 0), nSw, 1);
nDone = 0;

for k = 1:nSw
    dir = dir0*(-1)^(k - 1);

    n = nS(Ls(k)/Vs);
    V{k,1} = Vs*ones(n, 1); D{k,1} = zeros(n, 1);
    seg{k,1} = ones(n, 1);  tIdx{k,1} = zeros(n, 1);

    nd = nS((Vs - Vt)/cfg.accelMps2);
    Vdec = Vs + (Vt - Vs)*(1:nd)'/nd;

    [dTurn, beta] = turnSteering(isOmega(k), dir, R, w, Vt, cfg.wheelbaseM, cfg.understeerGrad, cfg.rampRateRadps, fs);
    nt = numel(dTurn);

    na = nS((Vs - Vt)/cfg.accelMps2);
    Vacc = Vt + (Vs - Vt)*(1:na)'/na;

    % column 1: straight, column 2: decel + turn + accel
    V{k,2} = [Vdec; Vt*ones(nt, 1); Vacc];
    D{k,2} = [zeros(nd, 1); dTurn; zeros(na, 1)];
    seg{k,2} = [2*ones(nd, 1); 3*ones(nt, 1); 2*ones(na, 1)];
    tIdx{k,2} = [zeros(nd, 1); k*ones(nt, 1); zeros(na, 1)];

    nDone = nDone + n;
    turnInfo(k).tStart = (nDone + nd)/fs;
    turnInfo(k).tEnd = (nDone + nd + nt)/fs;
    turnInfo(k).dir = dir; turnInfo(k).isOmega = isOmega(k); turnInfo(k).beta = beta;
    nDone = nDone + nd + nt + na;
    for j = 1:2
        swath{k,j} = k*ones(numel(V{k,j}), 1);
    end
end

Vx = timeCat(V); delta = timeCat(D); segment = timeCat(seg);
turnIdx = timeCat(tIdx); swathIdx = timeCat(swath);
N = numel(Vx);
t = (0:N-1)'/fs;

% operator corrections: the operator holds the row, so the heading wander is a band-limited
% process that is zero at the start and around the turns; steering is its kinematic derivative.
[bf, af] = butter(3, cfg.operatorBandHz/(fs/2), 'bandpass');   % steep roll-off: steering is the derivative
warm = round(10*fs);
psiW = filter(bf, af, randn(rs, N + warm, 1));
psiW = psiW(warm+1:end);
inTurn = double(segment == 3);
inTurn(1:round(3*fs)) = 1;
mask = 1 - movmean(movmean(movmax(inTurn, round(2*fs)), round(fs)), round(fs));   % smooth: steering needs mask''
corr = cfg.wheelbaseM/Vs*gradient(psiW.*mask, 1/fs);
corr = corr*opStd/std(corr(mask > 0.5));
delta = delta + corr;

dMax = cfg.deltaMaxRad;
delta = max(min(delta, dMax), -dMax);

prof.t = t; prof.delta = delta; prof.Vx = Vx;
prof.segment = uint8(segment); prof.swath = uint8(swathIdx); prof.turnIdx = uint8(turnIdx);
prof.fs = fs;
prof.info = struct('seed', seed, 'Vs', Vs, 'Vt', Vt, 'R', R, 'w', w, 'operatorStdRad', opStd, ...
    'straightLengthM', Ls, 'turns', turnInfo, 'wheelbaseM', cfg.wheelbaseM);
end

function x = timeCat(c)
% Concatenate an nSwaths x 2 cell array in time order (swath by swath).
c = c.';
x = vertcat(c{:});
end

function [delta, beta] = turnSteering(isOmega, dir, R, w, Vt, L, Kus, rate, fs)
% Piecewise-constant steering plateaus joined by raised-cosine ramps.
% Plateau angle from the nominal steady-state yaw gain, r = Vt*delta/(L + Kus*Vt^2) = Vt/R,
% so the open-loop turn ends at 180 deg heading and the swaths stay parallel.
dT = (L + Kus*Vt^2)/R;
if isOmega
    beta = acos((w/R + 2)/4);
    plateau = dir*[-1 1 -1]*dT;
    arcT = R*[beta, pi + 2*beta, beta]/Vt;
else
    beta = 0;
    plateau = dir*dT;
    arcT = R*pi/Vt;
end
jumps = diff([0 plateau 0]);
trj = (pi/2)*abs(jumps)/rate;
b = trj(1)/2 + [0 cumsum(arcT)];
Tturn = b(end) + trj(end)/2;
t = (0:ceil(Tturn*fs))'/fs;
delta = zeros(size(t));
for j = 1:numel(jumps)
    x = min(max((t - b(j))/trj(j) + 0.5, 0), 1);
    delta = delta + jumps(j)*0.5*(1 - cos(pi*x));
end
end

function cfg = fillDefaults(cfg)
p0 = change_detector.nominalParams();
d.fs = 100;
d.nSwaths = 3;
d.straightSpeedKmh = [13 16];
d.turnSpeedKmh = [4 6];
d.turnRadiusM = [8.5 11];
d.workingWidthM = [3 6];
d.straightLengthM = 100;
d.straightLengthJitter = 0.2;
d.omegaFraction = 0.7;
d.operatorStdDeg = [0.5 1.5];
d.operatorBandHz = [0.1 0.5];   % band-pass: an operator holds heading, so no slow steering drift
d.accelMps2 = 0.5;
d.rampRateRadps = deg2rad(22);   % peak steering rate, below the 25 deg/s limit
d.wheelbaseM = p0.L;
d.understeerGrad = p0.m/p0.L*(p0.lr/p0.Caf - p0.lf/p0.Car);   % nominal Kus [rad/(m/s^2)]
d.deltaMaxRad = p0.deltaMaxRad;
for f = fieldnames(d)'
    if ~isfield(cfg, f{1}), cfg.(f{1}) = d.(f{1}); end
end
end
