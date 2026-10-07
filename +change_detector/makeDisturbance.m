function dist = makeDisturbance(t, cfg, seed)
%MAKEDISTURBANCE Soil and terrain disturbance as lateral force and yaw moment.
%   dist = makeDisturbance(t, cfg, seed) returns zero-mean band-limited noise
%   Fyd [N] and Mzd [Nm] on the time vector t (uniform, 100 Hz). Zero mean on
%   purpose: a constant side force would turn the open-loop run into a circle.
%   cfg.FyStdN, cfg.MzStdNm, cfg.bandHz override the defaults.

if nargin < 2 || isempty(cfg), cfg = struct(); end
if ~isfield(cfg, 'FyStdN'),  cfg.FyStdN = 800; end
if ~isfield(cfg, 'MzStdNm'), cfg.MzStdNm = 400; end
if ~isfield(cfg, 'bandHz'),  cfg.bandHz = [0.05 2]; end

fs = 1/(t(2) - t(1));
rs = RandStream('mt19937ar', 'Seed', seed);
[b, a] = butter(2, cfg.bandHz/(fs/2), 'bandpass');
N = numel(t);
warm = round(60*fs);

nz = filter(b, a, randn(rs, N + warm, 2));
nz = nz(warm+1:end, :);
nz = nz./std(nz);

dist.Fyd = cfg.FyStdN*nz(:,1);
dist.Mzd = cfg.MzStdNm*nz(:,2);
end
