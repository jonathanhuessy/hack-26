function tt = addSensorNoise(tt, cfg, seed)
%ADDSENSORNOISE Add the four measured signals (with noise and biases) to a run timetable.
%   tt = addSensorNoise(tt, cfg, seed) appends deltaMeas, VxMeas, rMeas, ayMeas.
%   The clean signals stay in tt. Biases are constant per run, vibration is band-pass
%   noise on the accelerometer. cfg fields override the defaults below (units: rad, m, s).

if nargin < 2 || isempty(cfg), cfg = struct(); end
d = struct('gyroStd', 0.01, 'gyroBiasStd', 0.002, ...        % rad/s
           'accelStd', 0.2, 'accelBiasStd', 0.05, ...        % m/s^2
           'vibStd', 0.15, 'vibBandHz', [8 30], ...          % m/s^2, engine and terrain vibration
           'steerStd', deg2rad(0.2), 'steerBiasStd', deg2rad(0.1), ...
           'vxStd', 0.02);                                   % m/s
for f = fieldnames(d)'
    if ~isfield(cfg, f{1}), cfg.(f{1}) = d.(f{1}); end
end

rs = RandStream('mt19937ar', 'Seed', seed);
N = height(tt);
fs = 1/seconds(tt.Time(2) - tt.Time(1));

[b, a] = butter(2, cfg.vibBandHz/(fs/2), 'bandpass');
vib = filter(b, a, randn(rs, N, 1));
vib = cfg.vibStd*vib/std(vib);

tt.deltaMeas = single(double(tt.delta) + cfg.steerBiasStd*randn(rs) + cfg.steerStd*randn(rs, N, 1));
tt.VxMeas    = single(double(tt.Vx) + cfg.vxStd*randn(rs, N, 1));
tt.rMeas     = single(double(tt.r) + cfg.gyroBiasStd*randn(rs) + cfg.gyroStd*randn(rs, N, 1));
tt.ayMeas    = single(double(tt.ay) + cfg.accelBiasStd*randn(rs) + cfg.accelStd*randn(rs, N, 1) + vib);
end
