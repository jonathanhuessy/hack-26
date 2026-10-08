function x = turnFeatures(win, straight, fs)
%TURNFEATURES Feature vector of one turn window (contract 2); names in turnFeatureNames.
%   x = turnFeatures(win, straight, fs)
%     win       [delta Vx r ay] measured, marginS before turn entry to marginS after exit
%     straight  [delta Vx r ay] measured on the preceding straight (may have 0 rows)
%     fs        sample rate [Hz]
%   Uses only causal filters with fixed coefficients, sums, median and explicit DFTs on a
%   fixed frequency grid, so the Python port is line by line.
%   e = r - Vx*tan(delta)/L: yaw rate not explained by kinematics.
%   q = ay - Vx*r: lateral acceleration not explained by Vx*r (rear slip dynamics at the IMU).

L = 2.81;   % wheelbase [m], known geometry, not a property that changes
cfg = change_detector.turnTriggerConfig();
win = double(win); straight = double(straight);
N = size(win, 1);
nM = round(cfg.marginS*fs);
warm = round(2*fs);   % filter settling
use = (warm+1:N)';
core = (nM+1:N-nM)';

d = win(:,1); V = win(:,2); r = win(:,3); ay = win(:,4);
e = r - V.*tan(d)/L;
q = ay - V.*r;

bands = [0.2 0.6; 0.6 1.0; 1.0 1.5; 1.5 2.0; 2.0 3.0];
fGrid = 0.5:0.02:3;

Vt = mean(V(core));
bpE = bandLogPower(e, bands, fs, use);
bpQ = bandLogPower(q, bands, fs, use);
[fpkE, hE, zE] = spectralPeak(e(core), fGrid, fs);
[fpkQ, hQ] = spectralPeak(q(core), fGrid, fs);
[fAR, zAR] = ar2Mode(e, fs, use);

[gR, phR] = crossGain(d(use), r(use), [0.1 0.4; 0.4 0.8; 0.8 1.5], fs);
gR = gR/(Vt/L);

sel = core(abs(d(core)) > deg2rad(8));
if isempty(sel)
    ssGain = 1;
else
    ssGain = median(r(sel)./(V(sel).*tan(d(sel))/L));
end
vr = V(core).*r(core);
ayRatio = sum(ay(core).*vr)/max(sum(vr.^2), 1e-12);
s = sign(sum(r(core)));
oppSteer = mean(sign(d(core)) == -s & abs(d(core)) > deg2rad(3));

xs = zeros(1, 8);
if size(straight, 1) >= round(10*fs)
    ds = straight(:,1); Vs = straight(:,2); rs = straight(:,3); ays = straight(:,4);
    useS = (warm+1:size(straight, 1))';
    Vsm = mean(Vs);
    [gS, phS] = crossGain(ds(useS), rs(useS), [0.1 0.5], fs);
    [gSa, phSa] = crossGain(ds(useS), ays(useS), [0.1 0.5], fs);
    es = rs - Vs.*tan(ds)/L;
    bpS = bandLogPower(es, [1.0 2.0], fs, useS);
    [fpkS, hS] = spectralPeak(es(useS), fGrid, fs);
    xs = [Vsm, gS/(Vsm/L), phS, gSa/(Vsm^2/L), phSa, bpS, fpkS, hS];
end

x = [Vt, bpE, bpQ, fpkE, hE, zE, fpkQ, hQ, fAR, zAR, gR, phR, ssGain, ayRatio, oppSteer, xs];
end

function bp = bandLogPower(x, bands, fs, use)
% log10 of the mean power in each band (2nd-order Butterworth band-pass, causal).
bp = zeros(1, size(bands, 1));
x = x - x(1);
for i = 1:size(bands, 1)
    [b, a] = butter(2, bands(i,:)/(fs/2), 'bandpass');
    y = filter(b, a, x);
    bp(i) = log10(mean(y(use).^2) + 1e-12);
end
end

function X = dftAt(x, f, fs)
% Hann-windowed DFT of x at the frequencies f (row) [Hz].
n = numel(x);
k = (0:n-1)';
w = 0.5 - 0.5*cos(2*pi*k/(n - 1));
X = exp(-2i*pi*(f(:)/fs)*k')*((x(:) - mean(x)).*w);
end

function [fpk, peakHeight, zeta] = spectralPeak(x, f, fs)
% Peak of the periodogram on the grid f: frequency (parabolic interpolation on log power),
% height (log10 peak / median) and damping from the half-power bandwidth.
P = abs(dftAt(x, f, fs)).^2 + 1e-20;
[Pm, k] = max(P);
df = f(2) - f(1);
fpk = f(k);
if k > 1 && k < numel(P)
    l1 = log(P(k-1)); l2 = log(P(k)); l3 = log(P(k+1));
    den = l1 - 2*l2 + l3;
    if den < 0
        fpk = f(k) + 0.5*(l1 - l3)/den*df;
    end
end
peakHeight = log10(Pm/median(P));
kl = k; while kl > 1 && P(kl-1) >= Pm/2, kl = kl - 1; end
kr = k; while kr < numel(P) && P(kr+1) >= Pm/2, kr = kr + 1; end
zeta = (kr - kl + 1)*df/(2*fpk);
end

function [fn, zeta] = ar2Mode(x, fs, use)
% Natural frequency and damping of an AR(2) model fitted to x after band-pass 0.3-3 Hz
% and decimation by 5 (least squares).
[b, a] = butter(2, [0.3 3]/(fs/2), 'bandpass');
y = filter(b, a, x - x(1));
y = y(use(1):5:use(end));
fs2 = fs/5;
Phi = [y(2:end-1), y(1:end-2)];
c = (Phi'*Phi)\(Phi'*y(3:end));
a1 = c(1); a2 = c(2);
if a1^2 + 4*a2 < 0
    rad = sqrt(-a2);
    th = acos(min(max(a1/(2*rad), -1), 1));
    sig = -log(rad)*fs2; om = th*fs2;
    wn = sqrt(sig^2 + om^2);
    fn = wn/(2*pi); zeta = sig/wn;
else
    fn = 0; zeta = 1;
end
end

function [g, ph] = crossGain(x, y, bands, fs)
% Gain and phase of y/x per band from the summed cross-spectrum on a 0.02 Hz grid.
g = zeros(1, size(bands, 1)); ph = g;
for i = 1:size(bands, 1)
    f = bands(i,1):0.02:bands(i,2);
    X = dftAt(x, f, fs); Y = dftAt(y, f, fs);
    H = sum(Y.*conj(X))/max(sum(abs(X).^2), 1e-20);
    g(i) = abs(H); ph(i) = angle(H);
end
end
