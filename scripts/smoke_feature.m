% SMOKE_FEATURE  Evidence that the dataset carries a usable signal (P8). Not the final pipeline.
%   Two simple per-turn features on noisy measurements, median over the 3 turns of a run:
%     lag   delay of the delta -> r cross-correlation peak [s]
%     fpk   frequency of the yaw-rate spectrum peak between 0.6 and 3 Hz [Hz]
%   Uses nPerClass constant train runs per class. Prints the AUC for nominal vs the largest A
%   changes and vs B, then a cross-validated linear discriminant on band powers, and writes
%   data/smoke_feature.png.
projectRoot = fileparts(fileparts(mfilename('fullpath')));
addpath(projectRoot);
nPerClass = 50;
idx = load(fullfile(projectRoot, 'data', 'index.mat')).idx;
idx = idx(idx.split == "train" & idx.changeType == "constant", :);

classes = ["nominal", "A", "B"];
feat = struct('lag', [], 'fpk', [], 'bp', [], 'cls', strings(0, 1), 'dm', [], 'kf', []);
edges = [0.2 0.6 1 1.5 2 3 5];   % band edges [Hz] for the log band powers
for c = classes
    rows = find(idx.class == c, nPerClass);
    for k = rows'
        r = load(fullfile(projectRoot, 'data', 'runs', idx.file(k)));
        tt = change_detector.addSensorNoise(r.tt, [], 5000 + idx.id(k));
        fs = r.meta.fs;
        lag = nan(1, 3); fpk = nan(1, 3); bp = zeros(2, numel(edges) - 1);
        for tn = 1:3
            m = tt.turnIdx == tn;
            dl = double(tt.deltaMeas(m)); rr = double(tt.rMeas(m));
            [xc, lags] = xcorr(rr - mean(rr), dl - mean(dl), round(1.5*fs), 'normalized');
            [~, ip] = max(xc); lag(tn) = lags(ip)/fs;
            [pxx, f] = pwelch(rr - movmean(rr, round(fs)), hann(round(8*fs)), [], 2^12, fs);
            sel = f >= 0.6 & f <= 3; [~, jp] = max(pxx(sel)); fsel = f(sel); fpk(tn) = fsel(jp);
            ay = double(tt.ayMeas(m)); [pay, ~] = pwelch(ay - movmean(ay, round(fs)), hann(round(8*fs)), [], 2^12, fs);
            for b = 1:numel(edges) - 1
                fb = f >= edges(b) & f < edges(b + 1);
                bp(1, b) = bp(1, b) + sum(pxx(fb))/3; bp(2, b) = bp(2, b) + sum(pay(fb))/3;
            end
        end
        feat.lag(end+1) = median(lag); feat.fpk(end+1) = median(fpk); feat.bp(end+1, :) = log(bp(:))';
        feat.cls(end+1, 1) = c; feat.dm(end+1) = idx.dm(k); feat.kf(end+1) = idx.kf(k);
    end
end

for f = ["lag", "fpk", "dm", "kf"], feat.(f) = feat.(f)(:); end
auc = @(a, b) (sum(sum(a(:) > b(:)')) + 0.5*sum(sum(a(:) == b(:)')))/(numel(a)*numel(b));   % P(a > b)
nom = feat.cls == "nominal"; A = feat.cls == "A"; B = feat.cls == "B";
bigA = A & feat.dm >= 1500; midA = A & feat.dm >= 750 & feat.dm < 1500; Blow = B & feat.kf < 0.75;
sep = @(x, g) max(auc(x(g), x(nom)), 1 - auc(x(g), x(nom)));   % 0.5 = no separation, 1 = perfect
fprintf('AUC against nominal (0.5 = none, 1 = perfect), n = %d per class\n', nPerClass);
fprintf('%-26s %8s %8s\n', 'group', 'lag', 'fpk');
for g = {bigA, 'A dm >= 1500 kg'; midA, 'A 750 <= dm < 1500 kg'; Blow, 'B kf < 0.75'; B, 'B all'}'
    fprintf('%-26s %8.2f %8.2f\n', g{2}, sep(feat.lag, g{1}), sep(feat.fpk, g{1}));
end

% Simple multivariate check: 12 log band powers (r and ay, 6 bands), linear discriminant, 5-fold CV
rng(1);
pred = kfoldPredict(fitcdiscr(feat.bp, feat.cls, 'KFold', 5));
fprintf('\nLDA on 12 log band powers, 3 classes (nominal, A, B), 5-fold CV: accuracy %.2f\n', mean(pred == feat.cls));
sel = nom | bigA;
pred2 = kfoldPredict(fitcdiscr(feat.bp(sel, :), feat.cls(sel), 'KFold', 5));
fprintf('nominal vs A with dm >= 1500 kg: accuracy %.2f (n = %d)\n', mean(pred2 == feat.cls(sel)), nnz(sel));

fig = figure('Position', [100 100 1100 400]); tl = tiledlayout(fig, 1, 2);
names = {'lag [s]', 'fpk [Hz]'}; vals = {feat.lag, feat.fpk};
for q = 1:2
    nexttile(tl); hold on; grid on;
    histogram(vals{q}(nom), 15, 'Normalization', 'probability');
    histogram(vals{q}(A), 15, 'Normalization', 'probability');
    histogram(vals{q}(B), 15, 'Normalization', 'probability');
    legend('nominal', 'A', 'B'); xlabel(names{q}); title(['Per-run median ' names{q}]);
end
exportgraphics(fig, fullfile(projectRoot, 'data', 'smoke_feature.png'), 'Resolution', 110);
