function [cls, p, dm, kf] = aggregateVerdict(P, dmTurn, kfTurn)
%AGGREGATEVERDICT Verdict from the outputs of the last K turns (contract 4).
%   [cls, p, dm, kf] = aggregateVerdict(P, dmTurn, kfTurn)
%     P        K x 4 class probabilities of the last K turns (order nominal, A, B, AB)
%     dmTurn   K x 1 regressor outputs [kg], kfTurn K x 1
%     cls      0 nominal, 1 A, 2 B, 3 A+B: argmax of the mean probabilities
%     p        1 x 4 mean probabilities
%     dm, kf   median over the K turns, gated by cls: dm = 0 unless cls has A, kf = 1 unless cls has B

p = mean(P, 1);
[~, k] = max(p);
cls = k - 1;
dm = 0; kf = 1;
if cls == 1 || cls == 3, dm = median(dmTurn); end
if cls == 2 || cls == 3, kf = median(kfTurn); end
end
