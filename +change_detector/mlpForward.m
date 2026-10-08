function [P, dm, kf] = mlpForward(X, W)
%MLPFORWARD Forward pass of the exported classifier and regressor (contract 3).
%   [P, dm, kf] = mlpForward(X, W)
%     X    n x nFeat raw feature rows, columns in the order of W.featureNames
%     W    struct loaded from models/export/weights.mat
%     P    n x 4 class probabilities (order W.classNames: nominal, A, B, AB)
%     dm   n x 1 added mass [kg], kf n x 1 front stiffness factor
%   Layers act on column vectors: h1 = relu(W1*z + b1), h2 = relu(W2*h1 + b2),
%   P = softmax(W3*h2 + b3); the regressor is the same with V, c and a linear output
%   [dm; kf] = V3*g2 + c3. z = (x - mu)./sigma. Same maths as pi/model.py.

Z = ((X - W.mu)./W.sigma)';
H = max(W.W1*Z + W.b1, 0);
H = max(W.W2*H + W.b2, 0);
S = W.W3*H + W.b3;
S = exp(S - max(S, [], 1));
P = (S./sum(S, 1))';

G = max(W.V1*Z + W.c1, 0);
G = max(W.V2*G + W.c2, 0);
R = W.V3*G + W.c3;
dm = R(1,:)';
kf = R(2,:)';
end
