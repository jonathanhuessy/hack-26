function sys = bicycleMatrices(p, Vx)
%BICYCLEMATRICES Linear 3-state bicycle model with front tire relaxation.
%   States   x = [ydot; r; alphaF]  (lateral velocity, yaw rate, front slip angle)
%   Inputs   u = delta (front wheel angle), d = [Fyd; Mzd] (disturbance force, moment)
%   Outputs  [r; ay_imu], ay_imu = ydot_dot + Vx*r + xi*r_dot (xi: IMU ahead of the CG)
%
%   Conventions: y and delta positive left, r positive counter-clockwise.
%     Fyf = Caf*alphaF,  alphaF_dot = (Vx*delta - ydot - lf*r - Vx*alphaF)/sigmaF
%     Fyr = -Car*(ydot - lr*r)/Vx   (rear tire: no relaxation)
%     m*(ydot_dot + Vx*r) = Fyf + Fyr + Fyd,   Izz*r_dot = lf*Fyf - lr*Fyr + Mzd
%   Vx must be > 0.

m = p.m; Izz = p.Izz; lf = p.lf; lr = p.lr;
Caf = p.Caf; Car = p.Car; sig = p.sigmaF;

A = [ -Car/(m*Vx),      Car*lr/(m*Vx) - Vx,   Caf/m;
       lr*Car/(Izz*Vx), -lr^2*Car/(Izz*Vx),   lf*Caf/Izz;
      -1/sig,           -lf/sig,              -Vx/sig ];
B  = [0; 0; Vx/sig];
Bd = [1/m 0; 0 1/Izz; 0 0];

xi = p.imuXFromRearAxleM - lr;
Cr  = [0 1 0];
Cay = A(1,:) + [0 Vx 0] + xi*A(2,:);
C = [Cr; Cay];
Dd = [0 0; Bd(1,:) + xi*Bd(2,:)];

sys.A = A; sys.B = B; sys.Bd = Bd; sys.C = C; sys.Dd = Dd;
end
