function p = applyScenario(p0, scen)
%APPLYSCENARIO Turn a scenario into a consistent bicycle-model parameter set.
%   p = applyScenario(p0, scen) with optional scen fields (defaults in brackets):
%     addedMassKg  [0]  point mass added to the tractor
%     mountXM      [-1.2] its position along the wheelbase, measured from the
%                         rear axle, positive forward (rear hitch < 0, front weight > L)
%     kf           [1]  front cornering stiffness factor, Caf = kf*Caf0
%     soilCaf      [1]  soil factor on Caf
%     soilCar      [1]  soil factor on Car
%
%   m, lf, lr and Izz are always updated together: the CG moves with the
%   added mass and Izz follows the parallel-axis rule (point mass).

arguments
    p0 struct
    scen struct = struct()
end

dm    = getField(scen, 'addedMassKg', 0);
xm    = getField(scen, 'mountXM', -1.2);
kf    = getField(scen, 'kf', 1);
soilF = getField(scen, 'soilCaf', 1);
soilR = getField(scen, 'soilCar', 1);

p = p0;
p.m = p0.m + dm;

% CG position from the rear axle before and after the added mass
xcg0 = p0.lr;
xcg1 = (p0.m*xcg0 + dm*xm) / p.m;
p.lr = xcg1;
p.lf = p0.L - xcg1;

p.Izz = p0.Izz + p0.m*(xcg0 - xcg1)^2 + dm*(xm - xcg1)^2;

p.Caf = p0.Caf * kf * soilF;
p.Car = p0.Car * soilR;
end

function v = getField(s, name, default)
if isfield(s, name)
    v = s.(name);
else
    v = default;
end
end
