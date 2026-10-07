function p = nominalParams()
%NOMINALPARAMS Nominal bicycle-model parameters of the Arion 630.
%   Wraps arion_630_parameters and keeps only the fields the 3-state
%   bicycle model needs. Lengths in m, mass in kg, stiffness in N/rad.

a = arion_630_parameters();

p.m          = a.vehicleMassKg;
p.L          = a.vehicleWheelbaseM;
p.lf         = a.frontAxleToCGM;
p.lr         = a.rearAxleToCGM;
p.Izz        = a.vehicleZInertiaKgmm;
p.Caf        = a.frontTireCorneringStiffnessNprad;
p.Car        = a.rearTireCorneringStiffnessNprad;
p.sigmaF     = a.frontTireRelaxationLengthM;   % front tire relaxation length
p.deltaMaxRad = a.steeringAngleLimitRad;
p.deltaRateMaxRadps = a.steeringSlewRateLimitRadps;
p.imuXFromRearAxleM = 0;   % assumption: IMU above the rear axle, fixed to the chassis (not the CG)
end
