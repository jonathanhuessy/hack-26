function params = arion_630_parameters()
% Newly Tuned Parameters to fit the frequency response, obtained using
% parameter optimization. The parameters match the vehicle's physical
% dimensions better.
params.vehicleMassKg = 10178.6;
params.frontTireCorneringStiffnessNprad = 356356.4449;
params.rearTireCorneringStiffnessNprad = 270000;
params.vehicleWheelbaseM = 2.81;
params.chassisLengthM = 4.764;
params.rearAxleToCGM = 1.1538;
params.frontAxleToCGM = params.vehicleWheelbaseM - params.rearAxleToCGM;
params.steeringAngleLimitRad = deg2rad(19); % range (stationary): [-40°,32.8°], range (moving): [-19.4°,25.9°]
params.steeringCmdTs = 0.1;
params.steeringBandwidthHz = 1.12;
params.steeringDamping = 0.704;
params.steeringCmdConstantBiasErrorRad = deg2rad(0);
params.antennaToRearAxleXM = -0.1; % This corresponds to the 'alx' TAP
params.antennaToGroundZM = 3.162; % This corresponds to the 'alz' TAP
params.steeringSlewRateLimitRadps = deg2rad(25.0);
params.vehicleZInertiaKgmm = params.vehicleMassKg*params.frontAxleToCGM*...
    params.rearAxleToCGM;
params.vehicleXInertiaKgmm = 3242;
params.rollDampingKgmmps = 22479;
params.rollStiffnessKgmmpsps = 2561167;
params.cgheightM = 1.25;
params.frontTireRelaxationLengthM = 1.10369051;
params.rearTireRelaxationLengthM = realmax;
params.vehicleHeightM = 3.58;
params.frontGroundClearanceM = 0.647;
params.rearGroundClearanceM = 0.611;
params.vehicleWidthMaxM = 3.0; % source from Claas wesbite
params.vehicleXInertiaKgmm = params.vehicleMassKg*params.cgheightM*...
    (params.vehicleHeightM - params.cgheightM);
params.understeerCoefAdjLookupTableBreakpoints = [-0.436 -0.174 -0.087 0 0.087 0.174 0.436]; % default axion 850 values
params.understeerCoefAdjmentLookupTableValue = 0.0*[-0.02 -0.023 -0.013 0 0.013 0.023 0.02]; % default axion 850 values
params.yawRateSteadyStateAdjGain = 1; %3.5; % default axion 930 values
params.yawRateDynamicsAdjTFNum =  1; % [390.1143 1.8932e+03 7.7434e+03 3.8095e+03]; % default axion 930 values
params.yawRateDynamicsAdjTFDen =  1; %[1 475.3060 2.8130e+03 2.2346e+04 1.1602e+04]; % default axion 930 values
params.steeringActuator3rdOrderContSs = tf2ss_observable(tf([1.347 -71.03 1219], [1 20.12 296.5 1213]));%option2 = tf2ss_observable(tf([2.443 -33.81 976.6],[1 22.68 255.9 976.7]));%option1 = tf2ss_observable(tf([2.81 -40.95 1156],[1 39.7 216.9 1156])); %old = tf2ss_observable(tf([1.613,-33.04,934.9], [1,37.07,203.5,934.9]));
params.steeringActuatorContSs = ss([-9.372],[9.372],[1],[0]);
params.steeringActuatorDelayS = 0.127;
params.tireWidthMaxValueM = 0.711; % source from Class website
params.chassisHeightM = 1.5; % needs correction when data available
params.axleRadiusM = 0.05; % needs correction when data available
params.cabMassKg = 100;
end