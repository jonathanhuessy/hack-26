function sc = demoScenario(name)
%DEMOSCENARIO Fixed demo scenario: 6 swaths, nominal for the first 2 turns, then a step change.
%   sc = demoScenario(name), name 'A' (+1500 kg on the rear hitch), 'B' (k_f = 0.7) or 'AB' (both).
%   The change happens at the end of turn 2, so turns 3 to 6 see the changed tractor.
%   sc fields: name, prof, dist, pStart, pEnd, sched (tStart = tEnd), dm, kf, changeTurn.
%   Used by scripts/run_demo_replay.m (offline reference) and the Simulink and Pi demos.

p0 = change_detector.nominalParams();
switch name
    case 'A',  dm = 1500; kf = 1;
    case 'B',  dm = 0;    kf = 0.7;
    case 'AB', dm = 1500; kf = 0.7;
    otherwise, error('unknown demo scenario %s', name);
end
seed = 424242;   % fixed: the same maneuver for all three scenarios
prof = change_detector.makeManeuverProfile(struct('nSwaths', 6), seed);
dist = change_detector.makeDisturbance(prof.t, struct(), seed + 1);
changeTurn = 2;

sc.name = name;
sc.prof = prof;
sc.dist = dist;
sc.pStart = p0;
sc.pEnd = change_detector.applyScenario(p0, struct('addedMassKg', dm, 'mountXM', -1.2, 'kf', kf));
sc.sched = struct('tStart', prof.info.turns(changeTurn).tEnd, 'tEnd', prof.info.turns(changeTurn).tEnd);
sc.dm = dm;
sc.kf = kf;
sc.changeTurn = changeTurn;
end
