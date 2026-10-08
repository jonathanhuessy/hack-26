function x = selectedTurnFeatures(win, straight, fs)
%SELECTEDTURNFEATURES Return H1's 18 deployable features from [delta Vx r].
%   The exploratory turnFeatures function retains a fourth ay column. The
%   selected set does not use ay, so this adapter keeps the exploratory
%   reference unchanged while exposing the three-input deployment contract.

if size(win, 2) ~= 3 || size(straight, 2) ~= 3
    error('change_detector:selectedTurnFeatures:InputSize', ...
        'win and straight must have columns [delta Vx r]');
end
fullWin = [win, zeros(size(win, 1), 1)];
fullStraight = [straight, zeros(size(straight, 1), 1)];
allNames = change_detector.turnFeatureNames();
selectedNames = change_detector.selectedFeatureNames();
[ok, columns] = ismember(selectedNames, allNames);
if ~all(ok)
    error('change_detector:selectedTurnFeatures:FeatureName', ...
        'Selected feature name is not present in turnFeatureNames');
end
full = change_detector.turnFeatures(fullWin, fullStraight, fs);
x = full(columns);
end
