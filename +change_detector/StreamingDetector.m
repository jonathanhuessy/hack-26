classdef StreamingDetector < matlab.System
    %STREAMINGDETECTOR Sample-by-sample change detector (contract 4).
    %   det = change_detector.StreamingDetector('W', load('models/export/weights.mat'));
    %   [cls, p, dm, kf, clsTurn, newVerdict] = det(u)   % u = [delta Vx r ay], one sample
    %   Same maths as the offline path: findTurns (trigger), turnWindow, turnFeatures,
    %   mlpForward, aggregateVerdict over the last K turns. A verdict is due cfg.marginS after
    %   turn exit (end of the window); newVerdict is 1 on that sample only. Outputs hold the
    %   latest verdict in between (cls 0 nominal, 1 A, 2 B, 3 A+B; dm [kg]; kf; clsTurn = class
    %   of the latest turn alone, -1 before the first turn).
    %   Runs in MATLAB loops and in a MATLAB System block (Interpreted execution).

    properties (Nontunable)
        W = struct()          % weights struct from models/export/weights.mat
        fs = 100              % sample rate [Hz]
        K = 3                 % turns in the verdict
        bufferS = 150         % ring buffer length [s]; must hold window + preceding straight
    end

    properties (Access = private)
        cfg; sel; nBuf; holdN; nM; nS
        bLp; aLp; zLp
        buf; n
        inTurn; quiet; iEntry; iEntryPending; iExitPending
        P; dmT; kfT; nTurns
        out
    end

    methods
        function obj = StreamingDetector(varargin)
            setProperties(obj, nargin, varargin{:});
        end
    end

    methods (Access = protected)
        function setupImpl(obj)
            obj.cfg = change_detector.turnTriggerConfig();
            [~, obj.sel] = ismember(strsplit(obj.W.featureNames, ','), change_detector.turnFeatureNames());
            obj.nBuf = round(obj.bufferS*obj.fs);
            obj.holdN = round(obj.cfg.holdS*obj.fs);
            obj.nM = round(obj.cfg.marginS*obj.fs);
            obj.nS = round(obj.cfg.straightS*obj.fs);
            [obj.bLp, obj.aLp] = butter(2, obj.cfg.lowpassHz/(obj.fs/2));
        end

        function resetImpl(obj)
            obj.zLp = [0; 0];
            obj.buf = zeros(obj.nBuf, 4);
            obj.n = 0;
            obj.inTurn = false; obj.quiet = 0; obj.iEntry = 0; obj.iEntryPending = 0; obj.iExitPending = 0;
            obj.P = zeros(0, 4); obj.dmT = zeros(0, 1); obj.kfT = zeros(0, 1); obj.nTurns = 0;
            obj.out = {0, [1 0 0 0], 0, 1, -1};
        end

        function [cls, p, dm, kf, clsTurn, newVerdict] = stepImpl(obj, u)
            u = double(u(:)');
            obj.n = obj.n + 1;
            k = obj.n;
            obj.buf(mod(k - 1, obj.nBuf) + 1, :) = u;

            % trigger: same low-pass (direct form II transposed, as filter) and thresholds as findTurns
            b = obj.bLp; a = obj.aLp; z = obj.zLp;
            y = b(1)*u(1) + z(1);
            obj.zLp = [b(2)*u(1) + z(2) - a(2)*y; b(3)*u(1) - a(3)*y];
            dAbs = abs(y);
            if ~obj.inTurn
                if dAbs > obj.cfg.entryRad
                    obj.inTurn = true; obj.iEntry = k; obj.quiet = 0;
                end
            else
                if dAbs < obj.cfg.exitRad, obj.quiet = obj.quiet + 1; else, obj.quiet = 0; end
                if obj.quiet >= obj.holdN
                    iExit = k - obj.holdN + 1;
                    seg = obj.bufRows(obj.iEntry:iExit);
                    heading = sum(seg(:, 3))/obj.fs;
                    if abs(heading) >= obj.cfg.minHeadingRad
                        obj.iExitPending = iExit;
                        obj.iEntryPending = obj.iEntry;
                    end
                    obj.inTurn = false;
                end
            end

            newVerdict = false;
            if obj.iExitPending > 0 && k == obj.iExitPending + obj.nM
                obj.evaluateTurn();
                obj.iExitPending = 0;
                newVerdict = true;
            end
            [cls, p, dm, kf, clsTurn] = obj.out{:};
        end

        function num = getNumOutputsImpl(~), num = 6; end
        function num = getNumInputsImpl(~), num = 1; end
        function varargout = getOutputSizeImpl(~), varargout = {[1 1], [1 4], [1 1], [1 1], [1 1], [1 1]}; end
        function varargout = getOutputDataTypeImpl(~), varargout = {'double', 'double', 'double', 'double', 'double', 'logical'}; end
        function varargout = isOutputComplexImpl(~), varargout = repmat({false}, 1, 6); end
        function varargout = isOutputFixedSizeImpl(~), varargout = repmat({true}, 1, 6); end
        function varargout = getOutputNamesImpl(~), varargout = {'cls', 'p', 'dm', 'kf', 'clsTurn', 'newVerdict'}; end
        function name = getInputNamesImpl(~), name = 'u'; end
    end

    methods (Access = private)
        function x = bufRows(obj, idx)
            % Samples with absolute indices idx (must still be in the ring buffer).
            assert(idx(1) > obj.n - obj.nBuf, 'StreamingDetector: ring buffer too short (bufferS)');
            x = obj.buf(mod(idx - 1, obj.nBuf) + 1, :);
        end

        function evaluateTurn(obj)
            % Same window as turnWindow: the turn and margins, and the straight before it.
            i0 = max(1, obj.iEntryPending - obj.nM);
            i1 = obj.iExitPending + obj.nM;
            win = obj.bufRows(i0:i1);
            first = max(1, obj.n - obj.nBuf + 1);
            before = obj.bufRows(first:i0-1);
            cand = find(before(:, 2) > obj.cfg.straightMinVx);
            cand = cand(max(1, numel(cand) - obj.nS + 1):end);
            straight = before(cand, :);

            x = change_detector.turnFeatures(win, straight, obj.fs);
            [Pt, dmt, kft] = change_detector.mlpForward(x(obj.sel), obj.W);
            obj.P(end+1, :) = Pt; obj.dmT(end+1, 1) = dmt; obj.kfT(end+1, 1) = kft;
            obj.nTurns = obj.nTurns + 1;
            w = max(1, obj.nTurns - obj.K + 1):obj.nTurns;
            [cls, p, dm, kf] = change_detector.aggregateVerdict(obj.P(w, :), obj.dmT(w), obj.kfT(w));
            [~, kc] = max(Pt);
            obj.out = {cls, p, dm, kf, kc - 1};
        end
    end
end
