function features = extractFeatures(x,config)

% Berechnet die Parameter in Samples anstelle von Sekunden
segmentSamples = round(config.segmentDuration*config.fs);
frameSamples = round(config.frameDuration*config.fs);
hopSamples = round(config.hopDuration*config.fs);
overlapSamples = frameSamples - hopSamples;

% Padded zu kurze Samples mit Nullen
numSamples = size(x,1);
numToPadFront = floor( (segmentSamples - numSamples)/2 );
numToPadBack = ceil( (segmentSamples - numSamples)/2 );
xPadded = [zeros(numToPadFront,1,'like',x); x; zeros(numToPadBack,1,'like',x)];

% ToDo: Hier bitte mit dem melSpectrogram-Befehl das Mel-Spektrogramm
%  von xPadded berechnen. Setzen Sie die 'WindowNormalization' auf false (bessere
% Trainingsergebnise). Das Mel-Spektrogramm sollte die Zeitdimension als
% Zeilen haben.

features = melSpectrogram(xPadded, ...
    config.fs, ...
    Window=hamming(frameSamples, "periodic"), ...
    OverlapLength=overlapSamples, ...
    FFTLength=config.FFTLength, ...
    NumBands=config.numBands, ...
    WindowNormalization=false)';

features = log10(features + 1e-9);