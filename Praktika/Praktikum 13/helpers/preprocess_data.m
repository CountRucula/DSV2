function [X,Y,ads] = preprocess_data(data_folder,config)
% Liest die Files im data_folder als audioDatastore
ads = audioDatastore(data_folder, ...
    'IncludeSubfolders',true, ...
    'FileExtensions','.wav', ...
    'LabelSource','foldernames');

% Definiert 20% aller Samples, welche nicht in der Commands-Liste sind als
% "unknown". Es werden nur 20 % verwendet um die Grösse der Daten zu
% reduzieren. Für eine bessere Performance sollten alle Daten verwendet
% werden
isCommand = ismember(ads.Labels,config.commands);
isUnknown = ~isCommand;
includeFraction = 0.2;
mask = rand(numel(ads.Labels),1) < includeFraction;
isUnknown = isUnknown & mask;
ads.Labels(isUnknown) = categorical("unknown");

% Entfernt alle Samples, welche nicht in der Commands-Liste oder "unknown"
% sind
ads = subset(ads,isCommand|isUnknown);

% Verwendet nur 5 % der Daten wenn so konfiguriert
if config.reduceDataset
    numUniqueLabels = numel(unique(ads.Labels));
    ads = splitEachLabel(ads,round(numel(ads.Files) / numUniqueLabels / 20));
end

% Setzt einen parallel Pool auf, um die Daten mit Multi-Core-Processing zu
% verarbeiten
if ~isempty(ver('parallel'))
    pool = gcp;
    numPar = numpartitions(ads,pool);
else
    numPar = 1;
end

% Geht durch alle Samples und extrahiert die Features
segmentSamples = round(config.segmentDuration*config.fs);
parfor ii = 1:numPar
    subds = partition(ads,numPar,ii);
    X = zeros(config.numHops,config.numBands,1,numel(subds.Files));
    for idx = 1:numel(subds.Files)
        x = read(subds);
        xPadded = [zeros(floor((segmentSamples-size(x,1))/2),1);x;zeros(ceil((segmentSamples-size(x,1))/2),1)];
        X(:,:,:,idx) = extractFeatures(xPadded,config);
    end
    XC{ii} = X;
end

% Setzt alle Features der verschiedenen Cores zu einem grossen Datensatz zusammen 
X = cat(4,XC{:});

% Entfernt nicht verwendete Klassen vom Categorical Vector
Y = removecats(ads.Labels);

end

