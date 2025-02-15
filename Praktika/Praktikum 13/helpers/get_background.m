function [Xbkg,ads] = get_background(data_folder,config)
% Teilt die Files im Background-Ordner in 1s Samples mit einem zufälligen
% Gain
ads = audioDatastore(data_folder);
numBkgClips = 4000;

% Verwendet nur 5 % der Daten wenn so konfiguriert
if config.reduceDataset
    numBkgClips = numBkgClips/20;
end

numBkgFiles = numel(ads.Files);
numClipsPerFile = histcounts(1:numBkgClips,linspace(1,numBkgClips,numBkgFiles+1));
Xbkg = zeros(config.numHops,config.numBands,1,numBkgClips,'single');
bkgAll = readall(ads);
ind = 1;

volumeRange = log10([1e-4,1]);
for count = 1:numBkgFiles
    bkg = bkgAll{count};
    idxStart = randi(numel(bkg)-config.fs,numClipsPerFile(count),1);
    idxEnd = idxStart+config.fs-1;
    gain = 10.^((volumeRange(2)-volumeRange(1))*rand(numClipsPerFile(count),1) + volumeRange(1));
    for j = 1:numClipsPerFile(count)
        x = bkg(idxStart(j):idxEnd(j))*gain(j);
        x = max(min(x,1),-1);
        Xbkg(:,:,:,ind) = extractFeatures(x,config);
        ind = ind + 1;
    end
end
end

