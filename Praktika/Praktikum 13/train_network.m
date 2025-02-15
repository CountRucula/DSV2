%% Füge die Helper-Funktionen zum Pfad hinzu
addpath('./helpers')

%% Definiere einige relevante Parameter für die Feature-Extraktion
config = struct();
config.fs = 16000;
config.commands = categorical(["yes","no","up","down","left","right","on","off","stop","go"]);
config.reduceDataset = false;

config.segmentDuration = 1;
config.frameDuration = 0.025;
config.hopDuration = 0.01;
config.numBands = 16;
config.numHops = floor((config.segmentDuration-config.frameDuration)/config.hopDuration) + 1;
config.FFTLength = 512;

%% Create Datasets
dataFolder = './daten/google_speech';
% Lese Trainings- und Validationsdaten
[X_train,Y_train,ads_train] = preprocess_data(fullfile(dataFolder, 'train'),config);
[X_val,Y_val,ads_val] = preprocess_data(fullfile(dataFolder, 'validation'),config);

% Lese Background-Daten
[X_bkg, ads_bkg] = get_background(fullfile(dataFolder, 'background'),config);

% Füge den Trainings- und Validationsdaten einen Teil der Background Daten
% an
numTrainBkg = floor(0.85*length(X_bkg));
X_train(:,:,:,end+1:end+numTrainBkg) = X_bkg(:,:,:,1:numTrainBkg);
Y_train(end+1:end+numTrainBkg) = "background";

numValidationBkg = floor(0.15*length(X_bkg));
X_val(:,:,:,end+1:end+numValidationBkg) = X_bkg(:,:,:,numTrainBkg+1:end);
Y_val(end+1:end+numValidationBkg) = "background";


%% Visualisiere einige Samples aus den Trainingsdaten
specMin = min(X_train,[],'all');
specMax = max(X_train,[],'all');
idx = randperm(numel(ads_train.Files),3);
figure('Units','normalized','Position',[0.2 0.2 0.6 0.6]);
for i = 1:3
    [x,fs] = audioread(ads_train.Files{idx(i)});
    subplot(2,3,i)
    plot(x)
    axis tight
    title(string(ads_train.Labels(idx(i))))
    
    subplot(2,3,i+3)
    training_data_idx = idx(i);
    % ToDo: Hier bitte Trainingssample mit Index trainings_data_idx
    % auswählen. Hierzu die Shape von X_train mit size(X_train) anschauen.
%     spect = ...;
    spect = X_train(:,:,:,training_data_idx);
    pcolor(spect')
    caxis([specMin specMax])
    shading flat
    
    sound(x,fs)
    pause(2)
end
%% Plotte die Histogramme der Trainings- und Validationslabels
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5])
subplot(2,1,1)
histogram(Y_train)
title("Training Label Distribution")

subplot(2,1,2)
histogram(Y_val)
title("Validation Label Distribution")

%% Definiere das Deep Neural Network
classWeights = 1./countcats(Y_train);
classWeights = classWeights'/mean(classWeights);
numClasses = numel(categories(Y_train));
timePoolSize = ceil(config.numHops/8);
numF = 12;
dropoutProb = 0.2;

layers = [
    imageInputLayer([config.numHops config.numBands])
    
    convolution2dLayer(3,numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,2*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(3,'Stride',2,'Padding','same')
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    convolution2dLayer(3,4*numF,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer([timePoolSize,1])
    
    dropoutLayer(dropoutProb)
    fullyConnectedLayer(numClasses)
    softmaxLayer
    weightedClassificationLayer(classWeights)];

analyzeNetwork(layers)
%% Trainiere das DNN, kann einige Minuten dauern
miniBatchSize = 128;
validationFrequency = floor(numel(Y_train)/miniBatchSize); 
options = trainingOptions('adam', ...
    'InitialLearnRate',5e-4, ...
    'MaxEpochs',10, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{X_val,Y_val}, ...
    'ValidationFrequency',validationFrequency, ...
    'ExecutionEnvironment','auto'); % or cpu, gpu, parallel

trainedNet = trainNetwork(X_train,Y_train,layers,options);
save('trainedNet.mat','trainedNet','config')

%% Berechne die Genauigkeit auf Trainings- und Validationsset und plotte Konfusionsmatrix
Y_val_pred = classify(trainedNet,X_val);
Y_train_pred = classify(trainedNet,X_train);

% ToDo: Hier bitte Genauigkeit auf Trainings- und Validationset berechnen
validationAcc = sum(Y_val == Y_val_pred)/numel(Y_val);
trainAcc = sum(Y_train == Y_train_pred)/numel(Y_train);
disp("Training Accuracy: " + trainAcc*100 + "%")
disp("Validation Accuracy: " + validationAcc*100 + "%")

fig = figure;
cm = confusionchart(Y_val,Y_val_pred,'RowSummary','row-normalized');
fig_Position = fig.Position;
fig_Position(3) = fig_Position(3)*1.5;
fig.Position = fig_Position; 
title('Konfusionsmatrix')