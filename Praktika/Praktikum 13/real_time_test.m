%% Füge die Helper-Funktionen zum Pfad hinzu
addpath('./helpers')

%% Lade das trainierte DNN und die Konfiguration
load('trainedNet.mat')
fs = config.fs;
labels = trainedNet.Layers(end).Classes;

%% Initalisiere den Audioreader und die Buffer
classificationRate = 20;
countThreshold = ceil(classificationRate*0.2);
probThreshold = 0.7;

adr = audioDeviceReader('SampleRate',fs,'SamplesPerFrame',floor(fs/classificationRate));

audioBuffer = dsp.AsyncBuffer(fs);
YBuffer(1:classificationRate/2) = categorical("background");
probBuffer = zeros([numel(labels),classificationRate/2]);

%% Initalisiere die Figure und Klassifiziere Audioinput solange wie Figure existiert

h = figure('Units','normalized','Position',[0.2 0.1 0.6 0.8]);
while ishandle(h)
    
    % Extrahiere Audiodaten vom Microphone und füge sie zum Audiobuffer hinzu
    x = adr();
    write(audioBuffer,x);
    
    % Lese 1 s vom Audiobuffer
    y = read(audioBuffer,fs,fs-adr.SamplesPerFrame);   
    
    % Berechne Features und klassifizier diese
    %ToDo: Extrahieren Sie die Features aus dem Sample y und
    %klassifizieren dieses mit dem trainierten DNN
    spec = extractFeatures(y, config);
    [YPredicted,probs] = classify(trainedNet,spec);
    
    % Speichere Prediktion und die Wahrscheinlichkeiten in die
    % entsprechenden Buffer
    YBuffer = [YBuffer(2:end),YPredicted];
    probBuffer = [probBuffer(:,2:end),probs(:)];
    
    % Plot the current waveform and spectrogram.
    subplot(2,1,1)
    plot(y);
    axis tight
    ylim([-1,1])
    
    subplot(2,1,2)
    pcolor(spec');
    caxis([-4 2.6445])
    shading flat
    
    % Bestimme die häufigste Klasse im Buffer
    % ToDo: Finde die häufigste Klasse im Buffer mit dem mode-Befehl
    [YMode,count] = mode(YBuffer);
    
    % Bestimme die maximale Wahrscheinlichkeit dieser Klasse im Buffer
    maxProb = max(probBuffer(labels == YMode,:));
    
    % Schreibe den Name dieser Klasse in den Plot, wenn die Anzahl dieser
    % Klasse im Buffer und die maximale Wahrscheinlichkeit über den
    % Thresholds ist. Schreibe ansonsten nichts in den Titel.
    subplot(2,1,1)
    if YMode == "background" || count < countThreshold || maxProb < probThreshold
        title(" ")
    else
        title(string(YMode),'FontSize',20)
    end
    
    % Aktualisiere Plot
    drawnow
end
