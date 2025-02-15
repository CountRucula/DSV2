%% Einlesen eines Audio Beispiels
[y,fs_file] = audioread("Counting-16-44p1-mono-15secs.wav");
fs = 16000;
y = resample(y,fs,fs_file);
y = y/max(abs(y));
t = (0:length(y)-1)/fs;

%% Definition von relevanten Parametern
win = hamming(1024,"periodic");
noverlap = 1024-128;
fftLength = 1024;
numBands = 16;

%% Berechnung der stft und des Spektrogramms

% ToDo: Hier bitte STFT des Signals y mit obigen Parametern berechnen.
% Wählen Sie als FrequencyRange 'onesided'.
[S,F,t_spec] = stft(y, fs, Window=win,OverlapLength=noverlap, FFTLength=fftLength, FrequencyRange="onesided");

% Berechnung des Spektrogramms aus der STFT
powerSpec = abs(S).^2;

%% Erstellung der Mel-Filterbank und Berechnung des Mel-Spektrogramms

% Hier bitte Mel-Filterbank erstellen
[filterBank,cf] = designAuditoryFilterBank(fs, FrequencyScale='mel',FFTLength=fftLength, NumBands=numBands, Normalization='none');
                               
%% Plot Filterbank
figure;                             
plot(F,filterBank.')
grid on
title("Mel Filter Bank")
xlabel("Frequenz [Hz]")

%% Berechnung Mel-Spektrogramm

% ToDo: Mel-Spektrogramm aus filterBank und powerSpec berechnen
melSpec = filterBank*powerSpec;

%% Plots der Spektrogramme
figure; 
subplot(3,1,1);
plot(t,y);
xlabel('Zeit [s]'); title('Zeitsignal')
subplot(3,1,2);
surf(t_spec,F,10*log10(powerSpec+1e-9),"EdgeColor","none"); view([0,90]); axis tight
xlabel("Zeit [s]"); ylabel("Frequenz [Hz]"); title('Spektrogramm')
subplot(3,1,3);
surf(t_spec,cf,10*log10(melSpec+1e-9),"EdgeColor","none"); view([0,90]); axis tight
xlabel("Zeit [s]"); ylabel("Frequenz [Hz]"); title('Mel-Spektrogramm')

