%%Lab 3 Preparation
%Vavouliwtis Gewrgios 03112083
%Stavrakakis Dimitrios 03112017
clear all; close all;
%% Step 1
%get,process and save the given Data

data = cell(412,1);
mkdir('./ProcessedMusicFileSamples/');

for i = 1:412
    audiofile = sprintf('./MusicFileSamples/file%d.wav', i);
    [stereo,fs] = audioread(audiofile); % 2-column stereo file.
    mono = (stereo(:,1) + stereo(:,2))/2; % Convert stereo to mono.
    mono = resample(mono,1,2); % resample from 44100 to 22050 Hz. 
    savefile = sprintf('./ProcessedMusicFileSamples/file%d.wav', i);
    audiowrite(savefile,mono,22.05*1000,'BitsPerSample',8);
    data{i} =  mono;
end

save('data.mat', 'data');


%% Step 2
%Load the given Labelers
load('./EmotionLabellingData/Labeler1.mat');
Labeler1 = labelList;
load('./EmotionLabellingData/Labeler2.mat');
Labeler2 = labelList;
load('./EmotionLabellingData/Labeler3.mat');
Labeler3 = labelList;

Valence = cell(3,1);
Activation = cell(3,1);

Valence{1} = [Labeler1.valence];
Valence{2} = [Labeler2.valence];
Valence{3} = [Labeler3.valence];

Activation{1} = [Labeler1.activation];
Activation{2} = [Labeler2.activation];
Activation{3} = [Labeler3.activation];

mean_V = cellfun(@mean,Valence);
mean_A = cellfun(@mean,Activation);
s_V = cellfun(@std,Valence);
s_A = cellfun(@std,Activation);

Co_Occur = cell(4,1);

Co_Occur{4} = zeros(5,5);
freq = zeros(5,5);

for i = 1:3
    for j=1:412  
        freq(Valence{i}(j),Activation{i}(j)) = freq(Valence{i}(j),Activation{i}(j)) + 1;
    end
    Co_Occur{i} = freq;
    Co_Occur{4} = Co_Occur{4} + freq;
    freq = zeros(5,5);
    figure();
    image(Co_Occur{i});
    colorbar;
    colormap(copper(max(max(Co_Occur{i}))));
    title(['2D Histogram for Labeler',num2str(i)]);
    xlabel('Valence'); ylabel('Activation');
    print('-djpeg', sprintf('step2-%d',i));
end
i=4;
figure();
image(Co_Occur{i}/4);
colorbar;
colormap(copper(max(max(Co_Occur{i}))));
title('Total 2D Histogram for all Labelers');
xlabel('Valence'); ylabel('Activation');
print('-djpeg', sprintf('step2-%d',i));

%% Step3
%we will calculate the agreement for Valence,Activation and Total
%afterwards
%combinations of labelers: 1-2 , 1-3 , 2-3  
agreement_V = zeros(3,1);
agreement_A = zeros(3,1);
agreement_total = zeros(3,1);
%for labeler1 vs labeler2
agreement_V(1)=1-mean(abs((Valence{1}(:)-Valence{2}(:))/4));
agreement_A(1)=1-mean(abs((Activation{1}(:)-Activation{2}(:))/4));
agreement_total(1) = (agreement_V(1)+agreement_A(1))/2;
%for labeler1 vs labeler3
agreement_V(2)=1-mean(abs((Valence{1}(:)-Valence{3}(:))/4));
agreement_A(2)=1-mean(abs((Activation{1}(:)-Activation{3}(:))/4));
agreement_total(2) = (agreement_V(2)+agreement_A(2))/2;
%for labeler2 vs labeler3
agreement_V(3)=1-mean(abs((Valence{2}(:)-Valence{3}(:))/4));
agreement_A(3)=1-mean(abs((Activation{2}(:)-Activation{3}(:))/4));
agreement_total(3) = (agreement_V(3)+agreement_A(3))/2;
%save my results to txt
fileID = fopen('Step3Res.txt','w');
fprintf(fileID,'Presenation of Agreements\n\n');
combos = combntns([1,2,3],2);
for i=1:3
    fprintf(fileID,'Labeler%d VS Labeler%d\n',combos(i,1),combos(i,2));
    fprintf(fileID,'Valence Observed Agreement:%f10\n',agreement_V(i));
    fprintf(fileID,'Activation Observed Agreement:%f10\n',agreement_A(i));
    fprintf(fileID,'Total Mean Agreement:%f10\n\n',agreement_total(i));
    figure();
    subplot(1,2,1); hist(abs(Valence{combos(i,1)}(:)-Valence{combos(i,2)}(:))); xlabel('Valence'); ylabel('Samples');
    title(['labeler',num2str(combos(i,1)),' vs ','labeler',num2str(combos(i,2))]) 
    subplot(1,2,2); hist(abs(Activation{combos(i,1)}(:)-Activation{combos(i,2)}(:))); xlabel('Activation'); ylabel('Samples');
    title(['labeler',num2str(combos(i,1)),' vs ','labeler',num2str(combos(i,2))]) 
    print('-djpeg', sprintf('step3-%d',i));
end
fclose(fileID); 


%% Step 4
% code: http://www.mathworks.com/matlabcentral/fileexchange/36016-krippendorff-s-alpha/content/kriAlpha.m
fileID = fopen('Step4Res.txt','w');
fprintf(fileID,'Presenation of Krippendorff’s alpha coefficient\n\n');
alphaValence = kriAlpha(cell2mat(Valence),'ordinal');
alphaActivation = kriAlpha(cell2mat(Activation),'ordinal');
fprintf(fileID,'Krippendorff’s alpha coefficient for Valence: %f4\n',alphaValence);
fprintf(fileID,'Krippendorff’s alpha coefficient for Activation: %f4\n',alphaActivation);
fclose(fileID); 

%% Step 5
%calculatio of the mean of Valence and Activation of the 3 Labelers for
%each dimension separately
mean_Val = mean(cell2mat(Valence),1);
mean_Act = mean(cell2mat(Activation),1);
quant_points = linspace(1,5,13);

% quantize to the nearest value
%we use round function as our interpolation method for quantization has
%some declines from the wanting values. Those declines, though, can be
%omitted (10^-16)
mean_Val_quant = interp1q(quant_points',quant_points',mean_Val');
mean_Val_quant = round(mean_Val_quant',4);
mean_Act_quant = interp1q(quant_points',quant_points',mean_Act');
mean_Act_quant = round(mean_Act_quant',4);
quant_points = round(quant_points,4);


%calculation of co-occurence
Co_Occur_5 = zeros(length(quant_points),length(quant_points));
freq = zeros(length(quant_points),length(quant_points));
for i = 1:412
    row=find(quant_points==mean_Val_quant(i));
    col=find(quant_points==mean_Act_quant(i));
    freq(row,col) = freq(row,col)+1;
end
Co_Occur_5 = freq;

figure();
image(Co_Occur_5);
colormap(copper(max(max(Co_Occur_5))));
title('2D Histogram for final Co-Occurance Matrix');
xlabel('Valence'); ylabel('Activation');
set(gca, 'XTick', 1:length(quant_points)); % Change x-axis ticks
set(gca, 'XTickLabel', quant_points); % Change x-axis ticks labels.
set(gca, 'YTick', 1:length(quant_points)); % Change y-axis ticks
set(gca, 'YTickLabel', quant_points); % Change y-axis ticks labels.
print '-djpeg' 'Step5final.jpg';

close all;


%% Step 6: MIR Toolbox
addpath('C:\Users\dimst\Desktop\Pattern Recognition\Prop3\MIRtoolbox1.6.2\AuditoryToolbox\')
addpath('C:\Users\dimst\Desktop\Pattern Recognition\Prop3\MIRtoolbox1.6.2\')
MIRFeatures = cell(412,1);
load('data.mat');
T = 0.05;
Toverlap = 0.025;

for i = 1:412
    %loads the sound file 'filename' (in WAV or AU format) into a miraudio object.
    audio = miraudio(data{i},22050);
    %characteristics from MIR Toolbox:

    % Auditory roughness (with default parameters the desired oens) 
    AuditoryRoughness = mirroughness(audio);
    MIRFeatures{i}(1) = mirgetdata(mirmean(AuditoryRoughness));
    MIRFeatures{i}(2) = mirgetdata(mirstd(AuditoryRoughness));
    MedianValue   = mirgetdata(mirmedian(AuditoryRoughness));
    AuditoryRoughnessArray = mirgetdata(AuditoryRoughness); % convert from mirscalar object to double array
    MIRFeatures{i}(3) = mean(AuditoryRoughnessArray(AuditoryRoughnessArray < MedianValue));
    MIRFeatures{i}(4) = mean(AuditoryRoughnessArray(AuditoryRoughnessArray > MedianValue));

    % Fluctuation: Rythmic Periodicity Along Auditory Channels
    Fluctuation = mirfluctuation(audio,'Summary','Frame',T,'s',Toverlap,'s');
    FluctuationArray = mirgetdata(Fluctuation);
    MIRFeatures{i}(5) = max(FluctuationArray);
    MIRFeatures{i}(6) = mean(FluctuationArray);

    % Key Clarity: key estimation
    [~, keyclarity] = mirkey(audio,'Frame',T,'s',Toverlap,'s');
    MIRFeatures{i}(7) = mean(mirgetdata(keyclarity));

    % Modality: Major (1.0) vs Minor (-1.0)
    modality = mirmode(audio, 'Frame',T,'s',Toverlap,'s'); 
    MIRFeatures{i}(8) = mean(mirgetdata(modality));

    % Spectral Novelty
    [novelty, ~] = mirnovelty(audio,'Frame',T,'s',Toverlap,'s'); 
    MIRFeatures{i}(9) = mean(mirgetdata(novelty));

    % Harmonic Change Detection Function (HCDF)
    hcdf = mirhcdf(audio,'Frame',T,'s',Toverlap,'s'); 
    MIRFeatures{i}(10) = mean(mirgetdata(hcdf));

    %Step 7: Extract MFCCs
    
    % MIRMFCC
    MIRMFCCs = mirgetdata(mirmfcc(audio,'Frame',0.025,'s',0.010,'s','BANDS',26));%Q=26, frame=25ms,overlap=10ms
    %take the derivatives for delta and delta-delta MFCCs using deltas
    %function which found online and calculates the derivative for a
    %sequence
    MIRMFCCs_d = deltas(MIRMFCCs);
    MIRMFCCs_d2 = deltas(MIRMFCCs_d);
    %alternative way according to doc
    %MIRMFCCs_d = mirgetdata(mirmfcc(sample,'Frame',0.025,'s',0.010,'s','BANDS',26,'Delta',1));%Q=26, frame=25ms,overlap=10ms
    %MIRMFCCs_d2 = mirgetdata(mirmfcc(sample,'Frame',0.025,'s',0.010,'s','BANDS',26,'Delta',2));%Q=26, frame=25ms,overlap=10ms
    
    meanm_fcc = mean(MIRMFCCs,2); %mean value
    s_mfcc = std(MIRMFCCs,0,2); %standard deviation
    temp = sort(MIRMFCCs,2,'descend'); %sort our mffcs in descending order
    index = round(size(MIRMFCCs,2)*0.1); %find the 10% of our sample
    mean_high_MFCCs = mean(temp(:,1:index),2); %take the eman of the 10% with largest values
    temp = sort(MIRMFCCs,2); %sort our mffcs again in ascending order this time
    mean_low_MFCCs = mean(temp(:,1:index),2); %take the mean of the 10% with lowest values

    %do the same as above with delta-MFCCs
    mean_mfcc_d = mean(MIRMFCCs_d,2);
    s_mfcc_d = std(MIRMFCCs_d,0,2);
    temp = sort(MIRMFCCs_d,2,'descend');
    index = round(size(MIRMFCCs_d,2)*0.1);
    mean_high_MFCCs_d = mean(temp(:,1:index),2);
    temp = sort(MIRMFCCs_d,2);
    mean_low_MFCCs_d = mean(temp(:,1:index),2);

    %do the same as above with delta-delta-MFCCs
    mean_mfcc_d2 = mean(MIRMFCCs_d2,2);
    s_mfcc_d2 = std(MIRMFCCs_d2,0,2);
    temp = sort(MIRMFCCs_d2,2,'descend');
    index = round(size(MIRMFCCs_d2,2)*0.1);
    mean_high_MFCCs_d2 = mean(temp(:,1:index),2);
    temp = sort(MIRMFCCs_d2,2);
    mean_low_MFCCs_d2 = mean(temp(:,1:index),2);
    
    %concat into one array
    MIRFeatures{i} = [MIRFeatures{i},meanm_fcc',mean_mfcc_d',mean_mfcc_d2',...
                                     s_mfcc',s_mfcc_d',s_mfcc_d2',...
                                     mean_high_MFCCs',mean_high_MFCCs_d',mean_high_MFCCs_d2',...
                                     mean_low_MFCCs',mean_low_MFCCs_d',mean_low_MFCCs_d2'];
end

%now we got all the features we need