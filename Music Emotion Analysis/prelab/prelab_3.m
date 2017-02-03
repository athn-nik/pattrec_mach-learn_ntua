%% lab 3
%% Preprocess

close all;
clear all;
clc;


%% Preprocessing
%%Step 1

%if you want to run this code
%uncomment it I read -modify-store results for quicker processing


% for i = 1:412
%     
%     [music_data,fs] = audioread(['./MusicFileSamples/file',num2str(i),'.wav']);
%     %music_data has 2 channels it means stero 
%     %so we have to turn it into mono
%     mono_data = (music_data(:,1) + music_data(:,2))/2;
%     % cut current frequency in half because we want 22050 Hz
%     sample_rate=fs/22050;
%     mono_data = resample(mono_data,1,sample_rate);
%     MusicSamples{i} = int8(mono_data);
%     audiowrite(['./mysamples/file',num2str(i),'.wav'],mono_data,22050,...
%         'BitsPerSample',8)
%         
% end
% MusicSamples=MusicSamples';
% save('MusicSamples.mat', 'MusicSamples');

load('MusicSamples.mat');
%% Step 2

% Valence vs Activation vs Number of Occurences
load('./EmotionLabellingData/Labeler1.mat');  
L1=labelList; 
load('./EmotionLabellingData/Labeler2.mat');
L2=labelList; 
load('./EmotionLabellingData/Labeler3.mat');
L3=labelList; 
 



for i=1:412
    valence(1,i) = L1(i).valence;
    activation(1,i) = L1(i).activation;
    valence(2,i) = L2(i).valence;
    activation(2,i) = L2(i).activation;
    valence(3,i) = L3(i).valence;
    activation(3,i) = L3(i).activation;
end

for i = 1:3 % for each laberer
    %a mean value
    meanVal(i) = mean(valence(i,:));
    meanAct(i) = mean(activation(i,:));
    %b standard deviation
    stdVal(i) = std(valence(i,:));
    stdAct(i) = std(activation(i,:));
end
for i=1:3
   
    for j = 1:5
        for k = 1:5
            Occurence_Matrix{i}(j,k) = size(find(valence(i,:) == j & activation(1,:) == k),2);
        end
    end
   figure(i)
    image(Occurence_Matrix{i});
    colormap(copper(max(max(Occurence_Matrix{i}))));
     
    title(['2D Histogram for labeler',num2str(i)]);
    xlabel('Valence'); ylabel('Activation');
    figure(i+50);
    bar3(Occurence_Matrix{i});
    title('2D Histogram for final labels');
    xlabel('Valence'); ylabel('Activation');
    %fix values 1,...5 for each axis
    set(gca, 'XTick', 1:5); 
    set(gca, 'XTickLabel', 1:5); 
    set(gca, 'YTick', 1:5); 
    set(gca, 'YTickLabel', 1:5); 
   
    title(['3D Histogram to compare values for labeler',num2str(i)]);
    xlabel('Valence'); ylabel('Activation');
end
%% Step 3
number = 0;
n1=0;
figure(4);
for i = 1:3
    %for all labelers
    for k=(i+1):3
        %for all pairs
        differ_val = abs(valence(i,:) - valence(k,:));
        differ_act = abs(activation(i,:) - activation(k,:));
        n1=n1+1;
        Obs_val_agr(n1) = 1 - mean(differ_val/4);
        Obs_act_agr(n1) = 1 - mean(differ_act/4);
		Obs_tot_agr(n1) = (Obs_val_agr(n1) +Obs_act_agr(n1))/2;
   
        number = number + 1;
        subplot(3,2,number); hist(differ_val); xlabel('Valence'); ylabel('Number of samples');
        title(['labeler',num2str(i),' vs ','labeler',num2str(k)])     
        number = number + 1;
        subplot(3,2,number); hist(differ_act); xlabel('Activation'); ylabel('Number of samples');
        title(['labeler',num2str(i),' vs ','labeler',num2str(k)])   

    end
end

%agreement plot
x=1:3;

figure(5)
bar(x,Obs_val_agr,.5)
for i=1:numel(Obs_val_agr)
    text(x(i),Obs_val_agr(i),num2str(Obs_val_agr(i),'%0.5f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end
title('Observed valence agreement')
str={'L1 vs L2','L1 vs L3','L2 vs L3'};
set(gca,'XTickLabel',str,'XTick',1:numel(str))

figure(6)
bar(x,Obs_act_agr,.5)
for i=1:numel(Obs_act_agr)
    text(x(i),Obs_act_agr(i),num2str(Obs_act_agr(i),'%0.5f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end
title('Observed activation agreement')
set(gca,'XTickLabel',str,'XTick',1:numel(str))

figure(7)
bar(x,Obs_tot_agr,.5)
for i=1:numel(Obs_tot_agr)
    text(x(i),Obs_tot_agr(i),num2str(Obs_tot_agr(i),'%0.5f'),...
               'HorizontalAlignment','center',...
               'VerticalAlignment','bottom')
end
title('Observed total agreement')
set(gca,'XTickLabel',str,'XTick',1:numel(str))

%% Step 4
% www.mathworks.com/matlabcentral/fileexchange/36016-krippendorff-s-alpha/content/kriAlpha.m
alphaVal = kriAlpha([valence(1,:);valence(2,:);valence(3,:)],'ordinal')
alphaAct = kriAlpha([activation(1,:);activation(2,:);activation(3,:)],'ordinal')
%% Step 5
%mean of each dimension each labeler
finalVal = mean([valence(1,:);valence(2,:);valence(3,:)]);
finalAct = mean([activation(1,:);activation(2,:);activation(3,:)]);
quant_val = [1 1.3333 1.6667 2 2.3333 2.6667 3 3.3333 3.6667 4 4.3333 4.6667 5];

%can use interpl

% Round to nearest value  
for i=1:412
     % find in which value valence is nearest
     [~,min_idx] = min(abs(finalVal(i)-quant_val));
     %take its idx and quantize the value to the correct quantized value
     finalVal(i) = quant_val(min_idx);
     %same for activation
     [~,min_idx] = min(abs(finalAct(i)-quant_val));
     finalAct(i) = quant_val(min_idx);
end

for j = 1:length(quant_val)
    for k = 1:length(quant_val)
        %compute coccurrence same as before
        finalOccurences(j,k) = size(find(finalVal == quant_val(j) & finalAct == quant_val(k)),2); 
    end
end
save('finalVal.mat','finalVal');
save('finalAct.mat','finalAct');
% fix axis again for 1,1.333,...,5
figure(8);
image(finalOccurences);
colormap(copper(max(max(finalOccurences))));
title('2D Histogram for final labels');
xlabel('Valence'); ylabel('Activation');
set(gca, 'XTick', 1:length(quant_val)); 
set(gca, 'XTickLabel', quant_val);
set(gca, 'YTick', 1:length(quant_val)); 
set(gca, 'YTickLabel', quant_val); 

figure(9);
bar3(finalOccurences);
title('2D Histogram for final labels');
xlabel('Valence'); ylabel('Activation');
set(gca, 'XTick', 1:length(quant_val)); 
set(gca, 'XTickLabel', quant_val);
set(gca, 'YTick', 1:length(quant_val)); 
set(gca, 'YTickLabel', quant_val);
%% Step 6

MIRFeatures = cell(412,1);

T = 0.05;
Toverlap = 0.025;
addpath(genpath('./MIRtoolbox1.6.2'))
for i = 1:1
sample = miraudio(MusicSamples{i},22050);
%% 1 Auditory roughness
AuditoryRoughness = mirroughness(sample,'Frame'); % put 'FRAME' option?
%calculate all
MIRFeatures{i}(1) = mirgetdata(mirmean(AuditoryRoughness));
MIRFeatures{i}(2) = mirgetdata(mirstd(AuditoryRoughness));
RoughnessMedian   = mirgetdata(mirmedian(AuditoryRoughness));
AuditoryRoughness = mirgetdata(AuditoryRoughness); % convert from mirscalar to double
MIRFeatures{i}(3) = mean(AuditoryRoughness(AuditoryRoughness < RoughnessMedian));
MIRFeatures{i}(4) = mean(AuditoryRoughness(AuditoryRoughness > RoughnessMedian));

%% 2 Fluctuation: Rythmic Periodicity Along Auditory Channels
Fluctuation = mirgetdata(mirfluctuation(sample,'Summary'));
MIRFeatures{i}(5) = max(Fluctuation);
MIRFeatures{i}(6) = mean(Fluctuation);

%% 3 Key Clarity: key estimation
[whichkey, keyclarity] = mirkey(sample,'Frame',T,'s'); % Todo change frame params
MIRFeatures{i}(7) = mean(mirgetdata(keyclarity));

%% 4 Modality: Major (1.0) vs Minor (-1.0)
modality = mirgetdata(mirmode(sample, 'Frame',T,'s')); % change frame params
MIRFeatures{i}(8) = mean(modality);

%% 5 Spectral Novelty
[novelty, similaritymatrix] = mirgetdata(mirnovelty(sample)); 
MIRFeatures{i}(9) = mean(novelty);

%% 6 Harmonic Change Detection Function (HCDF)
hcdf =mirgetdata( mirhcdf(sample,'Frame',T,'s',Toverlap,'s')); % change frame parameters
MIRFeatures{i}(10) = mean(hcdf);
%% Step 7
%we have different overlap and frame duration
% MIRMFCC fix prameteers for frame same for deltas deltasdeltas
MIRMFCCs = mirgetdata(mirmfcc(sample,'Frame',0.025,'s',0.01,'s','Bands',26,'Rank',1:13));

meanmfcc = mean(MIRMFCCs,2);
stdmfcc = std(MIRMFCCs,0,2);
sorted = sort(MIRMFCCs,2,'descend');
% 10% of bigger
n_el = round(size(MIRMFCCs,2)*0.1);
mean_gr_MFCC = mean(sorted(:,1:n_el),2);
% 10% of smaller
sorted = sort(MIRMFCCs,2);
mean_low_MFCC = mean(sorted(:,1:n_el),2);
%deltas

MIRMFCCs_d = mirgetdata(mirmfcc(sample,'Frame',0.025,'s',0.01,'s','Bands',26,'Rank',1:13,'Delta',1));
%same calculations mean std 10% max 10% min
meanmfcc_d = mean(MIRMFCCs_d,2);
stdmfcc_d = std(MIRMFCCs_d,0,2);
sorted = sort(MIRMFCCs_d,2,'descend');
%can take 90% !! from previous sorting
n_el = round(size(MIRMFCCs_d,2)*0.1);
mean_gr_MFCC_d = mean(sorted(:,1:n_el),2);
sorted = sort(MIRMFCCs_d,2);
mean_low_MFCC_d = mean(sorted(:,1:n_el),2);
%deltas deltas same 
MIRMFCCs_d2 = mirgetdata(mirmfcc(sample,'Frame',0.025,'s',0.01,'s','Bands',26,'Rank',1:13,'Delta',2));
meanmfcc_d2 = mean(MIRMFCCs_d2,2);
stdmfcc_d2 = std(MIRMFCCs_d2,0,2);
sorted = sort(MIRMFCCs_d2,2,'descend');
n_el = round(size(MIRMFCCs_d2,2)*0.1);
mean_gr_MFCC_d2 = mean(sorted(:,1:n_el),2);
sorted = sort(MIRMFCCs_d2,2);
mean_low_MFCC_d2 = mean(sorted(:,1:n_el),2);
%fix 39*3 elems to fit in correct order in features matrix
MIRFeatures{i}(11:49)  = [meanmfcc' meanmfcc_d' meanmfcc_d2'];
MIRFeatures{i}(50:88) = [stdmfcc' stdmfcc_d' stdmfcc_d2'];
MIRFeatures{i}(89:127) = [mean_gr_MFCC' mean_gr_MFCC_d' mean_gr_MFCC_d2'];
MIRFeatures{i}(128:166) = [mean_low_MFCC' mean_low_MFCC_d' mean_low_MFCC_d2'];
end
save('MIRFeatures.mat','MIRFeatures');