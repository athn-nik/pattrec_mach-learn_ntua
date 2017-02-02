close all; 
clear all;
clc;
traindata = load('train.txt');

%% Step 1 
d = reshape(traindata(131,2:257),16,16)'; 
%poio_einai = traindata(131,1);
figure();
imagesc(d); colormap(gray);
title('131st digit');
print -djpeg 131digit.jpg

%Krathma olwn twn 16x16 instances kathe pshfiou se ena pinaka(ton d) o
%opoios einai 4D dld 16*16*10(0 ews 9)*diaforetikes emfaniseis
digit = 0;
cnt = zeros(1,10);
for i=1:size(traindata,1)  
    digit = traindata(i,1);
    % O pinakas cnt sthn thesh i exei poses fores emfanisthke to i-1 psifio
    cnt(digit+1) = cnt(digit+1) + 1; 
    d(:,:,digit+1,cnt(digit+1)) = reshape(traindata(i,2:257),16,16)';
    
end

digit = 0;

%% Step 2
mean_of_zero10 = sum(d(10,10,digit+1,:)) / cnt(digit+1)

%% Step 3
variance_of_zero10 = sum((d(10,10,digit+1,:)- mean_of_zero10).^2)/ cnt(digit+1)

%% Step 4 7a
for digit=0:9
    for i=1:cnt(digit+1)
        mean(:, :, digit+1 ,i) = sum(d(:,:,digit+1,:),4) / cnt(digit+1);
        %mesos oros dld tou psifiou 3rd dimension gia kathe emfanish 4th
        %dimension
    end
    variance(:,:,digit+1) = sum((d(:,:,digit+1,:) - mean(:,:,digit+1,:) ).^2, 4)/ cnt(digit+1);
    
 %% Step 5
 %% Step 7b 
    figure();
    imagesc(mean(:,:,digit+1,1)); colormap(gray);
    title(sprintf('Digit %d from Mean Value',digit));
    print(sprintf('%d_mean', digit),'-djpeg'); 
end
mean = mean(:,:,:,1);
    
%% Step 6
figure();
imagesc(variance(:,:,1)); colormap(gray);
title('Digit 0 from Variance');
print -djpeg 0_variance.jpg    
    
%% Step 8 
i = 101;
digit = 0;
testdata = load('test.txt');
 
test_digit = reshape(testdata(i,2:257),16,16)';
for digit=0:9
    distances(digit+1) = sum(sum( (test_digit - mean(:,:,digit+1)).^2 ));
end
 
whichDigit = find(distances == min(distances))-1
realDigit = testdata(i,1)


%% Step 9a
correctClassifications = 0;

timesOfDigit(1:10) = 0;
correctClassificationsOfDigit(1:10) = 0;
for i=1:size(testdata,1)
    test_digit = reshape(testdata(i,2:257),16,16)';
    for digit=0:9
        % Ypologismos apostashs psifiou pros elegxo apo kathe ena apo ta 0 - 9
        distances(digit+1) = pdist2(reshape(test_digit,1,256),reshape(mean(:,:,digit+1),1,256));
    end
    % briskei to psifio me tin elaxisti apostash apo auto pou eksetazoume
    whichDigit = find(distances == min(distances))-1;
    if (size(whichDigit,2) > 1) 
       whichDigit = whichDigit(1);
    end;
    distEuc(i, :) = distances;
    whichDigitEuc(i) = whichDigit;
    realDigit = testdata(i,1);
    % kai elegxei ean exei ginei swsta i provlepsi
    timesOfDigit(realDigit+1) = timesOfDigit(realDigit+1) + 1;
    %poses fores vrhkame to [pshfio gia analutiko pososto epituxias
    if (realDigit == whichDigitEuc(i))
        correctClassificationsOfDigit(realDigit+1) = correctClassificationsOfDigit(realDigit+1)+1;
        correctClassifications=correctClassifications+1; 
    end
end

%% Step 9b
save('classify_results.mat', 'whichDigitEuc');
save('conf_scores.mat', 'distEuc');
save('prep_variables.mat', 'mean');
save('prep_variables.mat', 'variance', '-append');
save('prep_variables.mat', 'timesOfDigit', '-append');
success_rate = correctClassifications/size(testdata,1)
success_rateOfDigit = correctClassificationsOfDigit ./ timesOfDigit