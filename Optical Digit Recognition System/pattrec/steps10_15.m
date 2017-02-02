load('prep_variables.mat');
traindata=load('train.txt');
testdata=load('test.txt');
heading = {'Digit0' 'Digit1' 'Digit2' 'Digit3' 'Digit4' 'Digit5' 'Digit6' 'Digit7' 'Digit8' 'Digit9'};


%% Step 10
% upologise tis apriori gia na voithiseis ton bayesian taksinomhth
    prior = zeros(1,10);
    for i = 0:9
        prior(i+1) = size(find(traindata(:,1)==i),1) / size(traindata,1);
    end
    
    %% Step 11
    correctClassificationsOfDigitBayes(1:10) = 0;
    correctClassificationsOfDigitBayes2(1:10) = 0;
    correctClassifications = 0;
    correctClassifications2 = 0;
    m = zeros(256,10);
    v = zeros(256,10);
    % upologismos meshs timhs kai diasporas ksana gia na tis
    % xrhsimopoihsoume ston bayesian taksinomhth (Gaussian Naive Bayesian)
    for digit = 0:9
        m(:,digit+1) = reshape(mean(:,:,digit+1)', 256,1);
        v(:,digit+1) = reshape(variance(:,:,digit+1)', 256,1)+0.001;
    end
    
    gauss_val = zeros(256,10);
    gauss_val2 = zeros(256,10);
    posterior = zeros(1,10);
    posterior2 = zeros(1,10);
    probBayes = zeros(size(testdata,1), 10);
    probBayes2 = zeros(size(testdata,1), 10);
    whichDigitBayes = zeros(1, size(testdata,1));
    whichDigitBayes2 = zeros(1, size(testdata,1));
    for i = 1:size(testdata,1)
        test_digit = testdata(i, 2:257)';
        for digit=0:9
        %    digit = 0;
            %xrhsimopoiw mesh timh kai diaspora upologismenh
            gauss_val(:,digit+1) = (1/sqrt(2*pi))*(v(:,digit+1).^(-1)).*(exp(-0.5*((test_digit-m(:,digit+1)).^2)./v(:,digit+1)));
            %xrhsimopoiw mesh timh kai diaspora 1
            gauss_val2(:,digit+1) = (1/sqrt(2*pi))*(exp(-0.5*((test_digit-m(:,digit+1)).^2)));
            %upologismos aposteriori kai stis duo periptwseis 
            posterior(digit+1) = sum(log(gauss_val(:,digit+1))) + log(prior(digit+1));
            posterior2(digit+1) = sum(log(gauss_val2(:,digit+1))) + log(prior(digit+1));
        end
        % taksinomhsh gia tis 2 periptwseis
        whichDigit = find(posterior == max(posterior))-1;
       
        whichDigit2 = find(posterior2 == max(posterior2))-1;
       
        % isopalies
        
        if (size(whichDigit,2) > 1) 
            whichDigit = whichDigit(1);
        end;
        
        probBayes(i, :) = posterior;
        whichDigitBayes(i) = whichDigit;
        if (size(whichDigit2,2) > 1) 
            whichDigit2 = whichDigit2(1);
        end;
        
        %upologismos epituxias kai gia tis 2 periptwseis
        
        probBayes2(i, :) = posterior2;
        whichDigitBayes2(i) = whichDigit2;
        realDigit = testdata(i,1);
    

        if (whichDigitBayes(i) == realDigit) 
            correctClassifications = correctClassifications + 1;
            correctClassificationsOfDigitBayes(realDigit+1) = correctClassificationsOfDigitBayes(realDigit+1) + 1;
        end
        
        if (whichDigitBayes2(i) == realDigit) 
            correctClassifications2 = correctClassifications2 + 1;
            correctClassificationsOfDigitBayes2(realDigit+1) = correctClassificationsOfDigitBayes2(realDigit+1) + 1;
        end
        
    end

    save('classify_results.mat', 'whichDigitBayes', '-append');
    save('classify_results.mat', 'whichDigitBayes2', '-append');
    save('conf_scores', 'probBayes', '-append');
    save('conf_scores', 'probBayes2', '-append');
    % gia mean var
    successRate = correctClassifications / size(testdata,1);
    % gia mean var=1
    successRate2 = correctClassifications2 / size(testdata,1);
    successRateOfDigitBayes = correctClassificationsOfDigitBayes ./ timesOfDigit;
    
    save('Bayes.mat', 'successRateOfDigitBayes');
    successRateOfDigitBayes2 = correctClassificationsOfDigitBayes2 ./ timesOfDigit;
    
    save('BayesVar1.mat', 'successRateOfDigitBayes2');
    
    
    fprintf('Total success rate var calculated: %f\n', successRate);
    
    disp('Success Rate per digit var calculated:');
    disp(cell2table(num2cell(successRateOfDigitBayes), 'VariableNames', heading));
    
    fprintf('Total success rate var 1: %f\n', successRate2);
    disp('Success Rate per digit var 1:');
    heading = {'Digit0' 'Digit1' 'Digit2' 'Digit3' 'Digit4' 'Digit5' 'Digit6' 'Digit7' 'Digit8' 'Digit9'};
    disp(cell2table(num2cell(successRateOfDigitBayes2), 'VariableNames', heading));

%% Step 13

    correctClassifications = 0;
    dist = zeros(1,1000);
    for i = 1:100
        %evresh apostashs gia katataksh
        for j = 1:1000
           dist(j) = sqrt(sum((testdata(i, 2:257)-traindata(j, 2:257)).^2)); 
        end
        % upologismos posostou
        whichDigit = traindata(dist == min(dist),1);
        realDigit = testdata(i,1);
        if (size(whichDigit,2) > 1) 
            whichDigit = whichDigit(1);
        end;
        if (whichDigit == realDigit) 
            correctClassifications = correctClassifications + 1;
        end
    end
    successRate100NN1 = correctClassifications / 100;
    fprintf('Total success rate NN1 100/1000: %f\n', successRate100NN1);

%% Step 14 a,b 
correctClassifications = 0;
correctClassificationsOfDigitNN1(1:10) = 0;
distAll = zeros(size(testdata,1),size(traindata,1));
whichDigitNN1 = zeros(1,size(testdata,1));
distNN1 = zeros(size(testdata,1), 10);
for i = 1:size(testdata, 1)
    testchars = testdata(i,2:257);
    testchars = repmat(testchars, size(traindata,1),1);
    distAll(i, :) =sqrt(sum((testchars - traindata(:, 2:257)).^2,2))';
    whichDigit = traindata(distAll(i,:) == min(distAll(i,:)),1);
    if (size(whichDigit,2) > 1) 
   %     moredist = 'There have been found more than one minimums!'
        whichDigit = whichDigit(1);
        index = index(1);
    end;
    whichDigitNN1(i) = whichDigit;
    realDigit = testdata(i,1);
    if (whichDigitNN1(i) == realDigit) 
        correctClassificationsOfDigitNN1(realDigit+1) = correctClassificationsOfDigitNN1(realDigit+1) + 1;
        correctClassifications = correctClassifications + 1;
    end
    for (digit = 0:9)
        tempdist = distAll(i, traindata(:,1)==digit);
        distNN1(i, digit+1) = min(tempdist);
    end
end

successRateAllNN1 = correctClassifications / size(testdata, 1);
successRateOfDigitNN1 = correctClassificationsOfDigitNN1 ./ timesOfDigit;
save('nn1.mat', 'successRateOfDigitNN1');
save('classify_results.mat', 'whichDigitNN1', '-append');
save('conf_scores.mat', 'distNN1', '-append');

    fprintf('Total success rate NN1 all: %f\n', successRateAllNN1);
    disp('Success Rate per digit NN1 all:');
    disp(cell2table(num2cell(successRateOfDigitNN1), 'VariableNames', heading));

%% Step 14c

clear closestKNeighbour;
[sortedDist, ind] = sort(distAll,2);
tmptraindata = traindata(:,1);

    kmax = 101;
    printvar = 1;
    successRate_kNN = zeros(1, floor(kmax/2));
    for k = 3:2:kmax        
        successRate_kNN(floor(k/2)) = knn(tmptraindata, testdata(:,1), ind, k, heading); 
    end
    figure();
    plot(3:2:kmax, successRate_kNN);
    title('Success Rate (y axis) k(x axis) Classic Knn');


%% Step 14d

successRate_kNNImprovedWeight = zeros(1, floor(kmax/2));
for k = 3:2:kmax
    successRate_kNNImprovedWeight(floor(k/2)) = knnImprovedWeights(tmptraindata, testdata, sortedDist, ind, k, heading, printvar);
end

    figure();
    plot(3:2:kmax, successRate_kNNImprovedWeight);
    title('Success Rate (y axis) k(x axis) Classic Knn (Weights Optimization)');

%% STEP 15

 [successRateLinearSVM successRatePolynomialSVM polSVMScore whichDigitPolSVM successRateOfDigitPolSVM linSVMScore whichDigitLinSVM successRateOfDigitLinSVM]=mySvm(traindata,testdata,timesOfDigit);

    save('svmLin.mat', 'successRateOfDigitLinSVM');
    save('classify_results.mat', 'whichDigitLinSVM', '-append');
    save('conf_scores.mat', 'linSVMScore', '-append');
    fprintf('Total success rate linear: %f\n', successRateLinearSVM);
    disp('Success Rate per digit:');
    disp(cell2table(num2cell(successRateOfDigitLinSVM), 'VariableNames', heading));


save('svmPol.mat', 'successRateOfDigitPolSVM');
save('classify_results.mat', 'whichDigitPolSVM', '-append');
save('conf_scores.mat', 'polSVMScore', '-append');

    fprintf('Total success rate polynomial: %f\n', successRatePolynomialSVM);
    disp('Success Rate per digit:');
    disp(cell2table(num2cell(successRateOfDigitPolSVM), 'VariableNames', heading));


%% Deuteros taksinomhths me decision trees
    features = traindata(:,2:257);
    Output = traindata(:,1);
    ctree = fitctree(features,Output);
    Result = predict(ctree, testdata(:,2:257));

    correctClassificationsDecTrees = 0;
    correctClassificationsDecTreesOfDigit(1:10) = 0;
    for i= 1:size(testdata,1)
        if (Result(i) == testdata(i,1))
            correctClassificationsDecTrees = correctClassificationsDecTrees+1;
            correctClassificationsDecTreesOfDigit(testdata(i,1) + 1) = correctClassificationsDecTreesOfDigit(testdata(i,1) + 1) + 1;
        end;
    end
    successRateDecTrees = correctClassificationsDecTrees/size(testdata,1);
    successRateOfDigitDecTrees = correctClassificationsDecTreesOfDigit ./ timesOfDigit;
    fprintf('Total success rate (decision trees): %f\n', successRateDecTrees);
    disp('Success Rate per digit:');
    disp(cell2table(num2cell(successRateOfDigitDecTrees), 'VariableNames', heading));
