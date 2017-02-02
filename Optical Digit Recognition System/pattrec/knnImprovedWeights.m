function successRate_kNNImproved = knnImprovedWeights( traindata, testdata, sortedDistances, indexes, k , heading, printvar)
    correctClassifications = 0;
    correctClassificationsOfDigitKNNWeights(1:10) = 0;
    timesOfDigit(1:10) = 0;
    weightOfDigit = zeros(1,10);
    whichDigitknnWeights = zeros(1,size(testdata, 1));
    weightsKnn = zeros(size(testdata, 1), 10);
    for i = 1:size(testdata, 1)
       clear digitPlace;
       digitMatrix = traindata(indexes(i,1:k))'; 
       for digit = 0:9
           digitPlace(digit+1,:) = (digitMatrix == digit);
           if (sum(digitPlace(digit+1,:)) > 0) 
                invdists = 1 ./ (sortedDistances(i,1:k).^2);
               weightOfDigit(digit+1) = sum(digitPlace(digit+1,:).*invdists);
           else
               weightOfDigit(digit+1) = -inf;
           end
       end
       whichDigitknnWeights(i) = find(weightOfDigit == max(weightOfDigit))-1;
       realDigit = testdata(i,1);
       weightsKnn(i, :) = weightOfDigit;
       timesOfDigit(realDigit + 1) = timesOfDigit(realDigit + 1) + 1;
       if (whichDigitknnWeights(i) == realDigit)
            correctClassifications = correctClassifications + 1;
            correctClassificationsOfDigitKNNWeights(realDigit + 1) = correctClassificationsOfDigitKNNWeights(realDigit + 1) + 1;
       end
    end
    successRate_kNNImproved = correctClassifications / size(testdata,1);
    if (k==3)
        success_rateOfDigitKNNWeights = correctClassificationsOfDigitKNNWeights ./ timesOfDigit;
        save('knnWeights.mat', 'success_rateOfDigitKNNWeights');
        save('classify_results.mat', 'whichDigitknnWeights', '-append');
        save('conf_scores.mat', 'weightsKnn', '-append');
        if (printvar == 1)
            disp('******************************************* NNK (Weights Optimization) *****************************************');
            fprintf('Total success rate: %f\n', successRate_kNNImproved);
            disp('Success Rate per digit:');
            disp(cell2table(num2cell(success_rateOfDigitKNNWeights), 'VariableNames', heading));
        end
    end
end