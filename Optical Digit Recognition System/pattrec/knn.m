function successRate = knn( traindata, testdata, indexes, k , heading)
        [closestKNeighbour, appearTimes] = mode(traindata(indexes(:, 1:k)),2);
        successRate = sum(closestKNeighbour==testdata) / size(testdata,1);
        success_RateKNNSimpleDigit = zeros(1,10);
        for digit = 0:9
            clear counterRightKNNSimpleDigit;
            counterRightKNNSimpleDigit = testdata(testdata == digit) == closestKNeighbour(testdata == digit);
            success_RateKNNSimpleDigit(digit + 1) = sum(counterRightKNNSimpleDigit) / size(counterRightKNNSimpleDigit, 1);
        end;
        if (k == 3)
            save('knnSimple.mat', 'success_RateKNNSimpleDigit');
            save('classify_results.mat', 'closestKNeighbour', '-append');
            save('conf_scores.mat', 'appearTimes', '-append');
            disp('************************************************* NNK ***********************************************');
            fprintf('Total success rate: %f\n', successRate);
            disp('Success Rate per digit:');
            disp(cell2table(num2cell(success_RateKNNSimpleDigit), 'VariableNames', heading));
        end;
end

