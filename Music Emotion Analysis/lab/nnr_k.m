function [classify, successRate] = nnr_k( traindata, testdata, k)
    
    for i = 1:size(testdata, 1)
        % take the correct number of test 
        test = testdata(i,1:size(traindata,2)-1);
        % replicate it train times for the distance
        test = repmat(test, size(traindata,1),1);
        % use the euclid distance 
        dist(i, :) =sqrt(sum((test - traindata(:, 1:(size(traindata,2)-1))).^2,2))';
    end
    %sort distances
    [~, idx] = sort(dist,2);
    %if nnr-1 take just the nearest 
    if k == 1
        classify = traindata(idx(:,1), size(traindata,2));
    else
        tmptraindata = traindata(:,size(traindata,2));
        %column vector contain most frequent value of each row of first k
        %indexes accorting to sort
        [classify, ~] = mode(tmptraindata(idx(:, 1:k)),2);        
    end
    %success=correct/all
    successRate = sum(classify == testdata(:, size(traindata,2)))/size(testdata,1);
end