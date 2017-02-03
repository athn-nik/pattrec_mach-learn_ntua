function [classifications, successRate] = bayes( traindata, testdata )
    finalcolumn = size(traindata,2);
    cnt = zeros(1,2);
    plusone = [];
    minusone = [];
    for i = 1:size(traindata,1)
        if traindata(i,finalcolumn) == 1
            plusone = [plusone; traindata(i, 1:finalcolumn-1)];
            cnt(2) = cnt(2) + 1;
        else
            minusone = [minusone; traindata(i, 1:finalcolumn-1)];
            cnt(1) = cnt(1) + 1;
        end
    end
    m_plus = repmat(sum(plusone)/cnt(2),cnt(2), 1);
    m_minus = repmat(sum(minusone)/cnt(1),cnt(1),1);
    sd_plus = sqrt(sum((plusone - m_plus).^2)/cnt(2));
    sd_minus = sqrt(sum((minusone - m_minus).^2)/cnt(1));
    m(1,:) = m_minus(1,:);
    m(2,:) = m_plus(1,:);
    sd(1,:) = sd_minus + 0.001;
    sd(2,:) = sd_plus + 0.001;
    
    prior = zeros(1,2);
    for i = 1:2
        prior(i) = cnt(i) / size(traindata,1);
    end
    
    classifications = ones(size(testdata,1),1);
    posterior = zeros(size(testdata,1),2);
    for i = 1:size(testdata,1)
        for j = 1:2
            gaussval = (1/sqrt(2*pi))*(sd(j, :).^(-1)).*(exp(-0.5*((testdata(i,1:finalcolumn-1)-m(j, :)).^2)./(sd(j, :))));
            posterior(i, j) = sum(log(gaussval)) + log(prior(j));
        end
    end
    classifications(posterior(:,1) > posterior(:,2)) = -1;
    successRate = sum(classifications == testdata(:, finalcolumn))/size(testdata,1);
end