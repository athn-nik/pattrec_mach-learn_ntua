function [successRateLinearSVM successRatePolynomialSVM polSVMScore whichDigitPolSVM successRateOfDigitPolSVM linSVMScore whichDigitLinSVM successRateOfDigitLinSVM]=mySVM(traindata,testdata,timesOfDigit)

warning('off', 'all');

for digit = 0:9
    myclass = traindata(:,1);
    myclass(myclass ~= digit) = -1;
    options = statset('maxiter', inf);
    SVMStruct(digit+1) = svmtrain(traindata(:, 2:257) ,myclass, 'kernel_function', 'linear', 'options', options, 'boxconstraint',0.5,'kktviolationlevel',0.03);
    SVMStructPol(digit+1) = svmtrain(traindata(:, 2:257) ,myclass, 'kernel_function', 'polynomial', 'options', options, 'boxconstraint',0.5,'kktviolationlevel',0.03);
end

for digit = 0:9
    GroupC(:,digit+1) = svm_classify(SVMStruct(digit+1),testdata(:,2:257))';
    GroupPolC(:,digit+1) = svm_classify(SVMStructPol(digit+1),testdata(:,2:257))';
end

correctClassificationsLinearSVM = 0;
correctClassificationsLinearSVMOfDigit(1:10) = 0;
correctClassificationsPolynomialSVM = 0;
correctClassificationsPolynomialSVMOfDigit(1:10) = 0;
whichDigitLinSVM = zeros(1,size(testdata,1));
whichDigitPolSVM = zeros(1,size(testdata,1));
for i= 1:size(testdata,1)
    whichDigitLinSVM(i) = (find(GroupC(i,:) == min(GroupC(i,:)))-1);
   if (whichDigitLinSVM(i) == testdata(i,1))
       correctClassificationsLinearSVM = correctClassificationsLinearSVM+1;
       correctClassificationsLinearSVMOfDigit(testdata(i,1) + 1) = correctClassificationsLinearSVMOfDigit(testdata(i,1) + 1) + 1;
    end;
    whichDigitPolSVM(i) = (find(GroupPolC(i, :) == min(GroupPolC(i,:)))-1);
    if (whichDigitPolSVM(i) == testdata(i,1))
       correctClassificationsPolynomialSVM = correctClassificationsPolynomialSVM+1;
       correctClassificationsPolynomialSVMOfDigit(testdata(i,1) + 1) = correctClassificationsPolynomialSVMOfDigit(testdata(i,1) + 1) + 1;
    end;
end

    successRateLinearSVM = correctClassificationsLinearSVM/size(testdata,1);
    successRateOfDigitLinSVM = correctClassificationsLinearSVMOfDigit ./ timesOfDigit;
    linSVMScore = GroupC;
    
    successRatePolynomialSVM = correctClassificationsPolynomialSVM/size(testdata,1);
successRateOfDigitPolSVM = correctClassificationsPolynomialSVMOfDigit ./ timesOfDigit;
polSVMScore = GroupPolC;
end