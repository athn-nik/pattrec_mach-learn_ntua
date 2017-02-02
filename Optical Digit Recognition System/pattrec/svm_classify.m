function f = svm_classify( svmStruct, sample )
    if ~isempty(svmStruct.ScaleData)
        for c = 1:size(sample, 2)
            sample(:,c) = svmStruct.ScaleData.scaleFactor(c) * ...
                (sample(:,c) +  svmStruct.ScaleData.shift(c));
        end
    end
    f = svmdecision(sample,svmStruct);

end

