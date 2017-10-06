function [accOverAll, accMCA, Cp] = getMACandCM(allLbls, pLabels, lbls)
allLbls = sort(allLbls);
if size(allLbls,1) ~=1
    allLbls = allLbls';
end


accOverAll = length(find(pLabels == lbls))/length(lbls)*100;

[C, order] = confusionmat(lbls, pLabels, 'order', allLbls);

totImagesInEachClass = zeros(length(allLbls),1);
for k = 1:length(allLbls)
    totImagesInEachClass(k) = length(find(lbls==allLbls(k)));
end
totImagesInEachClass = repmat(totImagesInEachClass, 1, length(totImagesInEachClass));
Cp = C./totImagesInEachClass*100;
Cp(isnan(Cp)) = 0;

accMCA = mean(diag(Cp));
end