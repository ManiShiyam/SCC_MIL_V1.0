% function cmd = getLibLinearOptPara(labelsTrain, featuresTrain)
% bestcv = 0;
% w = getW(labelsTrain);
% for log2c = -3:2:6
%     cmd = [w, ' -v 3 -c ', num2str(2^log2c), ' -q'];
%     
%     cv = linearTrain(double(labelsTrain), sparse(double(featuresTrain)), cmd);
%     if (cv > bestcv),
%       bestcv = cv; 
%       bestc = log2c; 
%     end
% end
% 
% fprintf('best c=%g\n', bestc);
% cmd = [w, ' -c ', num2str(2^bestc)];
% end


%% ----------------------------------
% OPTIMIZE Mean Class Accuracy
% ----------------------------------

function bestc = getLibLinearOptPara(labelsTrain, featuresTrain)
fprintf('Learning SVM optimal paramters : crossvalidation\n');
w = getW(labelsTrain);

bestcv = 0;
allLbls = unique(labelsTrain);

CV_DATA = subsample(labelsTrain, 3);
for log2c = -5:1:2
    cmd = [w, ' -c ', num2str(2^log2c), ' -q'];
        
    lblP = {};
    lblA = {};
    for k=1:length(CV_DATA)
        modelSVM = train(double(labelsTrain(CV_DATA{k}.trainIdx)), sparse(featuresTrain(CV_DATA{k}.trainIdx, :)), cmd);
        [pLabels, ~, ~] = predict(double(labelsTrain(CV_DATA{k}.testIdx)), sparse(featuresTrain(CV_DATA{k}.testIdx, :)), modelSVM, ' -q');
        lblP{end+1} = pLabels;
        lblA{end+1} = labelsTrain(CV_DATA{k}.testIdx);
    end
    lblP = cat(1, lblP{:});
    lblA = cat(1, lblA{:});
    [~, cv, ~] = getMACandCM(allLbls, lblP, lblA);
    fprintf('log2c=%g, cvMCA = %0.2f\n', 2^log2c, cv);

    if (cv > bestcv),
      bestcv = cv; 
      bestc = 2^log2c; 
    end
end

fprintf('best c=%g, best cv = %0.2f\n', bestc, bestcv);
cmd = [w, ' -c ', num2str(bestc)];


% cmd = [w, ' -c ', num2str(0.5)];
fprintf('\n');
end



%%
function CV_DATA = subsample(classLbl, nfold)
dlbl = unique(classLbl);
nclass = length(unique(dlbl));
cross_idx = zeros(length(classLbl), 1);

for cn = 1:nclass
    idx_imgfromClass_i = find(classLbl==dlbl(cn));
    cvIdx = crossvalind('Kfold', length(idx_imgfromClass_i), nfold);
    cross_idx(idx_imgfromClass_i) = cvIdx;
end

CV_DATA = {};
for k = 1:nfold
    CV_DATA{k}.trainIdx = cross_idx~=k;
    CV_DATA{k}.testIdx = cross_idx==k;
end
end