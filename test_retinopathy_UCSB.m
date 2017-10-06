function test_retinopathy_UCSB()
%% load data
[DataAll, DATA_train, DATA_test] = loadUCSBCancerDataset();

%% parameters
r = 1;
para.loss = 'mm'; % 'mm'-max-margin or 'ce' - cross entropy
para.K = 5;
para.lambda = 1e2;
para.svm_gamma = 0.1;

%% train and test
MU_init = rand(size(DataAll.x, 1), para.K);
accArr = [];
for foldNo = 1:length(DATA_train)
    % train
    [W, MU, r_learned] = learnDict(DATA_train{foldNo}, r, MU_init, para);
    % test
    [accTest, mcaTest, AUCTest, EERtest, AUC_PR, F1score] = testImgs(DATA_test{foldNo}, W, MU, r_learned);
    accArr(end+1, :) = [mcaTest, accTest, AUCTest, EERtest, AUC_PR, F1score]
end
meanAcc = mean(accArr)
stdAcc = std(accArr)  
end



function [acc, mca, AUC, EER, AUC_PR, F1score] = testImgs(DATA, W, MU, r)
% add bias
d = size(MU, 1) - size(DATA.x, 1);
[~, ninst] = size(DATA.x);
DATA.x = [DATA.x; ones(d, ninst)];

[P, pij, aij] = getP(DATA, MU, W, r);
pl = ones(size(DATA.y))*DATA.neg;
pl(P >= 0.5) = DATA.pos;
[acc, mca, CM] = getMACandCM([DATA.pos, DATA.neg], pl, DATA.y);

% posclass = DATA.pos;
% [X,Y,T,AUC] = perfcurve(DATA.y,P,posclass);
[~, ~, info] = vl_roc(DATA.y,P);
AUC = info.auc;
EER = info.eer;

[rc, pr, info] = vl_pr(DATA.y,P) ;
AUC_PR = info.auc;


[CM, order] = confusionmat(DATA.y, pl, 'order', [DATA.pos, DATA.neg]);
pre = CM(1,1)/sum(CM(:, 1));
re = CM(1,1)/sum(CM(1,:));
F1score = 2*pre*re/(pre + re);
end

