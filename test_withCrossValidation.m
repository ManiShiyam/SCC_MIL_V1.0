function test_withCrossValidation()
%% load data
dataset = 'retinopathy_CVPR_2015';
[DATA_train, DATA_test, DATA_train_CV, DATA_test_CV] = loadRetinopathyDataset_CVPR2015();
DATA_train = DATA_train{1};
DATA_test = DATA_test{1};

%% parameters
r = 1;
para.loss = 'mm'; % 'mm'-max-margin or 'ce' - cross entropy
para.K = 10;
para.lambda = 1e2;
para.svm_gamma = 0.1;
para.MU_initialization = 'rand'; % 'rand' or 'platt'

%% init sub-category classifiers
if strcmp(para.MU_initialization, 'rand')
    MU_init = rand(size(DATA_train.x, 1), para.K);
else
    MU_init = initMU(dataset, DATA_train.x, para.K);
end

%% learn para
[para.lambda, para.svm_gamma, r] = learnLambda(DATA_train_CV, DATA_test_CV, r, MU_init, para);

%% train
[W, MU, r_learned] = learnDict(DATA_train, r, MU_init, para);

%% test
[acc, mca, al, pl] = testImgs(DATA_test, W, MU, r_learned);
end



function [acc, mca, al, pl] = testImgs(DATA, W, MU, r)
% add bias
d = size(MU, 1) - size(DATA.x, 1);
[~, ninst] = size(DATA.x);
DATA.x = [DATA.x; ones(d, ninst)];

[P, pij, aij] = getP(DATA, MU, W, r);
pl = ones(size(DATA.y))*DATA.neg;
pl(P > 0.5) = DATA.pos;
[acc, mca, CM] = getMACandCM([DATA.pos, DATA.neg], pl, DATA.y);
al = DATA.y;
end


%%
function [bestLambda, bestGamma, bestR] = learnLambda(DATA_train, DATA_test, r, MU_init, para)
rArr = [1, 2, 5];
gammaArr = [0.1, 0.3, 0.5];
lamArr = [1e2, 1e3, 1e4];

bestAcc = 0;
AccArr = [];
for r = rArr
    for g = gammaArr
        para.svm_gamma = g;
        for lam = lamArr
            para.lambda = lam;
            meanAcc = 0;
            for f = 1:length(DATA_train)
                [W, MU, r_learned] = learnDict(DATA_train{f}, r, MU_init, para);
                [acc, mca, al, pl] = testImgs(DATA_test{f}, W, MU, r_learned);
                meanAcc = meanAcc + mca;
            end
            meanAcc = meanAcc/length(DATA_train);        

            AccArr(end+1, :) = [r, g, find(lamArr==lam), meanAcc]
            fprintf('r = %0.2f, gamma = %0.2f, lambda = %0.2f, acc = %0.2f\n\n', r, g, lam, meanAcc);
            if meanAcc > bestAcc
                bestLambda = lam;
                bestGamma = g;
                bestR = r;
                bestAcc = meanAcc;
            end   
        end
    end
end
fprintf('bestR = %0.2f, bestGamma = %0.2f, bestLambda = %0.2f, bestAcc = %0.2f\n\n', bestR, bestGamma, bestLambda, bestAcc);
end