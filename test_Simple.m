function test_Simple()
%% load data
[DATA_train, DATA_test, ~, ~] = loadRetinopathyDataset_CVPR2015();
DATA_train = DATA_train{1};
DATA_test = DATA_test{1};

%% parameters
r = 1;
para.loss = 'mm'; % 'mm for'-max-margin or 'ce' for cross entropy
para.K = 100;
para.lambda = 1e2;
para.svm_gamma = 0.1;

%% train
MU_init = rand(size(DATA_train.x, 1), para.K); % random intialization
[W, MU, r_learned] = learnDict(DATA_train, r, MU_init, para);

%% test
[acc, mca, ~, ~] = testImgs(DATA_test, W, MU, r_learned);
acc
mca
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


[CM, order] = confusionmat(DATA.y, pl, 'order', [DATA.pos, DATA.neg]);
pre = CM(1,1)/sum(CM(:, 1));
re = CM(1,1)/sum(CM(1,:));
F1score = 2*pre*re/(pre + re);
end

