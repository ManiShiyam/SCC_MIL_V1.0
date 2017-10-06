function MU_ret = initMU(dataset, X, K)
fprintf('Initializing sub-category classifiers...\n ');
savedDir = fullfile('.\TMP_FILES\MU_SVM', dataset);
if ~isdir(savedDir)
    mkdir(savedDir);
end
fn = fullfile(savedDir, [num2str(K), '.mat']);
if exist(fn, 'file')
    load(fn);
    return;
end

MU = vl_kmeans(X, K, 'Initialization', 'PLUSPLUS', 'NumRepetitions', 10, 'algorithm', 'elkan');
if K <= 2 
    MU_ret = MU;
    return;
end
[~, y] = min(vl_alldist(MU, X), [], 1);

%%
histVal = zeros(size(MU,2), 1);
histVal = vl_binsum(histVal, ones(size(y)), y);
idx = find(histVal < 50);
if length(idx)>1
    tmp = MU(:, idx);
    MU(:, idx) = [];
    [~, y] = min(vl_alldist(MU, X), [], 1);
end


plattModel = learnPPlatt(X', y');

MU_ret = {};
for k = 1:size(MU, 2)
    MU_ret{end+1} = [-plattModel.para(k, 2)*plattModel.modelSVM.w(plattModel.modelSVM.Label==k, :), -plattModel.para(k, 3)]';
end
MU_ret = cat(2, MU_ret{:});

if length(idx)>1
    K2 = size(MU_ret, 1) - size(MU, 1);
    if K2>0
        tmp = [tmp; ones(K2, size(tmp,2))];
    end
    MU_ret = [MU_ret, tmp];
end
save(fn, 'MU_ret');
end


%%
function plattModel = learnPPlatt(X, y)
fprintf('data size for training svm: %i x%i\n', size(X,1), size(X,2));

if ~isempty(find(y <= 0))
    error('IT MAY NOT SUPPORT');
end
bestc = getLibLinearOptPara(y, X);
 
CV_DATA = subsample_kfold(y, 3);
classes = unique(y);
nclass = length(classes);

aL = zeros(size(y));
dvArr = zeros(length(y), nclass);


for k = 1:length(CV_DATA)
    fprintf('Learning PLATT itr %i\n', k);
    trIdx = CV_DATA{k}.trainIdx;
    cmd = [' -c ', num2str(bestc), ' ', getW(y(trIdx))];
    modelSVM = train(y(trIdx), sparse(X(trIdx, :)), cmd);
    teIdx = CV_DATA{k}.testIdx;
    [~, ~, dv] = predict(y(teIdx), sparse(X(teIdx, :)), modelSVM);
    aL(teIdx) = y(teIdx);
    
    for ll = 1:nclass
        cno = classes(ll);
        if ~isempty(find(modelSVM.Label==cno))
            tmp = dv(:, modelSVM.Label==cno);
            dvArr(teIdx, cno) = tmp;
        end
    end
end

plattParas = zeros(nclass, 3);
for ll = 1:nclass
    cno = classes(ll);
    [A, B] = learnPlatt_2(aL, dvArr(:, cno), cno);
    plattParas(ll, :) = [cno, A, B];
end


%% final train
fprintf('Training SVM\n');
cmd = [' -c ', num2str(bestc), ' ',  getW(y)];
modelSVM = train(y, sparse(X), cmd);
% [pl, acc, dv] = predict(y, sparse(X), modelSVM);

plattModel = struct;
plattModel.para = plattParas;
plattModel.modelSVM = modelSVM;
end


function [A, B] = learnPlatt_2(y, dvArr, cno)
lbl = ones(size(y));
lbl(y~=cno) = -1;

n = min(length(find(lbl==1)), length(find(lbl~=1)));
n = max(n, 50);
negIdx = find(lbl==-1);
posIdx = find(lbl==1);
negIdx = negIdx(randperm(length(negIdx)));
if length(negIdx) > n
    negIdx = negIdx(1:n);
end
posIdx = posIdx(randperm(length(posIdx)));
if length(posIdx) > n
    posIdx = posIdx(1:n);
end
idx = [posIdx; negIdx];

lbl = lbl(idx);
dvArr = dvArr(idx);
[A, B] = platt(dvArr, lbl, length(find(lbl==-1)), length(find(lbl==1)));
end



