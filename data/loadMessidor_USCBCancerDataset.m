function [DATA_all, DATA_train, DATA_test] = loadMessidor_USCBCancerDataset(dataset)
fn = ['.\TMP_FILES\',dataset, '_cv_idx.mat'];
if exist(fn, 'file')
    load(fn);
else
    [DATA_all, DATA_train, DATA_test] = loadRetinopathyDataset2(dataset);
    save(fn, 'DATA_all', 'DATA_train', 'DATA_test');
end
end

%%
function [DATA_all, DATA_train, DATA_test] = loadRetinopathyDataset2(dataset)
if strcmp(dataset, 'messidor')
    load('.\data\messidor.mat');
    nfold = 2;
    ntimes = 10;
elseif strcmp(dataset, 'ucsb_breast')
    load('.\data\ucsb_breast.mat');   
    nfold = 4;
    ntimes = 10;
else
    error('wrong arg');
end
x.data(isnan(x.data)) = 0;
x.data = dataNormalize(x.data); % NxD
x.data = [x.data, ones(size(x.data, 1), 1)]; %% add bias

bagIDs = unique(x.ident.milbag);
bags = {};
y = zeros(length(bagIDs), 1);
for b = 1:length(bagIDs)
    idx = find(x.ident.milbag == bagIDs(b));

    y(b) = unique(x.nlab(idx));
    if y(b) ~= 1
        y(b) = -1;
    end
    
    aBag = struct;
    aBag.fea = x.data(idx, :);
    aBag.y = y(b);
    bags{end+1} = aBag;
end

DATA_all = loadData2(bagIDs, bags);

DATA_train = {};
DATA_test = {};
for t = 1:ntimes
    CVIdx = crossvalind('Kfold', length(y), nfold);
    for k = 1:nfold    
        DATA_train{end+1} = loadData2(bagIDs(CVIdx ~= k), bags);
        DATA_test{end+1} = loadData2(bagIDs(CVIdx == k), bags);
    end
end
end

%%
function DATA = loadData2(bagIds, bags)
nbag = length(bagIds);
low = zeros(nbag, 1);
up = zeros(nbag, 1);
y = zeros(nbag, 1);
fea = {};
instLbls = {};

k = 1;
for b = 1:nbag
    aBag = bags{bagIds(b)};
    nInst = size(aBag.fea, 1);
    
    fea{end+1} = aBag.fea;
    instLbls{end+1} = ones(nInst, 1)*aBag.y;
    
    y(b) = aBag.y;
    low(b) = k;
    up(b) = k + nInst - 1;
    k = k + nInst;    
end
fea = cat(1, fea{:});
fea(isnan(fea)) = 0;
instLbls = cat(1, instLbls{:});


pos = 1;
neg = -1;
y(y~=pos) = neg;
instLbls(instLbls~=pos) = neg;

DATA = struct;
DATA.x = (fea'); % d x N
DATA.y = y;
DATA.low = low;
DATA.up = up;
DATA.instLbls = instLbls;
DATA.idx_p = find(y==pos);
DATA.idx_n = find(y~=pos);
DATA.pos = pos;
DATA.neg = neg;
DATA.classNos = unique(y);
end



function x = dataNormalize(x)
mv = mean(x, 1);
x = bsxfun(@minus, x, mv);

stdv = std(x, [], 1);
idx = find(stdv == 0);
stdv = 1./stdv;
stdv(isnan(stdv)) = 0;
stdv(isinf(stdv)) = 0;
x = bsxfun(@times, x, stdv);

x(:, idx) = [];
end
