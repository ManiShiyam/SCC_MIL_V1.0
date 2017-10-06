function [DATA_train, DATA_test, DATA_train_CV, DATA_test_CV] = loadRetinopathyDataset_CVPR2015()
load('.\data\DR_data.mat');
useNormalization = 1;
allBags = cat(2, bag, testBags);
[~, mv, stdv] = loadData2(allBags, labels, useNormalization, [], []);

[DATA_train{1}, ~, ~] = loadData2(bag, labels, useNormalization, mv, stdv);
[DATA_test{1}, ~, ~] = loadData2(testBags, testlabels, useNormalization, mv, stdv);

ntimes = 2;
nfold = 3;
DATA_train_CV = {};
DATA_test_CV = {};
for t = 1:ntimes
    CVIdx = crossvalind('Kfold', length(labels), nfold);
    for k = 1:nfold    
        [DATA_train_CV{end+1}, ~, ~] = loadData2(bag(CVIdx ~= k), labels(CVIdx ~= k), useNormalization, mv, stdv);
        [DATA_test_CV{end+1}, ~, ~] = loadData2(bag(CVIdx == k), labels(CVIdx == k), useNormalization, mv, stdv);
    end
end
end

%%
function [DATA, mv, stdv] = loadData2(bags, labels, useNormalization, mv, stdv)
nbag = length(labels);
low = zeros(nbag, 1);
up = zeros(nbag, 1);
y = zeros(nbag, 1);
fea = {};
instLbls = {};

k = 1;
for b = 1:nbag
    aBag = bags{b};
    nInst = size(aBag, 2);
    
    fea{end+1} = aBag;
    instLbls{end+1} = ones(nInst, 1)*labels(b);
    
    y(b) = labels(b);
    low(b) = k;
    up(b) = k + nInst - 1;
    k = k + nInst;    
end
fea = cat(2, fea{:});
if useNormalization
    [fea, mv, stdv] = dataNormalize(fea, mv, stdv);
end
fea = [fea; ones(1, size(fea, 2))]; %% add bias



fea(isnan(fea)) = 0;
instLbls = cat(1, instLbls{:});


pos = 1;
neg = -1;
y(y~=pos) = neg;
instLbls(instLbls~=pos) = neg;

DATA = struct;
DATA.x = fea; % d x N
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



function [x, mv, stdv] = dataNormalize(x, mv, stdv)
if nargin == 1 || isempty(mv)
    mv = mean(x, 2);
end
x = bsxfun(@minus, x, mv);
if nargin == 1 || isempty(stdv)
    stdv = std(x, [], 2);
end
idx = find(stdv == 0);
stdv2 = 1./stdv;
stdv2(isnan(stdv2)) = 0;
stdv2(isinf(stdv2)) = 0;

x = bsxfun(@times, x, stdv2);
x(idx, :) = [];

%% l2 and power
% x = sign(x).*sqrt(abs(x));
% sv = sqrt(sum(x.^2, 1));
% sv = 1./sv;
% x = bsxfun(@times, x, sv);
% x(isnan(x)) = 0;
% x(isinf(x)) = 0;
% 
% sv = sum(x, 1);
% sv = 1./sv;
% sv(isnan(sv)) = 0;
% x = bsxfun(@times, x, sv);
end

% function x = dataNormalize(x)
% for d = 1:size(x, 1)
%     tmp = x(d, :);
%     minv = min(tmp);
%     maxv = max(tmp);
%     tmp = (tmp - minv)./(maxv - minv);
% %     tmp = (tmp - mean(tmp))./std(tmp);
%     x(d, :) = tmp;
% end
% x(isnan(x)) = 0;
% end
