function [cost, dLdMU] = caldL_dMU(MU, W, r, DATA, para)
d = size(DATA.x,1);
K = para.K;
MU = reshape(MU, [d, K]);
[cost, ~, hl] = getCost(DATA, W, MU, r, para);
if nargout == 1
    return;
end
[P, pij, zij] = getP(DATA, MU, W, r);

%%
N = length(DATA.y);
idx_p = DATA.idx_p;
idx_n = DATA.idx_n;
mulTerm = zeros(1, N);
if strcmp(para.loss, 'ce')
    mulTerm(idx_p) = -P(idx_p)*length(idx_p);
    mulTerm(idx_n) = (1 - P(idx_n))*length(idx_n);
    mulTerm = 1./mulTerm;
elseif strcmp(para.loss, 'mm')
    mulTerm(idx_p) = length(idx_p);
    mulTerm(idx_n) = length(idx_n);
    hl = -2*hl.*DATA.y;
    mulTerm = hl'./mulTerm;
end
mulTerm = para.lambda*mulTerm;

%%
idx_i = cell(N,1);
for i = 1:N
    idx_i{i} = DATA.low(i):DATA.up(i);
end

term_soft_max = pij.^r;
tmp = term_soft_max.*(1 - pij);
zij = zij.*(1-zij);

for i = 1:N
    if mulTerm(i) ~= 0
        mulTerm(i) = P(i)*mulTerm(i)/sum(term_soft_max(idx_i{i}));
    end
end
        
X = DATA.x;
dLdMU = zeros(size(MU));
% if para.K > 20
%     parfor k = 1:para.K
%         dPidMUk = W(k)*zij(k, :).*tmp;
%         dPidMUk = bsxfun(@times, X, dPidMUk);
% 
%         term1 = zeros(d, 1);
%         for i = 1:N
%             term1 = term1 + mulTerm(i) * sum(dPidMUk(:, idx_i{i}), 2); 
%         end
%         dLdMU(:, k) = term1;
%     end
% else
    for k = 1:para.K
        dPidMUk = W(k)*zij(k, :).*tmp;
        dPidMUk = bsxfun(@times, X, dPidMUk);

        term1 = zeros(d, 1);
        for i = 1:N
            if mulTerm(i) ~= 0
                term1 = term1 + mulTerm(i) * sum(dPidMUk(:, idx_i{i}), 2); 
            end
        end
        dLdMU(:, k) = term1;
    end
% end
dLdMU = reshape(dLdMU, d*K, 1);


% if ~isempty(find(isnan(dLdMU)))
    dLdMU(isnan(dLdMU)) = 0;
    dLdMU(isinf(dLdMU)) = 0;
% end
end