function [cost, dL_dW] = caldL_dW(W, MU, r, DATA, para)
[cost, ~, hl] = getCost(DATA, W, MU, r, para);
if nargout == 1
    return;
end


[P, pij, zij] = getP(DATA, MU, W, r);

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
term_soft_max = pij.^r;
tmp = term_soft_max.*(1 - pij); % Nx1
tmp = bsxfun(@times, zij, tmp); % (K+1)xN  , Nx1

dL_dW = W;
for i = 1:N
    idx_i = DATA.low(i):DATA.up(i);
    sum_tsm = sum(term_soft_max(idx_i));
    dL_dW = dL_dW + P(i)/sum_tsm * sum(tmp(:, idx_i), 2) * mulTerm(i);
end

% if length(find(isnan(dL_dW)))>0 || length(find(isinf(dL_dW)))>0
    dL_dW(isnan(dL_dW)) = 0;
    dL_dW(isinf(dL_dW)) = 0;
% end
end