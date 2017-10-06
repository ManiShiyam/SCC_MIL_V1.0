function [cost, dLdR] = caldL_dr(r, MU, W, DATA, para)
[cost, ~, hl] = getCost(DATA, W, MU, r, para);
if nargout == 1
    return;
end

[P, pij, ~] = getP(DATA, MU, W, r);

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

term_soft_max_r = pij.^r;
term_log = term_soft_max_r.*log(pij);


dLdR = 0;
for i = 1:N
    idx_i = DATA.low(i):DATA.up(i);
    tmp = log(P(i)) - sum(term_log(idx_i))/sum(term_soft_max_r(idx_i));
    dLdR = dLdR - P(i)/r * tmp * mulTerm(i); 
end

if isnan(dLdR) || isinf(dLdR)
    dLdR = 0;
end
end