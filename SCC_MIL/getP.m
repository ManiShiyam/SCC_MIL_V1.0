function [P, pij, zij] = getP(DATA, MU, W, r)
zij = getEmbedding(DATA.x, MU); % KxN
pij = 1./(1 + exp(-W'*zij));

N = length(DATA.y);
P = zeros(1, N);
for i = 1:N
    idx_i = DATA.low(i):DATA.up(i);
    tmp = sum(pij(idx_i).^r)/length(idx_i);
    P(i) = tmp.^(1/r);   
end

pij(pij < eps) = eps;
pij(pij > 0.9999) = 0.9999;

P(P < eps) = eps;
P(P > 0.9999) = 0.9999;
end