function zij = getEmbedding(x, MU)
zij = 1./(1 + exp(-MU'*x));%KxN
zij = [zij; ones(1, size(zij, 2))]; % (K+1) x N
zij(isinf(zij)) = 0;
end

