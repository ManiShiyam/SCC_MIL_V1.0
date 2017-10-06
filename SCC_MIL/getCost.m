function [cost, mca, hl] = getCost(DATA, W, MU, r, para)
[P, ~, ~] = getP(DATA, MU, W, r);
cost_reg_W = W.^2;
cost_reg_W = sum(cost_reg_W(:))/2;
if strcmp(para.loss, 'ce')
    cost = cost_reg_W - para.lambda/length(DATA.idx_p)*sum(log(P(DATA.idx_p))) - para.lambda/length(DATA.idx_n)*sum(log(1 - P(DATA.idx_n)));
    hl = 0;
elseif strcmp(para.loss, 'mm')
    hl = hingeLoss(P, para.svm_gamma, DATA.y);
    hl_2 = hl.^2;
    cost = cost_reg_W + para.lambda/length(DATA.idx_p)*sum(hl_2(DATA.idx_p)) + para.lambda/length(DATA.idx_n)*sum(hl_2(DATA.idx_n));
end

pl = ones(size(DATA.y))*DATA.neg;
pl(P >= 0.5) = DATA.pos;
[acc, mca, ~] = getMACandCM([DATA.pos, DATA.neg], pl, DATA.y);

if isnan(cost) || ~isempty(find(isnan(P),1))
    cost = realmax;
end
% fprintf('cost_Reg = %0.2f, cost = %0.5f, acc = %0.2f, mca = %0.2f\n', cost_reg, cost, acc, mca);
end


function hl = hingeLoss(P, gamma, y)
hl = gamma + y.*(0.5 - P');
hl = max(0, hl);
end