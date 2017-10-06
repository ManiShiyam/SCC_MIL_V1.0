function [W, MU, r] = learnDict(DATA, r, MU, para)
% add bias
d = size(MU, 1) - size(DATA.x, 1);
[~, ninst] = size(DATA.x);
DATA.x = [DATA.x; ones(d, ninst)];

%% init W 
W = zeros(para.K+1, 1);

%% learn
% figure; hold on;
% win(1) = subplot(1, 2, 1);
% win(2) = subplot(1, 2, 2);

costArr = [];
[cost, mca] = getCost(DATA, W, MU, r, para);
% costArr(end + 1, :) = [cost, mca];
fprintf('%i %0.5f\t%0.2f, minMU = %0.2f, maxMU = %0.2f\n', 0, cost, mca, min(MU(:)), max(MU(:)));
for k = 1:10
    W = learnW(W, MU, r, DATA, para);
    [cost, mca] = getCost(DATA, W, MU, r, para);
    costArr(end+1, :) = [cost, mca];
    fprintf('%i %0.5f\t%0.2f\n', k, cost, mca); 
%     plot(win(1), costArr(:, 1)); plot(win(2), costArr(:, 2)); pause(0.01);
    
    MU = learnMU(W, MU, r, DATA, para);
    [cost, mca] = getCost(DATA, W, MU, r, para);  
    costArr(end+1, :) = [cost, mca];
    fprintf('%i %0.5f\t%0.2f, minMU = %0.2f, maxMU = %0.2f\n', k, cost, mca, min(MU(:)), max(MU(:))); 
%     plot(win(1), costArr(:, 1)); plot(win(2), costArr(:, 2)); pause(0.01);
    
    r = learnR(W, MU, r, DATA, para);
    [cost, mca] = getCost(DATA, W, MU, r, para);  
    costArr(end+1, :) = [cost, mca];
    fprintf('%i %0.5f\t%0.2f, r = %0.2f\n', k, cost, mca, r);
%     plot(win(1), costArr(:, 1)); plot(win(2), costArr(:, 2)); pause(0.01);
    
    if k > 1 && convergenceTest(costArr(end), precost, 1e-5); %abs(costArr(end)- precost) <= para.lambda*1e-5 % 
        fprintf('converged...\n');
        break;
    end
    precost = costArr(end);
end
W = learnW(W, MU, r, DATA, para);
end

%% 
function r = learnR(W, MU, r, DATA, para)
fprintf('Learning r...\t\t');
opts.Display     = 'none';
opts.verbose     = false;
opts.TolFun      = 1e-5;
opts.MaxIter     = 200;
opts.Method      = 'lbfgs'; % for minFunc
opts.MaxFunEvals = 200;
opts.TolX        = 1e-5;
opts.progTol = 1e-5;

st = tic;
dLdR = @(r)caldL_dr(r, MU, W, DATA, para);
[r, opt.finalObj, opt.exitflag, opt.output] = minFunc(dLdR, r,opts);
fprintf(' time = %0.4f\t', toc(st));
end




function MU = learnMU(W, MU, r, DATA, para)
fprintf('Learning MU...\t\t');
opts.Display     = 'none';
opts.verbose     = false;
opts.TolFun      = 1e-5;
opts.MaxIter     = 200;
opts.Method      = 'lbfgs'; % for minFunc
opts.MaxFunEvals = 200;
opts.TolX        = 1e-5;
opts.progTol = 1e-5;

% profile on;
st = tic;
[d, K] = size(MU);
MU = reshape(MU, d*K, 1);
dLdMU = @(MU)caldL_dMU(MU, W, r, DATA, para);
[MU, opt.finalObj, opt.exitflag, opt.output] = minFunc(dLdMU, MU,opts);
MU = reshape(MU, [d, K]);
fprintf(' time = %0.4f\t', toc(st));
% profile viewer;
end


function W = learnW(W, MU, r, DATA, para)
fprintf('Learning W...\t\t');
opts.Display     = 'none';
opts.verbose     = false;
opts.TolFun      = 1e-5;
opts.MaxIter     = 200;
opts.Method      = 'lbfgs'; % for minFunc
opts.MaxFunEvals = 200;
opts.TolX        = 1e-5;
opts.progTol = 1e-5;

dLdW = @(W)caldL_dW(W, MU, r, DATA, para);
st = tic;
[W, opt.finalObj, opt.exitflag, opt.output] = minFunc(dLdW, W, opts);
fprintf(' time = %0.4f\t', toc(st));
end



