function ws = getW(labelsTrain)
% labelsTrain = [1,1,1,1,1,1,1,1,1,1,2,2,2,2,2,3,3,3,3,3];
dl = unique(labelsTrain)';
tot = length(labelsTrain);
w = [];
for k = 1:length(dl)
    w = [w, tot/sum(labelsTrain==dl(k))];
end
w = w./max(w);
ws = '';
for k = 1:length(dl)
    ws = strcat(ws, ' -w', num2str(dl(k)), [' ', num2str(w(k))]);
end
end

