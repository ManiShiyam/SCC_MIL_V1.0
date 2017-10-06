function CV_DATA = subsample_kfold(lbls, nfold)
uniqueLbls = unique(lbls);
nclass = length(unique(uniqueLbls));
cross_idx = zeros(length(lbls), 1);

for cn = 1:nclass
    idx_imgfromClass_i = find(lbls==uniqueLbls(cn));
    cvIdx = crossvalind('Kfold', length(idx_imgfromClass_i), nfold);
    cross_idx(idx_imgfromClass_i) = cvIdx;
end

CV_DATA = {};
for k = 1:nfold
    CV_DATA{k}.trainIdx = find(cross_idx~=k);
    CV_DATA{k}.testIdx = find(cross_idx==k);
    
    trlbl = lbls(CV_DATA{k}.trainIdx);
    telbl = lbls(CV_DATA{k}.testIdx);
    if length(unique(trlbl))~=nclass || length(unique(telbl))~=nclass
        error('some problems in the cross-validation indices');
    end
end
end