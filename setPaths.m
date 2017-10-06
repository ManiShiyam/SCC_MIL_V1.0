function inputFolder = setPaths()
inputFolder = 'E:\NewMachine\CODES\SCC_MIL_V1.0';

% Comment the following 2 lines if you are using random initializations to
% initialize sub-category classifiers
% addpath(genpath('E:\USEFULCODES\vlfeat-0.9.20'));
% addpath('E:\USEFULCODES\liblinear-2.1\matlab');

% uncomment the following if you want to calculate AUC and precision and
% recall - used in test_retinopathy_UCSB.m
% addpath(genpath('E:\USEFULCODES\vlfeat-0.9.20'));


addpath(genpath(inputFolder));
cd(inputFolder);
end
