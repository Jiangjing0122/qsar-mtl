clc
clear

% The SVM model used LIBSVM library which is available at: https://www.csie.ntu.edu.tw/~cjlin/libsvm/.
addpath('./libsvm_matlab');

% The RF models used the randomforest-matlab library, which was developed by Abhishek Jaiantilal, and 
% the codes are available at: https://code.google.com/p/randomforest-matlab.
addpath('./RF_MexStandalone-v0.02-precompiled/randomforest-matlab/RF_Class_C');

% The MTL models employed are based on the MALSAR library developed by Zhou
% et al., and the source codes of the library are available on Github at: https://github.com/jiayuzhou/MALSAR.
addpath('./MALSAR/utils/')
addpath('./MALSAR/functions/Lasso/')
addpath('./MALSAR/functions/dirty/'); 
addpath('./MALSAR/c_files/prf_lbm/'); 
addpath('./MALSAR/functions/joint_feature_learning/');
addpath('./MALSAR/functions/CMTL/');
addpath('./MALSAR/functions/low_rank/');


% the variable 'split' is used to split data
split{1} = [314,100,110,112]; %group1
split{2} = [500,122,132,128]; %group2
split{3}= [118,222,224]; %group3
split{4}= [978,976,984]; %group4
split{5} = [186,150,44]; %group5
split{6} = [2424,1578,102]; %group6

% the variables 'l1' and 'l2' are used by different MTL models
% they need to be optimized by different data

%% load data
g=1;
datafile = strcat('data\group', num2str(g),'.mat');
load(datafile);

%% preprocess data
% change the compound activity lables to '1' and '-1' for consistency
for s=1:size(data,1)
    if data(s,end)==0
        data(s,end)=-1;
    end
end

% split the data for multiple tasks and order the samples of each task randomly
msizes=split{g};
start=1;
finish=0;
for i=1:length(msizes)
    finish=start+msizes(i)-1;
    A=data(start:finish,:);
    rowrank = randperm(size(A, 1)); % random numbers
    split_data{i} = A(rowrank, :);
    start=finish+1;
end

% data normalization
for t = 1: length(msizes)
    split_data{t}(:,1:end-1) = zscore(split_data{t}(:,1:end-1));
end

%prepare indices for 5-fold cross validation
fold=5;
for j=1:length(msizes)
    indice = crossvalind('Kfold',size(split_data{j},1),fold);
    indices{j}=indice;
end
mtlIndices=indices;

%% the variable 'resutl' is used to store F1 values of different models
result=zeros(length(msizes),7);

%% multi-task learning (MTL)
% prepare X and Y for the MTL training and test
for tt = 1: length(msizes)
    X{tt} = split_data{tt}(:,1:end-1);
    Y{tt} = split_data{tt}(:,end);
end

% build and evaluate mtl-lasso model
least_lasso_f1 = mtl_least_lasso(X, Y, mtlIndices, fold, 20, 100);

% save mtl-lasso results
for i=1:length(msizes)
    result(i,5)=least_lasso_f1(i);
end

% build and evaluate mtl-trace model
least_trace_f1 = mtl_least_trace(X, Y, mtlIndices, fold, 190, 10);

% save mtl-trace results
for i=1:length(msizes)
    result(i,6)=least_trace_f1(i);
end

% build and evaluate mtl-l21 model
least_l21_f1 =mtl_least_l21(X, Y, mtlIndices, fold, 23, 122);

% save mtl-l21 results
for i=1:length(msizes)
    result(i,7)=least_l21_f1(i);
end

%% single task learning using the same data
for i=1:length(msizes)
    singleTaskData = split_data{i};
    result(i,1)=i;
    
    % BPNN
    bp_f1 = bp_activity(singleTaskData,indices{i}, fold, g);
    result(i,2)=bp_f1;
    
    % SVM
    svm_f1 =svm_activity(singleTaskData,indices{i}, fold, g);
    result(i,3)=svm_f1;
    
    % Random Forest
    rf_f1=rf_activity(singleTaskData, indices{i}, fold, g);
    result(i,4)=rf_f1;
    
    rocinfo = sprintf( 'Task %d: BP_F1=%g SVM_F1=%g RF_F1=%g\n',i, bp_f1, svm_f1, rf_f1);
    disp(rocinfo);
end

%% Save results
resultsFile = strcat('results\group', num2str(g),"_results");
save(resultsFile,'result');
