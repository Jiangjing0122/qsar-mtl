function [f1_mean,YP] = svm_activity (data, indices, fold, g)

for f=1:fold
    test_data=(indices==f);
    train_data=~test_data;
    
    trainX=data(train_data,1:end-1);
    trainY=data(train_data,end);
    
    testX=data(test_data,1:end-1);
    testY=data(test_data,end);
    
    % bestc and bestg need be optimized by different data
    bestc = 1;
    bestg = 1.0/size(data,2);
    
    
    cmd = ['-c ',num2str(bestc),' -g ',num2str(bestg)];
    
    %% training
    model = svmtrain(trainY, trainX, cmd);
    
    %% prediction
    [predict_label, a, decision_values] = svmpredict(testY, testX, model);
    
    f1_value(f)=eval_f1(testY,predict_label);
end
f1_value=rmmissing(f1_value);
f1_mean=mean(f1_value);
end