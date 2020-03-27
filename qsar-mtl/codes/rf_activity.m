function [f1_mean, YP] = rf_activity(data, indices, fold, g)

%tree_num needs be optimized by different data
tree_num=8;

for f=1:fold
    test_data=(indices==f);
    train_data=~test_data;
    
    trainX=data(train_data,1:end-1);
    trainY=data(train_data,end);
    
    testX=data(test_data,1:end-1);
    testY=data(test_data,end);
    
    % training
    model = classRF_train(trainX,trainY);
    
    % prediction
    predictY = classRF_predict(testX,model,tree_num);
    f1_value(f)=eval_f1(testY,predictY);
    YP{f}=predictY;
end
f1_mean=mean(f1_value);
end