function f1=mtl_least_trace(X, Y, indices, fold, L1, L2)
task_num=size(X,2);
results=zeros(fold,task_num);

for f=1:fold
    for t = 1: task_num
        clear test_data train_data
        
        test_data=(indices{t}==f);
        train_data=~test_data;
        
        X_tr{t}=X{t}(train_data,1:end-1);
        Y_tr{t}=Y{t}(train_data,end);
        
        X_te{t}=X{t}(test_data,1:end-1);
        Y_te{t}=Y{t}(test_data,end);
    end
    
    % training and prediction using least squares loss
    opts.init = 0;      % guess start point from data.
    opts.tFlag = 3;     % terminate after relative objective value does not changes much.
    opts.tol = 10^-3;   % tolerance.
    opts.maxIter = 1000; % maximum iteration number of optimization.
    opts.rho_L2=L2;
    
    W_pred = Least_Trace(X_tr, Y_tr, L1, opts);
    
    tn_val(f) = sum(svd(W_pred));
    rk_val(f) = rank(W_pred);
    
    clear output predict
    for i = 1: t
        output{i}=X_te{i} * W_pred(:, i);
        for j=1:length(output{i})
            if(output{i}(j)>=0)
                predict{i}(j)=1;
            else
                predict{i}(j)=-1;
            end
        end
        predict_Y{i}=predict{i}';
        results(f,i)=eval_f1(Y_te{i},predict_Y{i});
    end
end
results=results(all(~isnan(results),2),:);
f1=mean(results);
end